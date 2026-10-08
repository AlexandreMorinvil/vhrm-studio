"""DeepLabV3+ (CIHP human parsing, ONNX Runtime CUDA) and the skin refinement stage (notebook cells 6 and 9).

`keras-io/deeplabv3p-resnet50`, converted to ONNX with an in-graph argmax (input (N, 512, 512, 3) float32 RGB in
[-1, 1], output (N, 512, 512) uint8 CIHP class ids). Same session options as the notebook; do not change the batch
size or the cuDNN algorithm search for speed: both change a few class ids.
"""
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import crop_batch
from .refiner import SKIN_MASK_ERODE_PX

DEEPLAB_SIZE = 512        # fixed by the checkpoint
NUM_CLASSES = 20
INV_127_5 = 1.0 / 127.5
DEEPLAB_ONNX_PATH = Path(__file__).with_name('data') / 'deeplabv3p_resnet50_argmax.onnx'


class DeepLabSession:
    """ORT CUDA session bound to its own CUDA stream."""

    def __init__(self, device, onnx_path=DEEPLAB_ONNX_PATH):
        torch_lib = Path(torch.__file__).resolve().parent / 'lib'
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(str(torch_lib))  # cuDNN / cuBLAS DLLs for the ORT CUDA provider
        import onnxruntime as ort
        self.path = Path(onnx_path)
        if not self.path.is_file():
            raise FileNotFoundError(f'{self.path} not found')
        self.stream = torch.cuda.Stream(device=device)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options,
            # kSameAsRequested: 5.9 instead of 8.8 GB of arena at batch 8, identical outputs (the default doubles
            # each growth).
            providers=[('CUDAExecutionProvider', {'device_id': device.index or 0,
                                                  'cudnn_conv_algo_search': 'DEFAULT',
                                                  'arena_extend_strategy': 'kSameAsRequested',
                                                  'user_compute_stream': str(self.stream.cuda_stream)}),
                       'CPUExecutionProvider'])
        if self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError(f'ONNX Runtime did not activate CUDA: {self.session.get_providers()} '
                               '(install onnxruntime-gpu matching your CUDA version)')
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name
        if self.session.get_outputs()[0].type != 'tensor(uint8)':
            raise RuntimeError('Expected the argmax ONNX model with uint8 class-id output')
        self.ort_version = ort.__version__


class DeepLabStage:
    """DeepLab (ORT CUDA) and temporal skin refinement for one batch at a time.

    prepare() runs on the main stream (GPU crop + normalization). The rest runs on the session's stream, either
    inline (overlap=False) or on a worker thread (overlap=True) so that it overlaps the PEAR branch. io_binding=False
    reproduces the segmentation notebook's host round trip (numpy in, numpy out) for comparison.
    """

    def __init__(self, deeplab, batch_size, crop_size, refiner, device, io_binding=True, overlap=True):
        self.deeplab, self.session, self.refiner = deeplab, deeplab.session, refiner
        self.io_binding, self.overlap = io_binding, overlap
        self.crop_size = crop_size
        self.stream = deeplab.stream
        c, s = crop_size, DEEPLAB_SIZE
        self.x = torch.empty((batch_size, s, s, 3), dtype=torch.float32, device=device)
        self.ids_model = torch.empty((batch_size, s, s), dtype=torch.uint8, device=device)
        self.ids = self.ids_model if c == s else torch.empty((batch_size, c, c), dtype=torch.uint8, device=device)
        self.crop = torch.empty((batch_size, c, c, 3), dtype=torch.uint8, device=device)
        self.eroded = torch.empty((batch_size, c, c), dtype=torch.uint8, device=device)
        self.crop_host = torch.empty(self.crop.shape, dtype=torch.uint8, pin_memory=True)
        self.ids_host = torch.empty(self.ids.shape, dtype=torch.uint8, pin_memory=True)
        self.eroded_host = torch.empty(self.eroded.shape, dtype=torch.uint8, pin_memory=True)
        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.seconds = defaultdict(float)
        self.pending = None
        if overlap:
            self.jobs, self.results = queue.Queue(), queue.Queue()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def prepare(self, frames, inv_trans):
        """Main stream: RGB skin crop (rounded like uint8 cv2.warpAffine) and x / 127.5 - 1."""
        n, c, s = frames.shape[0], self.crop_size, DEEPLAB_SIZE
        crop = crop_batch(frames, inv_trans, c, 'round')
        self.crop[:n].copy_(crop.permute(0, 2, 3, 1))
        if c != s:
            crop = F.interpolate(crop, size=(s, s), mode='bilinear', align_corners=False).round_().clamp_(0, 255)
        torch.mul(crop.permute(0, 2, 3, 1), INV_127_5, out=self.x[:n])
        self.x[:n].sub_(1.0)
        self.n = n
        self.ready = torch.cuda.Event()
        self.ready.record()

    def submit(self):
        if self.overlap:
            self.jobs.put((self.n, self.ready))
        else:
            self.pending = self._run(self.n, self.ready)

    def result(self):
        """Waits for the batch; uploads the eroded masks on the current (main) stream."""
        out = self.results.get() if self.overlap else self.pending
        if isinstance(out, BaseException):
            raise RuntimeError('DeepLab stage failed') from out
        n = out['crop_class_ids'].shape[0]
        self.eroded[:n].copy_(self.eroded_host[:n], non_blocking=True)
        return out

    def close(self):
        if self.overlap and self.thread.is_alive():
            self.jobs.put(None)
            self.thread.join()

    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                break
            try:
                self.results.put(self._run(*job))
            except BaseException as exc:
                self.results.put(exc)

    def _run(self, n, ready):
        c, s = self.crop_size, DEEPLAB_SIZE
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            if self.io_binding:
                binding = self.session.io_binding()
                device_id = self.x.device.index or 0
                binding.bind_input(self.deeplab.input, 'cuda', device_id, np.float32, [n, s, s, 3], self.x.data_ptr())
                binding.bind_output(self.deeplab.output, 'cuda', device_id, np.uint8, [n, s, s],
                                    self.ids_model.data_ptr())
                self.session.run_with_iobinding(binding)
            else:
                ids = self.session.run([self.deeplab.output], {self.deeplab.input: self.x[:n].cpu().numpy()})[0]
                self.ids_model[:n].copy_(torch.from_numpy(ids))
            if c != s:
                self.ids[:n].copy_(F.interpolate(self.ids_model[:n, None].float(), size=(c, c),
                                                 mode='nearest')[:, 0].to(torch.uint8))
            t1 = time.perf_counter()
            self.crop_host[:n].copy_(self.crop[:n], non_blocking=True)
            self.ids_host[:n].copy_(self.ids[:n], non_blocking=True)
            self.stream.synchronize()
        t2 = time.perf_counter()
        crop, ids = self.crop_host[:n].numpy(), self.ids_host[:n].numpy()
        eroded = self.eroded_host[:n].numpy()
        skin_rgb = np.zeros((n, c, c, 3), dtype=np.uint8)
        skin_packed = np.empty((n, c, c // 8), dtype=np.uint8)
        for i in range(n):
            skin = self.refiner.refine_u8(crop[i], ids[i], cv2.COLOR_RGB2LAB)
            cv2.copyTo(crop[i], skin, skin_rgb[i])
            skin_packed[i] = np.packbits(skin, axis=-1)
            if SKIN_MASK_ERODE_PX > 0:
                cv2.erode(skin, self.kernel, dst=eroded[i], iterations=SKIN_MASK_ERODE_PX)
            else:
                eroded[i] = skin
        t3 = time.perf_counter()
        self.seconds['deeplab_ort'] += t1 - t0
        self.seconds['deeplab_d2h'] += t2 - t1
        self.seconds['skin_refine'] += t3 - t2
        return {'crop_skin_rgb': skin_rgb, 'crop_skin_mask_packed': skin_packed, 'crop_class_ids': ids.copy()}
