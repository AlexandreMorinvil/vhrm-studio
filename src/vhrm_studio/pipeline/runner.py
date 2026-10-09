"""PEAR SMPL-X UV mapping + DeepLabV3+ skin segmentation of one source into one HDF5 file.

The models load once (SkinUVPipeline), then run_video() processes any number of sources. The processing is the one
of pear_deeplab_skin_uv_video.ipynb (cell 12) with the same defaults; differences:

* frames can come from image sequences, and RGB videos are decoded as RGB (see frames.py),
* progress is reported through a callback and a run can be cancelled between batches,
* image sequences additionally store `source_frame_files` (the file of every row).

HDF5 layout: see README.md (identical to the notebook's, plus a few attributes).
"""
import contextlib
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import __version__
from .deeplab import DEEPLAB_SIZE, DeepLabSession, DeepLabStage
from .frames import SourcePlan
from .geometry import (FOCAL_LENGTH, RENDER_SIZE, PearUVBackprojector, SmplxTopology, UVTexelLayout,
                       apply_affine_batch, bbox_from_class_ids, crop_batch, gen_trans_from_patch_cv, lookup_nearest,
                       pack_mask_bits, process_bbox, raster_batch, raster_frames, sample_frame_colors, texel_geometry,
                       texels_to_atlas)
from .h5writer import SkinUVH5Writer
from .config import RUN_DEFAULTS
from .refiner import (CLASS_NAMES, COLORMAP, ENABLE_TEMPORAL_SKIN_COLOR_MODEL, SKIN_CLASS_IDS,
                      SKIN_COLOR_COV_REGULARIZATION, SKIN_COLOR_EVAL_SCALE, SKIN_COLOR_MAHALANOBIS_THRESHOLD,
                      SKIN_COLOR_MAX_REFERENCE_SAMPLES, SKIN_COLOR_MIN_REFERENCE_PIXELS, SKIN_COLOR_MODEL_EVERY,
                      SKIN_COLOR_REFERENCE_DISTANCE_PX, SKIN_COLOR_TEMPORAL_ALPHA, SKIN_MASK_ERODE_PX,
                      TemporalSkinRefiner)
from .temporal import TemporalParameterStabilizer

CROP_SIZE = 256           # PEAR input crop
BBOX_RATIO = 1.0          # shared by the PEAR and the skin crop
YOLO_CONF = 0.5
UV_TEXTURE_SIZE = 1024    # S
SKIN_CROP_SIZE = 512      # C; DeepLab always runs at 512 (other sizes are resized to 512 and back)
FILL_HOLES_PX = 0         # no hole filling: filled texels would be interpolated, not observed
PACKED_UV_LAYOUT_PATH = Path(__file__).with_name('data') / f'semantic_layout_{UV_TEXTURE_SIZE}.npz'
ATLAS_OUTPUTS = ('skin_rgb', 'skin_mask', 'visible_mask', 'class_ids')
H5_SUFFIX = '_skin_uv.h5'

class Cancelled(Exception):
    pass


class StageTimer:
    def __init__(self, enabled):
        self.enabled = enabled
        self.seconds = defaultdict(float)

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.seconds[name] += time.perf_counter() - t0


def load_yolo_model(model_path):
    """PEAR inference_images.load_yolo_model: full (non weights-only) torch.load for the ultralytics checkpoint."""
    from ultralytics import YOLO
    original_torch_load = torch.load

    def torch_load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_with_full_checkpoint
    try:
        return YOLO(str(model_path))
    finally:
        torch.load = original_torch_load


class SkinUVPipeline:
    """All models of the pipeline on one GPU (~10 GB with batch 8). log(text) receives the loading messages."""

    def __init__(self, assets_dir, checkpoint=None, yolo_weights=None, device='cuda:0', compile_backbone=True,
                 log=print):
        from ..assets import ASSETS_HELP, checkpoint_path, missing_assets, yolo_path
        from ..third_party.pear import build_ehm, build_ehm_pipeline, load_config

        self.log = log
        if not torch.cuda.is_available():
            raise RuntimeError('The skin-UV pipeline needs CUDA, but torch.cuda.is_available() returned False.')
        missing = missing_assets(assets_dir)
        if missing:
            raise FileNotFoundError(f'SMPL-X / FLAME assets incomplete in {assets_dir}: missing '
                                    f'{", ".join(missing[:6])}{" ..." if len(missing) > 6 else ""}. {ASSETS_HELP}')
        self.assets_dir = Path(assets_dir)
        self.device = device = torch.device(device)
        torch.set_float32_matmul_precision('high')  # run_video switches to 'highest' and restores this

        t0 = time.perf_counter()
        log('Loading the PEAR model...')
        self.topology = SmplxTopology(self.assets_dir / 'SMPLX').to(device).eval()
        cfg = load_config()
        self.ehm_model = build_ehm_pipeline(cfg)
        self.checkpoint = checkpoint_path(checkpoint, self.assets_dir)
        state = torch.load(str(self.checkpoint), map_location='cpu', weights_only=True)
        self.ehm_model.backbone.load_state_dict(state['backbone'], strict=False)
        self.ehm_model.head.load_state_dict(state['head'], strict=False)
        self.ehm_model = self.ehm_model.to(device).eval()
        del state
        log('Loading the SMPL-X / FLAME body model...')
        self.ehm = build_ehm(self.assets_dir).to(device).eval()
        self.ehm.smplx.smplx2flame_ind = torch.as_tensor(np.asarray(self.ehm.smplx.smplx2flame_ind), device=device)
        log('Loading the YOLO person detector...')
        self.yolo_weights = yolo_path(yolo_weights)
        self.detector = load_yolo_model(self.yolo_weights)
        log('Loading DeepLabV3+ (ONNX Runtime CUDA)...')
        self.deeplab = DeepLabSession(device)

        self.backbone_fp16 = copy.deepcopy(self.ehm_model.backbone).half()
        self.backbone_fp16.eval()  # ViT.train() returns None, so this cannot be chained
        # Only the FP16 copy runs. The FP32 weights (~2.5 GB) move to the CPU so that PEAR and DeepLab (~5.9 GB of
        # ORT memory at batch 8) fit in 12 GB; exceeding VRAM on Windows spills to system memory (several times
        # slower).
        self.ehm_model.backbone.to('cpu')
        self.backbone_run = self.backbone_fp16
        self.backbone_mode = 'eager FP16'
        if compile_backbone:
            from torch.utils._triton import has_triton
            if has_triton():
                self.backbone_run = torch.compile(self.backbone_fp16, dynamic=False)
                self.backbone_mode = 'torch.compile FP16'
            else:
                log('Triton is not available (pip install "triton-windows<3.5" on Windows): eager FP16 backbone.')
        torch.cuda.empty_cache()

        self.bp = PearUVBackprojector(self.topology, device, texture_size=UV_TEXTURE_SIZE, raster_size=512,
                                      render_size=RENDER_SIZE, crop_size=CROP_SIZE, depth_tol_frac=0.01)
        self.packed_uv_layout = UVTexelLayout(PACKED_UV_LAYOUT_PATH, self.bp, self.topology)
        log(f'Models ready in {time.perf_counter() - t0:.0f} s | UV texels per frame: '
            f'{self.bp.tex_flat.numel():,} standard, {self.packed_uv_layout.tex_flat.numel():,} packed | '
            f'backbone: {self.backbone_mode} | {torch.cuda.get_device_name(device)}')

    # ------------------------------------------------------------------ model wrappers
    def run_backbone(self, patch, batch_size):
        """FP16 ViT on (n, 3, 256, 256) patches in [0, 1]; pads to batch_size so the compiled graph is reused."""
        x = self.ehm_model.normalize(patch)[:, :, :, 32:-32].half()
        n = x.shape[0]
        if n < batch_size:
            x = torch.cat([x, x[-1:].expand(batch_size - n, -1, -1, -1)])
        return self.backbone_run(x.contiguous())[:n]

    def detect_person_bbox(self, frame_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        det = self.detector.predict(frame_bgr, device=str(self.device), classes=0, conf=YOLO_CONF, save=False,
                                    verbose=False)[0]
        boxes = det.boxes.xyxy.detach().cpu().numpy()
        if len(boxes) < 1:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes[int(np.argmax(areas))]
        return np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], dtype=np.float32)

    def _detect_batch_person_bbox(self, frames):
        """Return the latest detectable frame's bbox, its batch index and the number of YOLO calls."""
        n = frames.shape[0]
        if n < 1:
            raise ValueError('cannot detect a person in an empty batch')
        for index in range(n - 1, -1, -1):
            frame_rgb = frames[index].permute(1, 2, 0).to(torch.uint8).contiguous().cpu().numpy()
            bbox = self.detect_person_bbox(frame_rgb)
            if bbox is not None:
                return bbox, index, n - index
        return None, None, n

    # ------------------------------------------------------------------ one source
    def run_video(self, source, h5_path, settings=None, progress=None, should_stop=None, extra_attrs=None):
        """Process one source (a SourcePlan, or (kind, path) with kind 'video' | 'images') into h5_path.

        progress(state) is called after every batch with rows / done / ok / fps / elapsed; should_stop() is polled
        between batches and raises Cancelled when it returns True (h5_path is then incomplete). Returns stats."""
        cfg = {**RUN_DEFAULTS, **(settings or {})}
        log = self.log
        bp, device, ehm, ehm_model = self.bp, self.device, self.ehm, self.ehm_model
        packed_uv_layout, deeplab_session = self.packed_uv_layout, self.deeplab
        batch_size, max_frames = int(cfg['batch_size']), cfg['max_frames']
        overlap_deeplab, io_binding, accumulate = cfg['overlap_deeplab'], cfg['io_binding'], cfg['accumulate']
        packed_uv, texel_chunk_frames = cfg['packed_uv'], int(cfg['texel_chunk_frames'])
        packed_uv_outputs = tuple(cfg['packed_uv_outputs'])
        person_bbox_pad = float(cfg['person_bbox_pad'])
        if not set(packed_uv_outputs) <= set(ATLAS_OUTPUTS):
            raise ValueError(f'packed_uv_outputs must be a subset of {ATLAS_OUTPUTS}')
        S, C = bp.texture_size, SKIN_CROP_SIZE

        plan = source if isinstance(source, SourcePlan) else SourcePlan(
            source[0], source[1], target_fps=cfg['target_fps'], image_fps=cfg['image_fps'],
            ffmpeg_dir=cfg.get('ffmpeg_dir'))
        source_path = plan.path
        info, timestamps, sel = plan.info, plan.timestamps, plan.selection
        indices = sel['indices'] if not max_frames else sel['indices'][:int(max_frames)]
        n_rows = len(indices)
        temporal_config = copy.deepcopy(cfg['temporal'])
        temporal = TemporalParameterStabilizer(temporal_config, sel['effective_fps'], batch_size)
        rot_k = (info['rotation'] // 90) % 4
        stored_h, stored_w = info['height'], info['width']
        img_h, img_w = (stored_w, stored_h) if rot_k % 2 else (stored_h, stored_w)
        h5_path = Path(h5_path)
        stride_text = f", stride {sel['stride']}" if sel['mode'] == 'stride' else ''
        log(f"{source_path.name}: {img_w}x{img_h}, source {sel['source_fps']:.3f} fps ({len(timestamps)} frames, "
            f"{plan.decode_format}) | frame selection '{sel['mode']}'{stride_text} | effective "
            f"{sel['effective_fps']:.3f} fps | {n_rows} rows" + (f' (max {max_frames})' if max_frames else ''))

        frames_gpu = torch.zeros((batch_size, 3, img_h, img_w), dtype=torch.float32, device=device)
        refiner = TemporalSkinRefiner()
        deeplab = DeepLabStage(deeplab_session, batch_size, C, refiner, device, io_binding=io_binding,
                               overlap=overlap_deeplab)
        timer = StageTimer(cfg['profile_stages'])
        n_batches = 0

        def texel_stage(layout, raster, frames, pear_aff):
            with timer('uv_geometry'):
                valid, src_orig = texel_geometry(bp, raster, layout, pear_aff, (img_h, img_w))
            with timer('uv_colors'):
                colors = sample_frame_colors(frames, src_orig)
            return valid, src_orig, colors

        def atlas_stage(layout, prefix, valid, src_orig, colors, skin_fwd_t, start, outputs):
            """Skin / class lookups and the requested atlases of one layout for batch frames start:start + len(valid);
            its per-texel tensors are freed on return."""
            n, stop = valid.shape[0], start + valid.shape[0]
            with timer('uv_skin'):
                crop_xy = apply_affine_batch(src_orig, skin_fwd_t)
                skin_at, inside = lookup_nearest(deeplab.eroded[start:stop].view(n, -1), crop_xy, C)
                seen = valid & inside
                skin = seen & (skin_at > 0)
                skin_atlas = texels_to_atlas(skin, layout) if layout is bp or 'skin_mask' in outputs else None
                atlases = {}
                if 'skin_rgb' in outputs:
                    atlases[f'{prefix}skin_rgb'] = texels_to_atlas(torch.where(skin[..., None], colors, 0), layout)
                if 'skin_mask' in outputs:
                    atlases[f'{prefix}skin_mask_packed'] = pack_mask_bits(skin_atlas)
                if 'visible_mask' in outputs:
                    atlases[f'{prefix}visible_mask_packed'] = pack_mask_bits(texels_to_atlas(valid, layout))
                if 'class_ids' in outputs:
                    class_at, _ = lookup_nearest(deeplab.ids[start:stop].view(n, -1), crop_xy, C)
                    atlases[f'{prefix}class_ids'] = texels_to_atlas(torch.where(seen, class_at, 0), layout)
                return skin_atlas, atlases

        def process_batch(frames, box_args, frame_times=None, rows=None):
            """All GPU / DeepLab work of one batch (shared by the warm-up and the loop)."""
            n = frames.shape[0]
            pear_inv = gen_trans_from_patch_cv(*box_args, CROP_SIZE, CROP_SIZE, 1.0, 0.0, inv=True)
            skin_inv = gen_trans_from_patch_cv(*box_args, C, C, 1.0, 0.0, inv=True)
            skin_fwd = cv2.invertAffineTransform(skin_inv).astype(np.float32)
            pear_aff = torch.as_tensor(pear_inv, dtype=torch.float32, device=device).expand(n, 2, 3)
            skin_fwd_t = torch.as_tensor(skin_fwd, device=device).expand(n, 2, 3)
            with timer('crops'):
                patch = crop_batch(frames, pear_inv, CROP_SIZE, 'floor').div_(255.0)
                deeplab.prepare(frames, skin_inv)
            with timer('deeplab_submit' if overlap_deeplab else 'deeplab_inline'):
                deeplab.submit()
            with timer('backbone_fp16'):
                feats = self.run_backbone(patch, batch_size)
            with timer('head'):
                outputs = ehm_model.head(feats.float())
            diagnostic = {}
            if frame_times is not None:
                with timer('temporal_parameters'):
                    outputs, diagnostic = temporal.stabilize_model_outputs(outputs, frame_times)
            with timer('smplx'):
                verts = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')['vertices']
            with timer('uv_geometry'):
                raster = raster_batch(bp, verts, outputs['pd_cam'])
                screen = raster['screen']
            dl_out, parts, skin_parts = None, defaultdict(list), []
            layouts = [('uv_', bp, ATLAS_OUTPUTS)]
            if packed_uv:
                layouts.append(('packed_uv_', packed_uv_layout, packed_uv_outputs))
            for prefix, layout, wanted in layouts:
                for start in range(0, n, texel_chunk_frames):
                    stop = min(start + texel_chunk_frames, n)
                    stage = texel_stage(layout, raster_frames(raster, start, stop), frames[start:stop],
                                        pear_aff[start:stop])
                    if dl_out is None:  # the first texel pass is queued on the GPU before the CPU waits for DeepLab
                        with timer('deeplab_wait'):
                            dl_out = deeplab.result()
                    skin_atlas, atlases = atlas_stage(layout, prefix, *stage, skin_fwd_t[start:stop], start, wanted)
                    del stage
                    if layout is bp:
                        skin_parts.append(skin_atlas)
                    for name, value in atlases.items():
                        parts[name].append(value)
            del raster
            gpu_out = {name: values[0] if len(values) == 1 else torch.cat(values) for name, values in parts.items()}
            uv_skin = skin_parts[0] if len(skin_parts) == 1 else torch.cat(skin_parts)
            if frame_times is not None and temporal_config['diagnostics']:
                with timer('temporal_diagnostics'):
                    frame_vertices = apply_affine_batch(screen * (CROP_SIZE / RENDER_SIZE), pear_aff)
                    diagnostic.update(temporal.diagnose_geometry(frame_vertices, rows, n_batches))
            small = {'pd_cam': outputs['pd_cam'], 'crop_to_frame': np.broadcast_to(skin_inv, (n, 2, 3)),
                     'crop_box_xyxy': np.broadcast_to(np.r_[skin_inv[:, 2], skin_inv[:, :2] @ [C, C] + skin_inv[:, 2]],
                                                      (n, 4)).astype(np.float32),
                     'pear_crop_to_frame': np.broadcast_to(pear_inv, (n, 2, 3))}
            for group in ('body_param', 'flame_param'):
                small.update({f'{group}/{k}': v for k, v in outputs[group].items() if v is not None})
            small.update(diagnostic)
            return gpu_out, uv_skin, dl_out, small

        prev_precision = torch.get_float32_matmul_precision()
        # TF32 perturbs SMPL-X vertices by ~0.5 px; the FP16 backbone is unaffected.
        torch.set_float32_matmul_precision('highest')
        reader = writer = None
        running_rgb = running_skin = None
        bbox = None
        n_frames = n_ok = n_detect = 0
        t_start = time.perf_counter()
        completed = False
        try:
            # Two synthetic batches through the exact batch path: compiled backbone, cuDNN/cuSPARSE, ORT, YOLO and
            # the caching allocators are initialized before the clock starts.
            t_warm = time.perf_counter()
            with torch.inference_mode():
                self.detect_person_bbox(np.zeros((img_h, img_w, 3), dtype=np.uint8))
                for _ in range(2):
                    process_batch(frames_gpu, (img_w / 2, img_h / 2, float(img_w), float(img_w)))
                torch.cuda.synchronize()
            refiner.reset()
            temporal.reset_temporal_state()
            bp.front_sign = None
            deeplab.seconds.clear()
            timer.seconds.clear()
            log(f'warm-up: {time.perf_counter() - t_warm:.1f} s')

            reader = plan.open_reader(indices, n_slots=2 * batch_size + 2)
            frame_specs = {
                'uv_skin_rgb': (S, S, 3), 'uv_skin_mask_packed': (S, S // 8), 'uv_visible_mask_packed': (S, S // 8),
                'uv_class_ids': (S, S), 'crop_skin_rgb': (C, C, 3), 'crop_skin_mask_packed': (C, C // 8),
                'crop_class_ids': (C, C),
            }
            if packed_uv:
                ph, pw = packed_uv_layout.uv_body_mask.shape
                packed_specs = {'skin_rgb': (ph, pw, 3), 'skin_mask_packed': (ph, pw // 8),
                                'visible_mask_packed': (ph, pw // 8), 'class_ids': (ph, pw)}
                frame_specs.update({f'packed_uv_{name}': shape for name, shape in packed_specs.items()
                                    if name.removesuffix('_packed') in packed_uv_outputs})
            attrs = {
                'source_video_path': str(source_path.resolve()), 'source_type': plan.kind,
                'source_fps': sel['source_fps'],
                'source_fps_measured': sel['measured_fps'], 'source_r_frame_rate': info['r_frame_rate'],
                'source_timestamp_source': plan.timestamp_source, 'decode_pixel_format': plan.decode_format,
                'source_codec': info.get('codec_name', ''), 'source_pixel_format': info.get('pix_fmt', ''),
                'target_fps': sel['target_fps'], 'effective_fps': sel['effective_fps'],
                'frame_selection_mode': sel['mode'], 'frame_stride': sel['stride'],
                'decodable_frame_count': len(timestamps), 'reported_video_frame_count': info['nb_frames'],
                'max_frames': -1 if not max_frames else int(max_frames),
                'frame_width': img_w, 'frame_height': img_h, 'video_rotation': info['rotation'],
                'color_matrix': info['color_space'], 'color_range': info['color_range'],
                'texture_size': S, 'crop_size': C, 'pear_crop_size': CROP_SIZE, 'deeplab_input_size': DEEPLAB_SIZE,
                'focal_length': FOCAL_LENGTH, 'render_size': RENDER_SIZE, 'raster_size': bp.raster_size,
                'depth_tol_frac': bp.depth_tol_frac, 'fill_holes_px': FILL_HOLES_PX, 'bbox_ratio': BBOX_RATIO,
                'person_bbox_pad': person_bbox_pad,
                'bbox_initialization': 'YOLO seed from latest detectable batch frame, refined by DeepLab on that '
                                       'frame before saving all original batch rows',
                'yolo_conf': YOLO_CONF, 'batch_size': batch_size, 'float32_matmul_precision': 'highest',
                'cihp_class_names': CLASS_NAMES, 'cihp_colormap': COLORMAP, 'skin_class_ids': SKIN_CLASS_IDS,
                'skin_color_space': 'LAB (OpenCV 8-bit, from RGB)',
                'skin_mahalanobis_threshold': SKIN_COLOR_MAHALANOBIS_THRESHOLD,
                'skin_cov_regularization': SKIN_COLOR_COV_REGULARIZATION,
                'skin_reference_distance_px': SKIN_COLOR_REFERENCE_DISTANCE_PX,
                'skin_min_reference_pixels': SKIN_COLOR_MIN_REFERENCE_PIXELS,
                'skin_max_reference_samples': SKIN_COLOR_MAX_REFERENCE_SAMPLES,
                'skin_eval_scale': SKIN_COLOR_EVAL_SCALE, 'skin_temporal_model': ENABLE_TEMPORAL_SKIN_COLOR_MODEL,
                'skin_temporal_alpha': SKIN_COLOR_TEMPORAL_ALPHA, 'skin_model_every': SKIN_COLOR_MODEL_EVERY,
                'skin_mask_erode_px': SKIN_MASK_ERODE_PX,
                'mask_packing': 'np.packbits(mask, axis=-1) (MSB first); width = last dimension * 8',
                'crop_box_definition': 'crop_to_frame applied to crop points (0, 0) and (crop_size, crop_size)',
                'texel_order': 'np.flatnonzero(uv_body_mask) (row-major)',
                'deeplab_model': deeplab_session.path.name, 'yolo_model': Path(str(self.yolo_weights)).name,
                'pear_checkpoint': Path(self.checkpoint).name, 'pear_backbone': self.backbone_mode,
                'temporal_stabilization_config': json.dumps(temporal_config, sort_keys=True),
                'stored_parameters': 'stabilized PEAR parameters actually used for geometry; rotations retain head schema',
                'temporal_diagnostics_description': 'optional diagnostics/*; success rows only; bbox center/log(width,height); '
                                                    'rotation changes RMS radians; other changes RMS native units; '
                                                    'vertex displacement median original-frame pixels (includes real motion)',
                'packed_uv': bool(packed_uv),
                'producer': f'vHRM Studio {__version__}',
                **(extra_attrs or {}),
            }
            if packed_uv:
                tables = packed_uv_layout.tables
                attrs.update({
                    'packed_uv_height': ph, 'packed_uv_width': pw, 'packed_uv_layout_file': packed_uv_layout.path.name,
                    'packed_uv_outputs': list(packed_uv_outputs),
                    'packed_uv_arrangement': str(tables['arrangement']), 'packed_uv_gutter': int(tables['gutter']),
                    'packed_uv_chart_names': [str(name) for name in tables['chart_names']],
                    'packed_uv_mesh_digest': str(tables['mesh_digest']),
                    'packed_uv_texel_order': 'np.flatnonzero(packed_uv_body_mask) (row-major)',
                })
            writer = SkinUVH5Writer(h5_path, n_rows, frame_specs, [k for k in frame_specs if not k.startswith('crop_')],
                                    batch_size, attrs, gzip_level=int(cfg['gzip_level']),
                                    n_threads=int(cfg['compress_threads']))

            t_start = time.perf_counter()
            with torch.inference_mode():
                while True:
                    if should_stop is not None and should_stop():
                        raise Cancelled()
                    with timer('wait_decode'):
                        items = reader.next_batch(batch_size)
                    if not items:
                        break
                    n = len(items)
                    rows = [row for row, _ in items]
                    frame_times = timestamps[indices[rows]]

                    with timer('upload+convert'):
                        reader.upload(items, frames_gpu)
                        frames = frames_gpu[:n]

                    if bbox is None:
                        with timer('detect'):
                            detected_bbox, seed_index, detect_calls = self._detect_batch_person_bbox(frames)
                            n_detect += detect_calls
                        temporal.bbox_filter.reset()
                        seed_bbox = None if detected_bbox is None else process_bbox(
                            bbox=detected_bbox.copy(), img_width=img_w, img_height=img_h,
                            input_img_shape=[C, C], ratio=BBOX_RATIO)
                        if seed_bbox is not None and seed_index is not None:
                            with timer('bbox_initialize'):
                                seed_inv = gen_trans_from_patch_cv(
                                    float(seed_bbox[0] + 0.5 * seed_bbox[2]), float(seed_bbox[1] + 0.5 * seed_bbox[3]),
                                    float(seed_bbox[2]), float(seed_bbox[3]), C, C, 1.0, 0.0, inv=True)
                                deeplab.prepare(frames, seed_inv)
                                deeplab.submit()
                                seed_out = deeplab.result()
                                person_bbox = bbox_from_class_ids(
                                    seed_out['crop_class_ids'][seed_index], seed_inv, img_w, img_h,
                                    pad=person_bbox_pad)
                                refiner.reset()
                            bbox = temporal.stabilize_bbox(
                                detected_bbox if person_bbox is None else person_bbox, img_w, img_h, frame_times[0])
                    pbbox = None if bbox is None else process_bbox(
                        bbox=bbox.copy(), img_width=img_w, img_height=img_h,
                        input_img_shape=[CROP_SIZE, CROP_SIZE], ratio=BBOX_RATIO)
                    if pbbox is None:
                        torch.cuda.current_stream().synchronize()
                        reader.release(items)
                        writer.set_status(rows, 2 if bbox is None else 3)
                        log(f'rows {rows[0]}-{rows[-1]}: {"no person" if bbox is None else "invalid crop"}, re-detecting')
                        bbox = None
                        n_frames += n
                        if progress is not None:
                            progress(self._progress(n_rows, n_frames, n_ok, t_start))
                        continue

                    gpu_out, uv_skin, dl_out, small = process_batch(
                        frames, (float(pbbox[0] + 0.5 * pbbox[2]), float(pbbox[1] + 0.5 * pbbox[3]),
                                 float(pbbox[2]), float(pbbox[3])), frame_times=frame_times, rows=rows)
                    with timer('accumulate'):
                        if accumulate:
                            if running_rgb is None:
                                running_rgb = torch.zeros((S, S, 3), dtype=torch.uint8, device=device)
                                running_skin = torch.zeros((S, S), dtype=torch.bool, device=device)
                            for b in range(n):
                                running_rgb = torch.where(uv_skin[b, ..., None], gpu_out['uv_skin_rgb'][b], running_rgb)
                                running_skin |= uv_skin[b]
                    with timer('writer_wait'):
                        slot = writer.acquire_slot()
                    with timer('d2h+bbox'):
                        for name, tensor in gpu_out.items():
                            writer.slots[slot][name][:n].copy_(tensor, non_blocking=True)
                        torch.cuda.current_stream().synchronize()  # batch complete: slots and buffers can be reused
                        person_bbox = bbox_from_class_ids(
                            dl_out['crop_class_ids'][-1], small['crop_to_frame'][-1], img_w, img_h, pad=person_bbox_pad)
                        if person_bbox is None:  # nobody in the last crop: YOLO re-detects, the filter starts over
                            temporal.bbox_filter.reset()
                            bbox = None
                        else:
                            bbox = temporal.stabilize_bbox(person_bbox, img_w, img_h, frame_times[-1])

                    reader.release(items)
                    small = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in small.items()}
                    writer.submit(slot, rows, dl_out, small)
                    n_frames += n
                    n_ok += n
                    n_batches += 1
                    if progress is not None:
                        progress(self._progress(n_rows, n_frames, n_ok, t_start))
            completed = True
        finally:
            torch.set_float32_matmul_precision(prev_precision)
            deeplab.close()
            if reader is not None:
                reader.close()
            if writer is not None:
                extra = {'source_frame_index': indices, 'source_timestamp_s': timestamps[indices],
                         'uv_body_mask': bp.uv_body_mask.to(torch.uint8).cpu().numpy(),
                         'texel_fids': bp.texel_fids.to(torch.int32).cpu().numpy(),
                         'texel_bary': bp.texel_bary.cpu().numpy()}
                if packed_uv:
                    extra.update({'packed_uv_body_mask': packed_uv_layout.tables['uv_body_mask'].astype(np.uint8),
                                  'packed_uv_texel_fids': packed_uv_layout.tables['texel_fids'].astype(np.int32),
                                  'packed_uv_texel_bary': packed_uv_layout.tables['texel_bary'].astype(np.float32),
                                  'packed_uv_texel_chart': packed_uv_layout.tables['texel_chart']})
                if running_rgb is not None:
                    extra['accumulated_skin_rgb'] = running_rgb.cpu().numpy()
                    extra['accumulated_skin_mask_packed'] = pack_mask_bits(running_skin[None])[0].cpu().numpy()
                strings = {'source_frame_files': [plan.files[i] for i in indices]} if plan.kind == 'images' else None
                try:
                    writer.close(attrs={'front_sign': bp.front_sign if bp.front_sign is not None else 0.0,
                                        'decoded_rows': reader.n_read if reader is not None else 0,
                                        'complete': bool(completed)}, datasets=extra, strings=strings)
                except Exception:
                    if completed:
                        raise

        elapsed = time.perf_counter() - t_start
        stats = {'source': str(source_path), 'mode': sel['mode'], 'effective_fps': sel['effective_fps'],
                 'rows': n_rows, 'frames': n_frames, 'ok': n_ok, 'seconds': elapsed,
                 'fps': n_frames / max(elapsed, 1e-6), 'yolo_calls': n_detect, 'h5_path': str(h5_path),
                 'stage_ms': {k: 1000 * v / max(n_ok, 1) for k, v in timer.seconds.items()},
                 'deeplab_thread_ms': {k: 1000 * v / max(n_ok, 1) for k, v in deeplab.seconds.items()},
                 'skin_model_fits': refiner.model_fit_count, 'skin_undersized_refits': refiner.undersized_refits,
                 'gpu_used_gb': (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9}
        if reader is not None and reader.n_read != n_rows:
            log(f'WARNING: decoder delivered {reader.n_read} of {n_rows} selected frames')
        log(f"Done: {n_frames} frames ({n_ok} with a mesh) in {elapsed:.1f} s -> {stats['fps']:.1f} fps | "
            f"YOLO calls: {n_detect} | skin model fits: {refiner.model_fit_count} | "
            f"GPU memory in use {stats['gpu_used_gb']:.1f} GB")
        for name, ms in {**stats['stage_ms'], **stats['deeplab_thread_ms']}.items():
            log(f'{name:>16}: {ms:6.2f} ms/frame')
        if cfg['save_accumulated_png'] and running_rgb is not None:
            stem = h5_path.name.removesuffix('.partial').removesuffix('.h5').removesuffix('_skin_uv')
            cv2.imwrite(str(h5_path.with_name(f'{stem}_accumulated_skin_uv.png')),
                        cv2.cvtColor(running_rgb.cpu().numpy(), cv2.COLOR_RGB2BGR))
        return stats

    @staticmethod
    def _progress(n_rows, n_frames, n_ok, t_start):
        elapsed = time.perf_counter() - t_start
        return {'rows': n_rows, 'done': n_frames, 'ok': n_ok, 'elapsed': elapsed,
                'fps': n_frames / elapsed if elapsed > 0 else 0.0}
