"""Frame sources of the skin-UV pipeline: videos (ffmpeg / NVDEC) and image sequences (one image per frame).

Every reader delivers the selected frames, in order, into a ring of pinned host buffers and uploads a batch into a
(n, 3, H, W) float32 CUDA tensor holding exact uint8 levels, in display orientation.

* YUV videos are decoded by ffmpeg (NVDEC when available) into NV12 and converted on the GPU, exactly as in
  pear_deeplab_skin_uv_video.ipynb.
* RGB videos (e.g. the uncompressed bgr24 AVI files of UBFC-rPPG) are piped as rgb24: converting them to NV12 would
  subsample their chroma.
* Image sequences (e.g. PURE) are read with OpenCV on a few threads. Their timestamps come from the numbers in the
  file names when these look like time stamps (ns, us or ms), otherwise from a nominal frame rate.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TARGET_FPS = 30.0
FPS_MATCH_TOL = 0.01      # relative tolerance for "<= ~30 fps" and "close to an integer multiple of 30"
CFR_JITTER_TOL = 0.25     # stride mode requires every frame interval within 25% of 1 / fps
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm', '.ppm', '.webp')
RGB_PIXEL_FORMATS = re.compile(r'^(rgb|bgr|gbr|argb|abgr|0rgb|0bgr|rgba|bgra|pal8|gray|ya8|ya16|monob|monow)')

_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)  # no console window per call from the GUI
_FFMPEG = {}


def find_tool(name, configured=None):
    """ffmpeg / ffprobe: the configured folder or file, else PATH, else C:\\ffmpeg\\bin."""
    key = (name, configured)
    if key not in _FFMPEG:
        found = None
        if configured:
            c = Path(configured)
            for candidate in (c, c / f'{name}.exe', c / name, c / 'bin' / f'{name}.exe', c / 'bin' / name):
                if candidate.is_file():
                    found = str(candidate)
                    break
        found = found or shutil.which(name)
        if found is None and Path(rf'C:\ffmpeg\bin\{name}.exe').is_file():
            found = rf'C:\ffmpeg\bin\{name}.exe'
        _FFMPEG[key] = found
    if _FFMPEG[key] is None:
        raise RuntimeError(f'{name} was not found (install ffmpeg and put it on PATH, or set its folder in Settings)')
    return _FFMPEG[key]


# ---------------------------------------------------------------- frame selection (notebook, unchanged)
def _rate(text):
    num, _, den = (text or '0/0').partition('/')
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def probe_video(path, ffmpeg_dir=None):
    cmd = [find_tool('ffprobe', ffmpeg_dir), '-v', 'error', '-select_streams', 'v:0', '-of', 'json', '-show_entries',
           'stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,color_space,'
           'color_range:stream_tags=rotate:stream_side_data=rotation', str(path)]
    streams = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True,
                                        creationflags=_NO_WINDOW).stdout).get('streams') or []
    if not streams:
        raise RuntimeError(f'{path}: no video stream')
    stream = streams[0]
    rotation = -int(float(stream.get('tags', {}).get('rotate', 0)))  # legacy tag has the opposite sign
    for side_data in stream.get('side_data_list', []):
        if 'rotation' in side_data:
            rotation = int(side_data['rotation'])
    return {
        'width': int(stream['width']), 'height': int(stream['height']),
        'r_frame_rate': _rate(stream.get('r_frame_rate')), 'avg_frame_rate': _rate(stream.get('avg_frame_rate')),
        'nb_frames': int(stream.get('nb_frames') or 0), 'rotation': rotation,
        'duration': float(stream.get('duration') or 0.0),
        'color_space': stream.get('color_space', 'unknown'), 'color_range': stream.get('color_range', 'tv'),
        'codec_name': stream.get('codec_name', ''), 'pix_fmt': stream.get('pix_fmt', ''),
    }


def probe_frame_timestamps(path, ffmpeg_dir=None):
    """Presentation timestamps (s) of every decodable frame, in decoder output order.

    Demux only when packet PTS are available; otherwise probe decoded-frame best-effort timestamps.
    Packets flagged D (discarded by the edit list) are never output by the decoder."""
    return _probe_frame_timestamps(path, ffmpeg_dir)[0]


def _probe_frame_timestamps(path, ffmpeg_dir=None):
    cmd = [find_tool('ffprobe', ffmpeg_dir), '-v', 'error', '-select_streams', 'v:0', '-show_entries',
           'packet=pts_time,flags', '-of', 'csv=p=0', str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=_NO_WINDOW).stdout
    timestamps = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pts, _, flags = line.strip().partition(',')
        if 'D' in flags:
            continue
        if not pts or pts == 'N/A':
            warnings.warn(f'{path}: packet presentation timestamps are missing; probing decoded-frame '
                          'best-effort timestamps instead (slower).', RuntimeWarning, stacklevel=2)
            cmd = [find_tool('ffprobe', ffmpeg_dir), '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                   'frame=best_effort_timestamp_time', '-of', 'json', str(path)]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=_NO_WINDOW).stdout
            decoded = json.loads(out).get('frames', [])
            timestamps = []
            for index, frame in enumerate(decoded):
                timestamp = frame.get('best_effort_timestamp_time')
                if not timestamp or timestamp == 'N/A':
                    raise RuntimeError(f'{path}: decoded frame {index} without a presentation timestamp')
                timestamps.append(float(timestamp))
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
                raise RuntimeError(f'{path}: decoded-frame timestamps are not finite or not in presentation order')
            return timestamps, 'decoded frames (ffprobe best-effort timestamps)'
        timestamps.append(float(pts))
    return np.sort(np.asarray(timestamps, dtype=np.float64)), 'container packets (ffprobe)'


def select_frames(timestamps, source_fps, target_fps=TARGET_FPS):
    """Choose source frames for ~target_fps processing. Returns mode ('all' | 'stride' | 'timestamp'), indices,
    stride and effective fps."""
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    if n == 0:
        raise ValueError('no frames')
    ratio = source_fps / target_fps
    stride = int(round(ratio))
    period = 1.0 / source_fps
    regular = n < 2 or bool(np.abs(np.diff(ts) - period).max() <= CFR_JITTER_TOL * period)
    if ratio <= 1.0 + FPS_MATCH_TOL:
        mode, indices, effective, stride = 'all', np.arange(n), source_fps, 1
    elif stride >= 2 and regular and abs(ratio - stride) <= FPS_MATCH_TOL * stride:
        mode, indices, effective = 'stride', np.arange(0, n, stride), source_fps / stride
    else:
        mode, stride = 'timestamp', 0
        targets = ts[0] + np.arange(int(np.floor((ts[-1] - ts[0]) * target_fps + 0.5)) + 1) / target_fps
        right = np.clip(np.searchsorted(ts, targets), 1, max(n - 1, 1)) if n > 1 else np.zeros(len(targets), int)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(ts[left] - targets) <= np.abs(ts[right] - targets), left, right)
        indices = np.unique(nearest)  # never the same frame twice
        span = ts[indices[-1]] - ts[indices[0]]
        effective = (len(indices) - 1) / span if span > 0 else target_fps
    return {'mode': mode, 'indices': indices.astype(np.int64), 'stride': stride, 'source_fps': float(source_fps),
            'target_fps': float(target_fps), 'effective_fps': float(effective), 'regular_timestamps': regular}


def _measured_fps(ts):
    return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0


# ---------------------------------------------------------------- image sequences
_TRAILING_INT = re.compile(r'(\d+)(?!.*\d)')
_TIME_UNITS = ((1e-9, 'ns'), (1e-6, 'us'), (1e-3, 'ms'))


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(name))]


def list_images(folder):
    """Image files directly inside folder, in natural order (frame2 before frame10)."""
    names = [e.name for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(IMAGE_EXTENSIONS)]
    return sorted(names, key=natural_key)


def image_timestamps(names, nominal_fps=TARGET_FPS):
    """Timestamps (s, first frame = 0) of an image sequence and how they were obtained.

    The last number of each file name is a frame index when consecutive numbers mostly differ by 1 (gaps are kept),
    a time stamp when they differ by a frame interval in ns, us or ms (PURE: Image<ns>.png); otherwise frames are
    taken to be 1 / nominal_fps apart."""
    n = len(names)
    numbers = []
    for name in names:
        match = _TRAILING_INT.search(Path(name).stem)
        numbers.append(int(match.group(1)) if match else None)
    if n > 1 and None not in numbers and max(numbers) < 2 ** 63:
        relative = np.asarray([v - numbers[0] for v in numbers], dtype=np.int64).astype(np.float64)  # exact for ns
        steps = np.diff(relative)
        if (steps > 0).all():
            median = float(np.median(steps))
            if median <= 2.0:  # frame numbers (possibly with gaps), not time stamps
                return relative / nominal_fps, f'frame index in the file name, {nominal_fps:g} fps'
            for unit, label in _TIME_UNITS:
                if 1.0 <= 1.0 / (median * unit) <= 1000.0:
                    return relative * unit, f'time stamp in the file name ({label})'
    return np.arange(n, dtype=np.float64) / nominal_fps, f'nominal {nominal_fps:g} fps'


def read_image_rgb(path):
    """(H, W, 3) uint8 RGB of any OpenCV-readable image (grey, BGRA and 16-bit images are converted)."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)  # cv2.imread does not take non-ASCII Windows paths
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f'cannot read image {path}')
    if image.dtype == np.uint16:
        image = ((image.astype(np.uint32) + 128) // 257).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def probe_images(folder, names):
    import cv2  # noqa: F401  (fail early)
    first = read_image_rgb(Path(folder) / names[0])
    return {'width': int(first.shape[1]), 'height': int(first.shape[0]), 'r_frame_rate': 0.0,
            'avg_frame_rate': 0.0, 'nb_frames': len(names), 'rotation': 0, 'duration': 0.0,
            'color_space': 'rgb', 'color_range': 'pc', 'codec_name': 'images', 'pix_fmt': 'rgb24',
            'image_extension': Path(names[0]).suffix.lower()}


# ---------------------------------------------------------------- source plan
class SourcePlan:
    """Probe + frame selection of one source; open_reader() starts decoding the selected frames."""

    def __init__(self, kind, path, target_fps=TARGET_FPS, image_fps=TARGET_FPS, ffmpeg_dir=None):
        self.kind, self.path, self.ffmpeg_dir = kind, Path(path), ffmpeg_dir
        if kind == 'images':
            self.files = list_images(self.path)
            if not self.files:
                raise RuntimeError(f'{self.path}: no images')
            self.info = probe_images(self.path, self.files)
            self.timestamps, self.timestamp_source = image_timestamps(self.files, image_fps)
            measured = _measured_fps(self.timestamps)
            source_fps = measured or image_fps
            self.decode_format = 'image'
        else:
            self.files = None
            self.info = probe_video(self.path, ffmpeg_dir)
            self.timestamps, self.timestamp_source = _probe_frame_timestamps(self.path, ffmpeg_dir)
            if not len(self.timestamps):
                raise RuntimeError(f'{self.path}: no decodable frames')
            measured = _measured_fps(self.timestamps)
            source_fps = self.info['avg_frame_rate'] or measured or self.info['r_frame_rate']
            self.decode_format = 'rgb24' if RGB_PIXEL_FORMATS.match(self.info['pix_fmt'] or '') else 'nv12'
        self.selection = select_frames(self.timestamps, source_fps, target_fps)
        self.selection['measured_fps'] = float(measured)

    def open_reader(self, indices, n_slots):
        if self.kind == 'images':
            return ImageSequenceReader(self.path, [self.files[i] for i in indices], self.info, n_slots)
        return FfmpegFrameReader(self.path, indices, n_slots, self.info, self.decode_format, self.ffmpeg_dir)


# ---------------------------------------------------------------- readers
def select_filter(frame_indices):
    """ffmpeg filter that keeps exactly frame_indices (strictly increasing). None = keep every frame."""
    idx = np.asarray(frame_indices, dtype=np.int64)
    if len(idx) == 0 or (len(idx) > 1 and np.any(np.diff(idx) <= 0)):
        raise ValueError('frame indices must be non-empty and strictly increasing')
    steps = np.unique(np.diff(idx))
    if idx[0] == 0 and len(steps) <= 1:
        stride = int(steps[0]) if len(steps) else 1
        return None if stride == 1 else f'select=not(mod(n\\,{stride}))'
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts, ends = idx[np.r_[0, breaks]], idx[np.r_[breaks - 1, len(idx) - 1]]

    def build(lo, hi):  # binary search tree: O(log ranges) evaluation per decoded frame
        if hi - lo == 1:
            return f'eq(n,{starts[lo]})' if starts[lo] == ends[lo] else f'between(n,{starts[lo]},{ends[lo]})'
        mid = (lo + hi) // 2
        return f'if(lt(n,{starts[mid]}),{build(lo, mid)},{build(mid, hi)})'

    return f"select='{build(0, len(starts))}'"


def yuv_to_rgb_coeffs(color_space, color_range):
    kr, kb = (0.2126, 0.0722) if color_space == 'bt709' else (0.299, 0.114)
    kg = 1.0 - kr - kb
    y_scale, c_scale, y_offset = (1.0, 1.0, 0.0) if color_range == 'pc' else (255 / 219, 255 / 224, 16.0)
    return {'ys': y_scale, 'yo': y_offset, 'rv': c_scale * 2 * (1 - kr), 'bu': c_scale * 2 * (1 - kb),
            'gu': c_scale * 2 * kb * (1 - kb) / kg, 'gv': c_scale * 2 * kr * (1 - kr) / kg}


def nv12_to_rgb_(nv12, out, height, width, rot_k, coeffs):
    """nv12: (n, frame_bytes) uint8 CUDA (stored orientation). Writes display-oriented planar RGB into out[:n]
    (n, 3, H, W) float32 holding exact uint8 levels."""
    import torch
    plane = height * width
    for i in range(nv12.shape[0]):
        y = torch.rot90(nv12[i, :plane].view(height, width), rot_k, (0, 1)).contiguous()
        uv = torch.rot90(nv12[i, plane:].view(height // 2, width // 2, 2), rot_k, (0, 1)).float().sub_(128.0)
        h2, w2 = y.shape[0] // 2, y.shape[1] // 2
        yf = y.float().sub_(coeffs['yo']).mul_(coeffs['ys']).view(h2, 2, w2, 2)
        u, v = uv[..., 0][:, None, :, None], uv[..., 1][:, None, :, None]
        r, g, b = (out[i, c].view(h2, 2, w2, 2) for c in range(3))
        torch.add(yf, v, alpha=coeffs['rv'], out=r)
        torch.add(yf, u, alpha=-coeffs['gu'], out=g).add_(v, alpha=-coeffs['gv'])
        torch.add(yf, u, alpha=coeffs['bu'], out=b)
    out[:nv12.shape[0]].round_().clamp_(0, 255)


def rgb_to_planar_(rgb, out, height, width, rot_k):
    """rgb: (n, H*W*3) uint8 CUDA, stored orientation -> out[:n] (n, 3, H', W') float32 in display orientation."""
    import torch
    for i in range(rgb.shape[0]):
        image = rgb[i].view(height, width, 3)
        if rot_k:
            image = torch.rot90(image, rot_k, (0, 1))
        out[i].copy_(image.permute(2, 0, 1))


class _PinnedRing:
    """Pinned host slots filled by a producer thread; next_batch / release / close as in the notebook reader."""

    def __init__(self, frame_bytes, n_slots):
        import torch
        self.frame_bytes = frame_bytes
        self.slots = [torch.empty(frame_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(n_slots)]
        self.free = queue.Queue()
        for slot in range(n_slots):
            self.free.put(slot)
        self.ready = queue.Queue()
        self.stop = threading.Event()
        self.eof = False
        self.error = None
        self.n_read = 0
        self._staging = None

    def next_batch(self, n):
        """Up to n (row, slot) pairs, row = position in the selected frames; an empty list at end of stream."""
        items = []
        while len(items) < n and not self.eof:
            item = self.ready.get()
            if item is None:
                self.eof = True
                self._check_end()
                break
            items.append(item)
        return items

    def _check_end(self):
        if self.error is not None:
            raise RuntimeError('Frame reader failed') from self.error

    def release(self, items):
        for _, slot in items:
            self.free.put(slot)

    def staging(self, n, device):
        import torch
        if self._staging is None or self._staging.shape[0] < n or self._staging.device != device:
            self._staging = torch.empty((n, self.frame_bytes), dtype=torch.uint8, device=device)
        return self._staging[:n]


class FfmpegFrameReader(_PinnedRing):
    """Decodes with ffmpeg (NVDEC when the codec supports it) and pipes only the requested frames, as NV12 or rgb24,
    into a ring of pinned buffers. Frames that are not selected are dropped inside ffmpeg and never reach Python.
    Frames stay in the stored (unrotated) orientation until upload()."""

    def __init__(self, path, frame_indices, n_slots, info, pix_fmt='nv12', ffmpeg_dir=None):
        self.info = info
        self.pix_fmt = pix_fmt
        self.width, self.height = info['width'], info['height']
        self.rot_k = (info['rotation'] // 90) % 4
        self.coeffs = yuv_to_rgb_coeffs(info['color_space'], info['color_range'])
        frame_bytes = self.width * self.height * 3 // (2 if pix_fmt == 'nv12' else 1)
        super().__init__(frame_bytes, n_slots)
        self.stderr_tail = []
        self.filter_script = None
        cmd = [find_tool('ffmpeg', ffmpeg_dir), '-hide_banner', '-loglevel', 'error', '-nostdin']
        if pix_fmt == 'nv12':
            cmd += ['-hwaccel', 'cuda']
        cmd += ['-noautorotate', '-i', str(path), '-map', '0:v:0']
        vf = select_filter(frame_indices)
        if vf is not None and len(vf) < 2000:
            cmd += ['-vf', vf]
        elif vf is not None:
            with tempfile.NamedTemporaryFile('w', suffix='.ffscript', delete=False) as script:
                script.write(vf)
            self.filter_script = script.name
            cmd += ['-/filter:v', self.filter_script]
        cmd += ['-fps_mode', 'passthrough', '-frames:v', str(len(frame_indices)),
                '-pix_fmt', pix_fmt, '-f', 'rawvideo', 'pipe:1']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                     creationflags=_NO_WINDOW)
        self.threads = [threading.Thread(target=self._read_frames, daemon=True),
                        threading.Thread(target=self._drain_stderr, daemon=True)]
        for thread in self.threads:
            thread.start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self.stderr_tail = (self.stderr_tail + [line.decode(errors='replace').rstrip()])[-20:]

    def _read_frames(self):
        try:
            index = 0
            while not self.stop.is_set():
                slot = self.free.get()
                if slot is None:
                    break
                view = memoryview(self.slots[slot].numpy())
                filled = 0
                while filled < self.frame_bytes:
                    got = self.proc.stdout.readinto(view[filled:])
                    if not got:
                        break
                    filled += got
                if filled < self.frame_bytes:
                    break
                self.ready.put((index, slot))
                index += 1
                self.n_read = index
        except Exception as exc:
            self.error = exc
        finally:
            self.ready.put(None)

    def _check_end(self):
        super()._check_end()
        if self.proc.wait() != 0 and not self.stop.is_set():
            raise RuntimeError('ffmpeg failed:\n' + '\n'.join(self.stderr_tail))

    def upload(self, items, frames):
        """Copy a batch to the GPU and convert it into frames[:n] (display-oriented planar RGB levels)."""
        n = len(items)
        staging = self.staging(frames.shape[0], frames.device)
        for b, (_, slot) in enumerate(items):
            staging[b].copy_(self.slots[slot], non_blocking=True)
        if self.pix_fmt == 'nv12':
            nv12_to_rgb_(staging[:n], frames, self.height, self.width, self.rot_k, self.coeffs)
        else:
            rgb_to_planar_(staging[:n], frames, self.height, self.width, self.rot_k)

    def close(self):
        self.stop.set()
        self.free.put(None)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for thread in self.threads:
            thread.join(timeout=5)
        if self.filter_script is not None:
            Path(self.filter_script).unlink(missing_ok=True)


class ImageSequenceReader(_PinnedRing):
    """Reads the given image files (display orientation, RGB) on a few threads, in order."""

    def __init__(self, folder, names, info, n_slots, n_threads=4):
        self.info = info
        self.folder, self.names = Path(folder), list(names)
        self.width, self.height = info['width'], info['height']
        super().__init__(self.width * self.height * 3, n_slots)
        self.pool = ThreadPoolExecutor(n_threads, thread_name_prefix='images')
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        try:
            pending = queue.Queue()
            ahead = max(2, len(self.slots) // 2)
            names = iter(enumerate(self.names))
            for index, name in names:  # keep `ahead` decodes in flight, deliver in order
                pending.put((index, name, self.pool.submit(read_image_rgb, self.folder / name)))
                if pending.qsize() < ahead:
                    continue
                if not self._deliver(pending.get()):
                    return
            while not pending.empty():
                if not self._deliver(pending.get()):
                    return
        except Exception as exc:
            self.error = exc
        finally:
            self.ready.put(None)

    def _deliver(self, job):
        index, name, future = job
        image = future.result()
        if image.shape != (self.height, self.width, 3):
            raise RuntimeError(f'{name}: {image.shape[1]}x{image.shape[0]} differs from the first image '
                               f'({self.width}x{self.height})')
        while True:
            if self.stop.is_set():
                return False
            try:
                slot = self.free.get(timeout=0.2)
                break
            except queue.Empty:
                pass
        if slot is None:
            return False
        self.slots[slot].numpy()[:] = image.reshape(-1)
        self.ready.put((index, slot))
        self.n_read = index + 1
        return True

    def upload(self, items, frames):
        n = len(items)
        staging = self.staging(frames.shape[0], frames.device)
        for b, (_, slot) in enumerate(items):
            staging[b].copy_(self.slots[slot], non_blocking=True)
        rgb_to_planar_(staging[:n], frames, self.height, self.width, 0)

    def close(self):
        self.stop.set()
        self.free.put(None)
        self.thread.join(timeout=10)
        self.pool.shutdown(wait=True, cancel_futures=True)
