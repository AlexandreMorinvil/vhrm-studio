"""vHRM Studio: viewer of the per-video skin-UV HDF5 files, and dataset conversion (Dataset > Convert a dataset).

    vhrm-studio                          # then choose folders and settings inside the application
    vhrm-studio [FILE_OR_FOLDER ...] [--assets DIR] [--no-mesh] [--load-mesh] [--video-scale 0.5]
    python -m vhrm_studio ...

Browse a folder of HDF5 files, step through or play the rows of one file and show any mix of the stored UV / crop
products, the source video or image sequence (when it can be found) and the SMPL-X mesh rebuilt from the stored PEAR
parameters. Requirements: PySide6, pyqtgraph, h5py, numpy, opencv, av (source video, MP4 export).
Optional: torch + pytorch3d + the SMPL-X / FLAME assets (mesh views, loaded on demand; also needed for conversion).

Keys: Space play/pause, Left/Right step, Shift+Left/Right 10 rows, Home/End, I/O in/out marks, Ctrl+I clear marks,
PgUp/PgDn previous/next file, F fit all panels, Ctrl+S snapshot, Ctrl+E export video.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
import warnings
import zlib
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from fractions import Fraction
from pathlib import Path

os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
os.environ.setdefault('QT_LOGGING_RULES', 'qt.core.qobject.connect=false')  # harmless pyqtgraph/PySide6 warning

import cv2
import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg

_AV_IMPORT_ERROR = None
try:
    import av
except ImportError as exc:
    av = None
    _AV_IMPORT_ERROR = exc


def _pyav_error_message():
    missing = isinstance(_AV_IMPORT_ERROR, ModuleNotFoundError) and _AV_IMPORT_ERROR.name == 'av'
    status = 'PyAV is not installed' if missing else 'PyAV could not be imported'
    reason = (f'{type(_AV_IMPORT_ERROR).__name__}: {_AV_IMPORT_ERROR}'
              if _AV_IMPORT_ERROR is not None else 'av is unavailable in this process')
    diagnostic = f'"{sys.executable}" -c "import av"'
    repair = (f'Install into this interpreter: "{sys.executable}" -m pip install av.'
              if missing else 'Check native-library dependencies and Windows Application Control/signing policy. '
              'A policy block needs an administrator-approved package, not just pip install av.')
    return (f'{status}: {reason}\nPython: {sys.executable}\nCheck: {diagnostic}\n{repair} '
            'Restart the viewer after the import check succeeds.')


if av is None:
    warnings.warn(_pyav_error_message(), RuntimeWarning)

STATUS_TEXT = {0: 'not processed', 1: 'ok', 2: 'no person detected', 3: 'invalid crop'}
DEFAULT_CLASS_NAMES = [
    'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
    'Torso-skin', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe',
]
DEFAULT_COLORMAP = np.array([
    [0, 0, 0], [31, 119, 180], [44, 160, 44], [44, 127, 125], [52, 225, 143],
    [217, 222, 163], [254, 128, 37], [130, 162, 128], [121, 7, 166], [136, 183, 248],
    [85, 1, 76], [22, 23, 62], [159, 50, 15], [101, 93, 152], [252, 229, 92],
    [167, 173, 17], [218, 252, 252], [238, 126, 197], [116, 157, 140], [214, 220, 252],
], dtype=np.uint8)
SKIN_REGIONS = {'All skin': (10, 13, 14, 15, 16, 17), 'Face': (13,), 'Torso': (10,), 'Arms': (14, 15),
                'Legs': (16, 17)}
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int64)
UV_EMPTY_LEVEL = 32       # grey of body texels without data when the UV silhouette is on
MESH_RASTER_SIZE = 768
PREFETCH_AHEAD = 8
LAYOUTS = {'1x1': (1, 1), '1x2': (1, 2), '2x1': (2, 1), '1x3': (1, 3), '2x2': (2, 2), '2x3': (2, 3),
           '1x4': (1, 4), '2x4': (2, 4), '3x3': (3, 3), '3x4': (3, 4)}
DEFAULT_PANELS = ['frame_skin', 'crop_classes_overlay', 'uv_skin_rgb', 'uv_coverage', 'uv_classes', 'uv_accum',
                  'crop_video', 'uv_age', 'frame_mesh', 'mesh3d', 'crop_skin_rgb', 'uv_accum_final']


# ---------------------------------------------------------------------------------------------------------- views
class View:
    def __init__(self, key, label, space, items, base=False):
        self.key, self.label, self.space, self.items, self.base = key, label, space, tuple(items), base


VIEWS = [
    View('frame', 'Frame: source video + crop box', 'frame', ['video']),
    View('frame_skin', 'Frame: video + refined skin mask', 'frame', ['video', 'crop_skin_mask']),
    View('frame_classes', 'Frame: video + CIHP classes', 'frame', ['video', 'crop_class_ids']),
    View('frame_mesh', 'Frame: video + SMPL-X mesh', 'frame', ['video', 'mesh']),
    View('crop_video', 'Crop: from source video', 'crop', ['video']),
    View('crop_skin_rgb', 'Crop: crop_skin_rgb (stored)', 'crop', ['crop_skin_rgb']),
    View('crop_skin_mask', 'Crop: refined skin mask', 'crop', ['crop_skin_mask']),
    View('crop_skin_outline', 'Crop: image + skin mask outline', 'crop', ['crop_skin_mask'], base=True),
    View('crop_classes', 'Crop: CIHP classes', 'crop', ['crop_class_ids']),
    View('crop_classes_overlay', 'Crop: image + CIHP classes', 'crop', ['crop_class_ids'], base=True),
    View('crop_mesh', 'Crop: image + SMPL-X mesh', 'crop', ['mesh'], base=True),
    View('uv_skin_rgb', 'UV: skin RGB', 'uv', ['uv_skin_rgb', 'uv_skin_mask']),
    View('uv_coverage', 'UV: coverage (body / visible / skin)', 'uv', ['uv_visible_mask', 'uv_skin_mask']),
    View('uv_skin_mask', 'UV: skin mask', 'uv', ['uv_skin_mask']),
    View('uv_visible_mask', 'UV: visible mask', 'uv', ['uv_visible_mask']),
    View('uv_classes', 'UV: CIHP classes (visible texels)', 'uv', ['uv_class_ids', 'uv_visible_mask']),
    View('uv_classes_overlay', 'UV: skin RGB + CIHP classes', 'uv',
         ['uv_class_ids', 'uv_visible_mask', 'uv_skin_rgb', 'uv_skin_mask']),
    View('uv_accum', 'UV: accumulated skin up to this row', 'uv', ['accum']),
    View('uv_age', 'UV: time since texel last seen as skin', 'uv', ['accum']),
    View('uv_accum_final', 'UV: accumulated skin, whole video (stored)', 'uv', []),
    View('packed_uv_skin_rgb', 'Packed UV: skin RGB', 'packed_uv', ['packed_uv_skin_rgb']),
    View('mesh3d', 'Mesh: SMPL-X orbit view (drag / wheel)', '3d', ['mesh']),
        View('standard', 'Standard pose: shaded SMPL-X', '3d', ['standard_mesh']),
        View('standard_uv_skin_rgb', 'Standard pose: UV skin RGB', '3d', ['standard_mesh', 'uv_skin_rgb', 'uv_skin_mask']),
        View('standard_crop_skin_rgb', 'Standard pose: projected skin crop RGB', '3d',
            ['standard_mesh', 'mesh', 'crop_skin_rgb', 'crop_skin_mask', 'uv_visible_mask']),
        View('standard_crop_skin_mask', 'Standard pose: projected refined skin mask (before erosion)', '3d',
            ['standard_mesh', 'mesh', 'crop_skin_mask', 'uv_visible_mask']),
        View('standard_crop_classes', 'Standard pose: projected crop CIHP classes', '3d',
            ['standard_mesh', 'mesh', 'crop_class_ids', 'uv_visible_mask']),
        View('standard_uv_skin_mask', 'Standard pose: UV skin mask (after erosion)', '3d',
            ['standard_mesh', 'uv_skin_mask']),
        View('standard_uv_visible_mask', 'Standard pose: observed surfaces', '3d', ['standard_mesh', 'uv_visible_mask']),
        View('standard_uv_coverage', 'Standard pose: coverage (body / visible / skin)', '3d',
            ['standard_mesh', 'uv_visible_mask', 'uv_skin_mask']),
        View('standard_uv_classes', 'Standard pose: UV CIHP classes', '3d',
            ['standard_mesh', 'uv_class_ids', 'uv_visible_mask']),
        View('standard_uv_accum', 'Standard pose: accumulated skin up to this row', '3d', ['standard_mesh', 'accum']),
        View('standard_uv_accum_final', 'Standard pose: accumulated skin, whole video', '3d', ['standard_mesh']),
]
VIEW_BY_KEY = {v.key: v for v in VIEWS}
H5_ROW_ITEMS = {'uv_skin_rgb', 'uv_class_ids', 'uv_skin_mask', 'uv_visible_mask', 'crop_skin_rgb', 'crop_class_ids',
                'crop_skin_mask', 'packed_uv_skin_rgb', 'packed_uv_skin_mask'}


class Rendered:
    __slots__ = ('image', 'aux')

    def __init__(self, image, aux=None):
        self.image, self.aux = image, aux


class Placeholder:
    __slots__ = ('text',)

    def __init__(self, text):
        self.text = text


class MeshResult:
    __slots__ = ('verts', 'renderer', 'image', 'orbit', 'texture', 'pan')

    def __init__(self, verts, renderer, image, orbit, texture=None, pan=(0.0, 0.0)):
        self.verts, self.renderer, self.image, self.orbit = verts, renderer, image, orbit
        self.texture = texture
        self.pan = pan


MESH_ORBIT = [0.0, 0.0, 1.0]  # yaw (deg), pitch (deg), zoom shared by the 3D panels
MESH_PAN = [0.0, 0.0]


class Unavailable(Exception):
    pass


# --------------------------------------------------------------------------------------------------- small helpers
def attr_value(value):
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    if isinstance(value, np.ndarray) and value.dtype.kind in 'OSU':
        return [v.decode(errors='replace') if isinstance(v, bytes) else str(v) for v in value.ravel()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_value(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, suppress_small=True, threshold=24, max_line_width=160)
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, list) and len(value) > 24:
        return f'[{", ".join(map(str, value[:24]))}, ... ({len(value)} items)]'
    return str(value)


def affine3(a):
    return np.vstack([np.asarray(a, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])


def compose(*affines):
    """compose(a, b, c) = a o b o c (c applied first)."""
    out = np.eye(3)
    for a in affines:
        out = out @ affine3(a)
    return out[:2]


def invert(a):
    return cv2.invertAffineTransform(np.asarray(a, dtype=np.float64))


def blend(base, color, weight):
    """base (H, W, 3) uint8; color (H, W, 3) uint8 or an RGB triple; weight (H, W) in [0, 1]."""
    if np.ndim(color) == 1:
        fill = np.empty_like(base)
        fill[:] = color
        color = fill
    w = np.asarray(weight, dtype=np.float32)
    return cv2.blendLinear(np.ascontiguousarray(base), np.ascontiguousarray(color), 1.0 - w, w)


def as_u8(mask):
    return mask.view(np.uint8) if mask.dtype == bool else mask


def masked(image, mask, background):
    """image where mask, background elsewhere."""
    out = background.copy()
    cv2.copyTo(np.ascontiguousarray(image), np.ascontiguousarray(as_u8(mask)), out)
    return out


def colorize(ids, lut):
    """Class ids (H, W) uint8 -> RGB through a (256, 1, 3) lookup table."""
    return cv2.applyColorMap(np.ascontiguousarray(ids), lut)


def gray3(mask):
    return cv2.cvtColor(as_u8(mask) * np.uint8(255), cv2.COLOR_GRAY2RGB)


COVERAGE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)  # bit 0 body, bit 1 visible, bit 2 skin
COVERAGE_LUT[1] = 70
COVERAGE_LUT[2:4] = (40, 90, 210)
COVERAGE_LUT[4:8] = (60, 210, 90)


def even(x):
    return max(2, int(round(x / 2.0)) * 2)


def qimage_to_rgb(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    buf = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    return buf.reshape(h, image.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()


# ------------------------------------------------------------------------------------------------------ data access
# This HDF5 build fails under concurrent calls from several threads ("no VOL object wrap context"), so every h5py
# call runs on one thread; chunk decompression stays in the calling threads.
_H5_POOL = ThreadPoolExecutor(1, thread_name_prefix='hdf5')
_H5_THREAD = _H5_POOL.submit(threading.get_ident).result()


def h5call(fn, *args, **kwargs):
    if threading.get_ident() == _H5_THREAD:
        return fn(*args, **kwargs)
    return _H5_POOL.submit(fn, *args, **kwargs).result()


class SkinUVFile:
    """Read-only access to one skin-UV HDF5 file. Per-frame images are read one chunk at a time; the zlib
    decompression runs outside the HDF5 lock so that several threads can decode in parallel."""

    def __init__(self, path):
        self.path = Path(path)
        self._final = None
        self._texels = None
        self._lock = threading.Lock()
        h5call(self._open)
        a = self.attrs
        self.n_rows = len(self.status)
        self.texture_size = int(a.get('texture_size', 1024))
        self.crop_size = int(a.get('crop_size', 512))
        self.fps = float(a.get('effective_fps') or a.get('source_fps') or 30.0)
        if self.timestamps is None:
            self.timestamps = np.arange(self.n_rows) / self.fps
        if self.frame_index is None:
            self.frame_index = np.arange(self.n_rows)
        names = a.get('cihp_class_names')
        self.class_names = list(names) if isinstance(names, list) and names else DEFAULT_CLASS_NAMES
        cmap = np.asarray(a.get('cihp_colormap', DEFAULT_COLORMAP), dtype=np.uint8).reshape(-1, 3)
        self.colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        self.colormap[:len(cmap), 0] = cmap
        S = self.texture_size
        self.uv_bg = np.zeros((S, S, 3), dtype=np.uint8)
        self.uv_body_u8 = None
        if self.uv_body is not None:
            self.uv_bg[self.uv_body] = UV_EMPTY_LEVEL
            self.uv_body_u8 = self.uv_body.view(np.uint8)

    def _open(self):
        """Runs on the HDF5 thread: everything small is read here, per-row images later chunk by chunk."""
        self.h5 = f = h5py.File(self.path, 'r')
        if 'frame_status' not in f:
            f.close()
            raise ValueError(f'{self.path.name}: no frame_status dataset (incomplete or not a skin-UV file)')
        self.attrs = {k: attr_value(f.attrs[k]) for k in f.attrs}
        self.status = f['frame_status'][:].astype(np.uint8)
        self.timestamps = f['source_timestamp_s'][:].astype(np.float64) if 'source_timestamp_s' in f else None
        self.frame_index = f['source_frame_index'][:] if 'source_frame_index' in f else None
        self.small = {name: f[name][:] for name in ('crop_to_frame', 'pear_crop_to_frame', 'crop_box_xyxy', 'pd_cam')
                      if name in f}
        self.params = {g: {k: f[g][k][:] for k in f[g]} for g in ('body_param', 'flame_param') if g in f}
        self.uv_body = f['uv_body_mask'][:].astype(bool) if 'uv_body_mask' in f else None
        self.packed_uv_body = f['packed_uv_body_mask'][:].astype(bool) if 'packed_uv_body_mask' in f else None
        self.frame_files = ([v.decode() if isinstance(v, bytes) else str(v) for v in f['source_frame_files'][:]]
                            if 'source_frame_files' in f else None)  # image sequences: the file of every row
        self._ds = {}
        self.dataset_info = []

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            fast = (obj.chunks == (1,) + obj.shape[1:] and obj.compression in ('gzip', None) and not obj.shuffle
                    and not obj.fletcher32 and obj.scaleoffset is None)
            self._ds[name] = (obj, obj.id, fast, obj.compression == 'gzip', (0,) * (obj.ndim - 1), obj.shape[1:],
                              obj.dtype)
            self.dataset_info.append((name, f'{obj.shape} {obj.dtype}  chunks {obj.chunks}  '
                                            f'{obj.compression or "raw"}  {obj.id.get_storage_size() / 1e6:.1f} MB'))

        f.visititems(visit)

    def close(self):
        h5call(self.h5.close)

    def has(self, item):
        return item in self._ds or f'{item}_packed' in self._ds

    def ok(self, row):
        return int(self.status[row]) == 1

    def status_text(self, row):
        code = int(self.status[row])
        return STATUS_TEXT.get(code, f'status {code}')

    def read_raw(self, name, row):
        ds, dsid, fast, gz, offset, shape, dtype = self._ds[name]
        if fast:
            try:
                filter_mask, raw = h5call(dsid.read_direct_chunk, (int(row),) + offset)
            except Exception:
                raw = None
            if raw is not None and filter_mask == 0:
                data = zlib.decompress(raw) if gz else raw
                return np.frombuffer(data, dtype=dtype).reshape(shape)
        return h5call(ds.__getitem__, int(row))

    def read(self, item, row):
        """Per-row array; '<x>_mask' items are unpacked from '<x>_mask_packed' to bool."""
        if item in self._ds:
            return self.read_raw(item, row)
        if f'{item}_packed' in self._ds:
            return np.unpackbits(self.read_raw(f'{item}_packed', row), axis=-1).view(bool)
        raise Unavailable(f'{item} is not in this file')

    def row_params(self, row):
        return {g: {k: v[row] for k, v in d.items()} for g, d in self.params.items()}

    def texel_tables(self):
        with self._lock:
            if self._texels is None:
                if self.uv_body is None or not all(self.has(name) for name in ('texel_fids', 'texel_bary')):
                    raise Unavailable('UV body mask / texel tables are not in this file')
                self._texels = tuple(h5call(self._ds[name][0].__getitem__, ())
                                     for name in ('texel_fids', 'texel_bary'))
            return self._texels

    def accumulated_final(self):
        with self._lock:
            if self._final is None:
                if 'accumulated_skin_rgb' not in self._ds:
                    raise Unavailable('accumulated_skin_rgb is not in this file')
                rgb = h5call(self._ds['accumulated_skin_rgb'][0].__getitem__, ())
                if 'accumulated_skin_mask_packed' in self._ds:
                    packed = h5call(self._ds['accumulated_skin_mask_packed'][0].__getitem__, ())
                    mask = np.unpackbits(packed, axis=-1).astype(bool)
                else:
                    mask = rgb.any(-1)
                self._final = (rgb, mask)
        return self._final


def resolve_video_path(h5_path, attrs, override=None):
    """Source video file (or image folder) of a skin-UV file: the user's override, the recorded path, the source
    listed in the dataset manifest (vhrm_manifest.json above the file), then the same name next to the file or one
    folder up."""
    from .dataset import source_from_manifest
    candidates = []
    if override:
        candidates.append(Path(override))
    src = attrs.get('source_video_path')
    src = Path(str(src)) if src else None
    if src is not None:
        candidates.append(src)
    listed = source_from_manifest(h5_path)
    if listed is not None:
        candidates.append(listed)
    if src is not None:
        candidates += [Path(h5_path).parent / src.name, Path(h5_path).parent.parent / src.name]
    images = attrs.get('source_type') == 'images'
    for c in candidates:
        if c.is_dir() if images else c.is_file():
            return c
    return None


class ImageSequenceSource:
    """Source frames of an image-sequence file: one image per row (source_frame_files), scaled like VideoSource."""

    def __init__(self, folder, names, attrs, scale=0.5):
        if not names:
            raise RuntimeError('this file does not list its source images (source_frame_files)')
        from .pipeline.frames import read_image_rgb
        self.read = read_image_rgb
        self.path, self.names = Path(folder), list(names)
        full_w, full_h = float(attrs.get('frame_width') or 0), float(attrs.get('frame_height') or 0)
        if not full_w or not full_h:
            first = self.read(self.path / self.names[0])
            full_h, full_w = first.shape[:2]
        self.out_w, self.out_h = even(full_w * scale), even(full_h * scale)
        sx, sy = self.out_w / full_w, self.out_h / full_h
        self.scale_affine = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]])  # frame px -> decoded
        self.size = (self.out_w, self.out_h)

    def frame_for_row(self, row):
        path = self.path / self.names[row]
        if not path.is_file():
            raise Unavailable(f'source image missing: {path.name}')
        image = self.read(path)
        if image.shape[1::-1] != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        return image

    def close(self):
        pass


class VideoSource:
    """PyAV decoder of the source video, addressed by presentation timestamp. Sequential requests decode forward;
    others seek to the previous keyframe first. Frames come out display-oriented and scaled by `scale`."""

    def __init__(self, path, attrs, scale=0.5):
        if av is None:
            raise RuntimeError(_pyav_error_message()) from _AV_IMPORT_ERROR
        self.path = Path(path)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.time_base = float(self.stream.time_base)
        cc = self.stream.codec_context
        stored_w, stored_h = cc.width, cc.height
        self.rot_k = (int(attrs.get('video_rotation', 0)) // 90) % 4
        self.out_w, self.out_h = even(stored_w * scale), even(stored_h * scale)
        disp_w, disp_h = (self.out_h, self.out_w) if self.rot_k % 2 else (self.out_w, self.out_h)
        full_w = float(attrs.get('frame_width') or (stored_h if self.rot_k % 2 else stored_w))
        full_h = float(attrs.get('frame_height') or (stored_w if self.rot_k % 2 else stored_h))
        sx, sy = disp_w / full_w, disp_h / full_h
        self.scale_affine = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]])  # frame px -> decoded
        self.size = (disp_w, disp_h)
        self.reformat = {'width': self.out_w, 'height': self.out_h, 'format': 'rgb24', 'interpolation': 'AREA'}
        try:
            from av.video.reformatter import ColorRange, Colorspace
            self.reformat['src_colorspace'] = (Colorspace.ITU709 if attrs.get('color_matrix') == 'bt709'
                                               else Colorspace.ITU601)
            self.reformat['src_color_range'] = ColorRange.JPEG if attrs.get('color_range') == 'pc' else ColorRange.MPEG
        except Exception:
            pass
        self.tolerance = 0.5 / float(attrs.get('source_fps') or 30.0)
        self._iter = None
        self._last_time = None
        self._last = None

    def close(self):
        self.container.close()

    def _convert(self, frame):
        try:
            img = frame.to_ndarray(**self.reformat)
        except (TypeError, ValueError):
            img = frame.to_ndarray(width=self.out_w, height=self.out_h, format='rgb24')
        if self.rot_k:
            img = np.ascontiguousarray(np.rot90(img, self.rot_k, axes=(0, 1)))
        return img

    def frame_at(self, ts):
        tol = self.tolerance
        if self._last_time is not None and abs(self._last_time - ts) < tol:
            return self._last
        sequential = (self._iter is not None and self._last_time is not None
                      and ts > self._last_time and ts - self._last_time < 3.0)
        if not sequential:
            self.container.seek(max(0, int(ts / self.time_base)), stream=self.stream, backward=True, any_frame=False)
            self._iter = self.container.decode(self.stream)
            self._last_time = None
        for frame in self._iter:
            t = frame.time
            if t is None or t < ts - tol:
                continue
            self._last_time, self._last = t, self._convert(frame)
            return self._last
        self._iter = None
        raise Unavailable('end of the source video reached before this timestamp')


class OrderedWorker:
    """One thread for a stateful per-row producer (video decoder, UV accumulator, mesh). Pending rows are served in
    increasing order from the last produced row, so out-of-order prefetch requests do not cause seeks / restarts.
    With batch > 1, fn receives a list of up to `batch` rows and returns a list of values."""

    def __init__(self, fn, name, cache_rows=32, on_stop=None, batch=1):
        self.fn, self.on_stop, self.batch = fn, on_stop, batch
        self.cache = OrderedDict()
        self.cache_rows = cache_rows
        self.pending = {}
        self.cond = threading.Condition()
        self.position = -1
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def request(self, row):
        with self.cond:
            if row in self.cache:
                self.cache.move_to_end(row)
                fut = Future()
                fut.set_result(self.cache[row])
                return fut
            fut = self.pending.get(row)
            if fut is None:
                fut = self.pending[row] = Future()
                self.cond.notify()
            return fut

    def keep_only(self, rows):
        with self.cond:
            for row in [r for r in self.pending if r not in rows]:
                self.pending.pop(row).cancel()

    def stop(self):
        with self.cond:
            self.stopped = True
            for fut in self.pending.values():
                fut.cancel()
            self.pending.clear()
            self.cond.notify_all()
        self.thread.join(timeout=10)
        if self.on_stop is not None:
            self.on_stop()

    def _run(self):
        while True:
            with self.cond:
                while not self.pending and not self.stopped:
                    self.cond.wait()
                if self.stopped:
                    return
                rows = sorted(self.pending)
                ahead = [r for r in rows if r >= self.position]
                rows = (ahead + [r for r in rows if r < self.position])[:self.batch]
                futs = [self.pending.pop(r) for r in rows]
            live = [(r, f) for r, f in zip(rows, futs) if f.set_running_or_notify_cancel()]
            if not live:
                continue
            try:
                if self.batch > 1:
                    values = self.fn([r for r, _ in live])
                else:
                    values = [self.fn(live[0][0])]
                self.position = live[-1][0]
                with self.cond:
                    for (row, _), value in zip(live, values):
                        self.cache[row] = value
                    while len(self.cache) > self.cache_rows:
                        self.cache.popitem(last=False)
                for (_, fut), value in zip(live, values):
                    fut.set_result(value)
            except BaseException as exc:
                for _, fut in live:
                    fut.set_exception(exc)


class UVAccumulator:
    """Running skin texture (latest observed colour per texel) and the row each texel was last seen as skin, as
    in the notebook's ACCUMULATE. Checkpoints every `every` rows make random access cheap.
    at(row) -> (rgb (S, S, 3) uint8, mask (S, S) uint8 0/1, last_seen_row (S, S) int32, -1 = never)."""

    def __init__(self, src):
        self.src = src
        self.every = max(16, -(-src.n_rows // 24))
        self.ckpts = {}
        self.lock = threading.Lock()
        self.keep_going = lambda row: True  # checked while accumulating; False stops (progress is kept)
        self._reset()

    def _reset(self):
        S = self.src.texture_size
        self.row = -1
        self.rgb = np.zeros((S, S, 3), dtype=np.uint8)
        self.mask = np.zeros((S, S), dtype=np.uint8)
        self.last = np.full((S, S), -1, dtype=np.int32)

    def at(self, row):
        with self.lock:
            if row < self.row or row - self.row > self.every:
                base = max((k for k in list(self.ckpts) if k <= row), default=None)
                if row < self.row or (base is not None and base > self.row):
                    if base is None:
                        self._reset()
                    else:
                        rgb, packed, last = self.ckpts[base]
                        self.row, self.rgb, self.last = base, rgb.copy(), last.copy()
                        self.mask = np.unpackbits(packed, axis=-1)
            while self.row < row:
                if self.row % 16 == 0 and not self.keep_going(row):
                    raise CancelledError()
                self.row += 1
                self._add(self.row, self.rgb, self.mask, self.last)
            return self.rgb.copy(), self.mask.copy(), self.last.copy()

    def _add(self, r, rgb, mask, last):
        src = self.src
        if src.ok(r):
            skin = as_u8(src.read('uv_skin_mask', r))
            cv2.copyTo(src.read('uv_skin_rgb', r), skin, rgb)
            cv2.bitwise_or(mask, skin, dst=mask)
            np.copyto(last, r, where=skin.view(bool))
        if (r + 1) % self.every == 0 and r not in self.ckpts:
            self.ckpts[r] = (rgb.copy(), np.packbits(mask, axis=-1), last.copy())

    def prefill(self, stop_event):
        """Walk the whole file once on private buffers to create every checkpoint (far seeks become cheap)."""
        S = self.src.texture_size
        rgb = np.zeros((S, S, 3), dtype=np.uint8)
        mask = np.zeros((S, S), dtype=np.uint8)
        last = np.full((S, S), -1, dtype=np.int32)
        for r in range(self.src.n_rows):
            if stop_event.is_set():
                return
            self._add(r, rgb, mask, last)
            time.sleep(0.003)  # background priority: leave the GIL and the HDF5 thread to playback


class StatsWorker(threading.Thread):
    """Per-row statistics for the plots. Pass 1 (automatic): skin / visible texel and crop skin pixel counts from
    the packed masks, then the accumulator checkpoints. Pass 2 (on request): mean UV skin colour per CIHP skin
    region."""

    def __init__(self, src, accum=None):
        super().__init__(daemon=True, name='stats')
        self.src, self.accum = src, accum
        n = src.n_rows
        self.counts = {k: np.full(n, np.nan) for k in ('uv_skin', 'uv_visible', 'crop_skin')}
        self.body_texels = int(src.uv_body.sum()) if src.uv_body is not None else src.texture_size ** 2
        self.colour = np.full((n, len(SKIN_REGIONS), 3), np.nan)
        self.colour_texels = np.zeros((n, len(SKIN_REGIONS)))
        self.done = [0, 0]
        self.want_colour = threading.Event()
        self.stop_event = threading.Event()
        self.error = None

    def run(self):
        src = self.src
        try:
            names = {'uv_skin': 'uv_skin_mask_packed', 'uv_visible': 'uv_visible_mask_packed',
                     'crop_skin': 'crop_skin_mask_packed'}
            names = {k: v for k, v in names.items() if v in src._ds}
            for r in range(src.n_rows):
                if self.stop_event.is_set():
                    return
                if src.ok(r):
                    for key, name in names.items():
                        self.counts[key][r] = POPCOUNT[src.read_raw(name, r)].sum()
                self.done[0] = r + 1
            if self.accum is not None and src.has('uv_skin_mask') and src.has('uv_skin_rgb'):
                self.accum.prefill(self.stop_event)
            while not self.want_colour.wait(0.2):
                if self.stop_event.is_set():
                    return
            if not all(src.has(k) for k in ('uv_skin_mask', 'uv_class_ids', 'uv_skin_rgb')):
                raise RuntimeError('uv_skin_mask, uv_class_ids and uv_skin_rgb are needed')
            regions = [list(ids) for ids in SKIN_REGIONS.values()]
            for r in range(src.n_rows):
                if self.stop_event.is_set():
                    return
                if src.ok(r):
                    skin = src.read('uv_skin_mask', r)
                    ids = src.read('uv_class_ids', r)[skin]
                    rgb = src.read('uv_skin_rgb', r)[skin].astype(np.float64)
                    count = np.bincount(ids, minlength=256)
                    sums = np.stack([np.bincount(ids, weights=rgb[:, c], minlength=256) for c in range(3)], 1)
                    for j, cls in enumerate(regions):
                        c = count[cls].sum()
                        self.colour_texels[r, j] = c
                        if c > 0:
                            self.colour[r, j] = sums[cls].sum(0) / c
                self.done[1] = r + 1
        except Exception as exc:
            self.error = exc
            traceback.print_exc()


class MeshManager(QtCore.QObject):
    """Loads mesh.SmplxMesh on demand (background thread) and runs it on one GPU worker thread."""
    stateChanged = Signal()

    def __init__(self, assets_dir, enabled=True):
        super().__init__()
        self.assets_dir = Path(assets_dir).resolve() if assets_dir else None
        self.enabled = enabled
        self.model = None
        self.state, self.error = self._idle_state()

    def _idle_state(self):
        if not self.enabled:
            return 'disabled', 'Mesh views are disabled (Settings)'
        if self.assets_dir is None:
            return 'disabled', 'Body-model assets folder not set (Settings)'
        return 'off', ''

    def configure(self, assets_dir, enabled):
        """Apply new settings. Returns True when a restart is needed: a model from another assets folder is in use."""
        assets_dir = Path(assets_dir).resolve() if assets_dir else None
        restart = (self.model is not None or self.state == 'loading') and assets_dir != self.assets_dir
        self.enabled = enabled
        if self.state == 'loading':
            pass  # _load picks up self.enabled when it finishes
        elif self.model is None:
            self.assets_dir = assets_dir
            self.state, self.error = self._idle_state()
        elif enabled:
            self.state, self.error = 'ready', ''
        else:
            self.state, self.error = 'disabled', 'Mesh views are disabled (Settings)'
        self.stateChanged.emit()
        return restart

    def load(self):
        if self.state != 'off':
            return
        self.state = 'loading'
        self.stateChanged.emit()
        threading.Thread(target=self._load, daemon=True, name='mesh-load').start()

    def _load(self):
        try:
            from .mesh import SmplxMesh
            self.model = SmplxMesh(self.assets_dir)
            self.state, self.error = ('ready', '') if self.enabled else self._idle_state()
        except Exception as exc:
            traceback.print_exc()
            self.state, self.error = 'error', f'mesh model failed to load: {type(exc).__name__}: {exc}'
        self.stateChanged.emit()

    def message(self):
        if not self.enabled:
            return 'Mesh views are disabled (Settings)'
        return {'off': 'SMPL-X model not loaded (Mesh > Load SMPL-X model)',
                'loading': 'Loading the SMPL-X model...'}.get(self.state, self.error)


# ------------------------------------------------------------------------------------------------------- rendering
class RowContext:
    def __init__(self, session, row):
        self.session, self.src, self.row = session, session.src, row
        self.cache = {}
        self.cancelled = False
        self.waiting = []  # slow futures; the row is recomposed when they finish

    def get(self, item):
        if item in self.cache:
            return self.cache[item]
        s, src, row = self.session, self.src, self.row
        if item == 'accum':
            if not (src.has('uv_skin_mask') and src.has('uv_skin_rgb')):
                raise Unavailable('uv_skin_mask / uv_skin_rgb are not in this file')
        elif item in H5_ROW_ITEMS or item in ('mesh', 'standard_mesh'):
            if item not in ('mesh', 'standard_mesh') and not src.has(item):
                raise Unavailable(f'{item} is not in this file')
            if not src.ok(row):
                raise Unavailable(f'row {row}: {src.status_text(row)}')
        try:
            fut = s.raw_future(row, item)
            if item == 'accum':
                try:
                    value = fut.result(timeout=0.3)
                except FutureTimeout:
                    self.waiting.append(fut)
                    raise Unavailable(f'Accumulating the UV texture up to row {row} '
                                      f'(currently at row {s.accum.row})...')
            else:
                value = fut.result()
        except CancelledError:
            self.cancelled = True
            raise Unavailable('cancelled')
        self.cache[item] = value
        return value

    def crop_base(self):
        """Background for crop overlays: the crop cut from the source video, else crop_skin_rgb."""
        if 'crop_base' not in self.cache:
            try:
                self.cache['crop_base'] = video_crop(self)
            except Unavailable:
                self.cache['crop_base'] = self.get('crop_skin_rgb')
        return self.cache['crop_base']


def video_crop(ctx):
    src, row = ctx.src, ctx.row
    if 'crop_to_frame' not in src.small or not src.ok(row):
        raise Unavailable(f'row {row}: {src.status_text(row)}')
    frame = ctx.get('video')
    C = src.crop_size
    m = invert(compose(ctx.session.video.scale_affine, src.small['crop_to_frame'][row]))
    return cv2.warpAffine(frame, m, (C, C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def mesh_layer(ctx, target_from_frame, size):
    """Shaded mesh (RGBA uint8) warped into a target image whose pixels are target_from_frame(frame pixels)."""
    m = ctx.get('mesh')
    if 'rgba' not in m:
        raise Unavailable('PyTorch3D is not available: no mesh raster')
    src, row = ctx.src, ctx.row
    k = float(src.attrs.get('pear_crop_size', 256)) / MESH_RASTER_SIZE
    raster_to_frame = compose(src.small['pear_crop_to_frame'][row], [[k, 0, 0.5 * k], [0, k, 0.5 * k]])
    return cv2.warpAffine(m['rgba'], compose(target_from_frame, raster_to_frame), size, flags=cv2.INTER_LINEAR)


MESH_COLOR = (110, 190, 255)


def overlay_mesh(base, rgba, alpha):
    return blend(base, np.ascontiguousarray(rgba[..., :3]), rgba[..., 3].astype(np.float32) * (alpha / 255.0))


def standard_texture(key, ctx, opts):
    src, row = ctx.src, ctx.row
    background = np.full_like(src.uv_bg, 110)
    if key.startswith('standard_crop_'):
        mesh = ctx.get('mesh')
        fids, bary = src.texel_tables()
        screen = (mesh['screen'][mesh['faces'][fids]] * bary[..., None]).sum(axis=1)
        screen *= float(src.attrs.get('pear_crop_size', 256)) / float(src.attrs.get('render_size', 1024))
        affine = compose(invert(src.small['crop_to_frame'][row]), src.small['pear_crop_to_frame'][row])
        xy = screen @ affine[:, :2].T + affine[:, 2]
        finite = np.isfinite(xy).all(axis=1)
        pixels = np.rint(np.where(np.isfinite(xy), xy, -1)).astype(np.int64)
        inside = finite & ((pixels >= 0) & (pixels < src.crop_size)).all(axis=1)
        valid = ctx.get('uv_visible_mask')[src.uv_body] & inside
        px, py = np.clip(pixels, 0, src.crop_size - 1).T
        if key == 'standard_crop_classes':
            colors = src.colormap[ctx.get('crop_class_ids')[py, px], 0]
        else:
            valid &= ctx.get('crop_skin_mask')[py, px]
            colors = (ctx.get('crop_skin_rgb')[py, px] if key == 'standard_crop_skin_rgb'
                      else np.broadcast_to(np.array([60, 210, 90], dtype=np.uint8), (len(px), 3)))
        values = background[src.uv_body]
        values[valid] = colors[valid]
        background[src.uv_body] = values
        return background
    uv_key = key.removeprefix('standard_')
    if uv_key == 'uv_skin_rgb':
        return masked(ctx.get('uv_skin_rgb'), ctx.get('uv_skin_mask'), background)
    if uv_key in ('uv_skin_mask', 'uv_visible_mask'):
        background[ctx.get(uv_key)] = (60, 210, 90) if uv_key == 'uv_skin_mask' else (40, 90, 210)
        return background
    if uv_key == 'uv_classes':
        return masked(colorize(ctx.get('uv_class_ids'), src.colormap), ctx.get('uv_visible_mask'), background)
    if uv_key in ('uv_accum', 'uv_accum_final'):
        rgb, mask = ctx.get('accum')[:2] if uv_key == 'uv_accum' else src.accumulated_final()
        return masked(rgb, mask, background)
    return render_view(uv_key, ctx, opts).image


def render_view(key, ctx, opts):
    src, row, alpha = ctx.src, ctx.row, opts['alpha']
    uv_bg = src.uv_bg if opts['silhouette'] else np.zeros_like(src.uv_bg)
    class_aux = lambda ids: {'class_ids': ids, 'class_names': src.class_names}

    if key == 'standard' or key.startswith('standard_'):
        verts = ctx.get('standard_mesh')
        model = ctx.session.mesh.model
        texture = None if key == 'standard' else standard_texture(key, ctx, opts)
        orbit = tuple(MESH_ORBIT)
        pan = tuple(MESH_PAN)
        image = model.render_orbit(verts, *orbit, color=(110, 110, 110), texture=texture, pan=pan)
        return MeshResult(verts, model, image, orbit, texture, pan)

    if key.startswith('frame'):
        img = ctx.get('video')
        s_aff = ctx.session.video.scale_affine
        size = ctx.session.video.size
        if key == 'frame':
            img = img.copy()
            if src.ok(row) and 'crop_box_xyxy' in src.small:
                x0, y0, x1, y1 = src.small['crop_box_xyxy'][row]
                p0 = s_aff @ [x0, y0, 1.0]
                p1 = s_aff @ [x1, y1, 1.0]
                cv2.rectangle(img, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), (0, 255, 0),
                              max(2, img.shape[1] // 300))
            return Rendered(img)
        if key == 'frame_mesh':
            return Rendered(overlay_mesh(img, mesh_layer(ctx, s_aff, size), alpha))
        if 'crop_to_frame' not in src.small:
            raise Unavailable('crop_to_frame is not in this file')
        m = compose(s_aff, src.small['crop_to_frame'][row])
        if key == 'frame_skin':
            mask = as_u8(ctx.get('crop_skin_mask')) * np.uint8(255)
            w = cv2.warpAffine(mask, m, size, flags=cv2.INTER_LINEAR)
            return Rendered(blend(img, (0, 255, 80), w.astype(np.float32) * (alpha / 255.0)))
        if key == 'frame_classes':
            ids = ctx.get('crop_class_ids')
            color = cv2.warpAffine(colorize(ids, src.colormap), m, size, flags=cv2.INTER_NEAREST)
            w = cv2.warpAffine((ids > 0).view(np.uint8) * np.uint8(255), m, size, flags=cv2.INTER_LINEAR)
            return Rendered(blend(img, color, w.astype(np.float32) * (alpha / 255.0)))

    if key == 'crop_video':
        return Rendered(video_crop(ctx))
    if key == 'crop_skin_rgb':
        return Rendered(ctx.get('crop_skin_rgb').copy())
    if key == 'crop_skin_mask':
        return Rendered(gray3(ctx.get('crop_skin_mask')))
    if key == 'crop_classes':
        ids = ctx.get('crop_class_ids')
        return Rendered(colorize(ids, src.colormap), class_aux(ids))
    if key == 'crop_classes_overlay':
        ids = ctx.get('crop_class_ids')
        weight = (ids > 0).view(np.uint8).astype(np.float32) * alpha
        return Rendered(blend(ctx.crop_base(), colorize(ids, src.colormap), weight), class_aux(ids))
    if key == 'crop_skin_outline':
        mask = np.ascontiguousarray(as_u8(ctx.get('crop_skin_mask')))
        img = ctx.crop_base().copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 80), 1)
        return Rendered(img)
    if key == 'crop_mesh':
        if 'crop_to_frame' not in src.small:
            raise Unavailable('crop_to_frame is not in this file')
        base = ctx.crop_base()
        C = src.crop_size
        layer = mesh_layer(ctx, invert(src.small['crop_to_frame'][row]), (C, C))
        return Rendered(overlay_mesh(base, layer, alpha))

    if key == 'uv_skin_rgb':
        return Rendered(masked(ctx.get('uv_skin_rgb'), ctx.get('uv_skin_mask'), uv_bg))
    if key == 'packed_uv_skin_rgb':
        rgb = ctx.get('packed_uv_skin_rgb')
        if not opts['silhouette'] or src.packed_uv_body is None:
            return Rendered(rgb.copy())
        background = np.zeros_like(rgb)
        background[src.packed_uv_body] = UV_EMPTY_LEVEL
        skin = ctx.get('packed_uv_skin_mask') if src.has('packed_uv_skin_mask') else rgb.any(-1)
        return Rendered(masked(rgb, skin, background))
    if key == 'uv_coverage':
        code = (as_u8(ctx.get('uv_visible_mask')) << 1) | (as_u8(ctx.get('uv_skin_mask')) << 2)
        if src.uv_body_u8 is not None:
            code |= src.uv_body_u8
        return Rendered(colorize(code, COVERAGE_LUT))
    if key == 'uv_skin_mask':
        return Rendered(gray3(ctx.get('uv_skin_mask')))
    if key == 'uv_visible_mask':
        return Rendered(gray3(ctx.get('uv_visible_mask')))
    if key == 'uv_classes':
        ids, visible = ctx.get('uv_class_ids'), ctx.get('uv_visible_mask')
        return Rendered(masked(colorize(ids, src.colormap), visible, uv_bg), class_aux(ids))
    if key == 'uv_classes_overlay':
        ids, visible = ctx.get('uv_class_ids'), ctx.get('uv_visible_mask')
        base = masked(ctx.get('uv_skin_rgb'), ctx.get('uv_skin_mask'), uv_bg)
        weight = as_u8(visible).astype(np.float32) * alpha
        return Rendered(blend(base, colorize(ids, src.colormap), weight), class_aux(ids))
    if key == 'uv_accum':
        rgb, mask, _ = ctx.get('accum')
        return Rendered(masked(rgb, mask, uv_bg))
    if key == 'uv_age':
        _, mask, last = ctx.get('accum')
        age = (row - last).astype(np.float32) * np.float32(1.0 / src.fps)
        level = cv2.convertScaleAbs(age, alpha=-255.0 / opts['age_window_s'], beta=255.0)  # |255 - 255 age / window|
        level[age >= opts['age_window_s']] = 0
        colored = cv2.cvtColor(cv2.applyColorMap(level, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        age[mask == 0] = np.nan
        return Rendered(masked(colored, mask, uv_bg), {'age_s': age})
    if key == 'uv_accum_final':
        rgb, mask = src.accumulated_final()
        return Rendered(masked(rgb, mask, uv_bg))
    if key == 'mesh3d':
        m = ctx.get('mesh')
        model = ctx.session.mesh.model
        orbit = tuple(MESH_ORBIT)
        pan = tuple(MESH_PAN)
        return MeshResult(m['view'], model, model.render_orbit(m['view'], *orbit, pan=pan), orbit, pan=pan)
    raise Unavailable(f'unknown view {key}')


# --------------------------------------------------------------------------------------------------------- session
class Session(QtCore.QObject):
    """One open file: data sources, compose threads (rows in priority order) and the result cache."""
    composed = Signal(int, int)
    N_COMPOSERS = 3

    def __init__(self, path, mesh, video_scale=0.5, video_override=None):
        super().__init__()
        self.src = SkinUVFile(path)
        self.mesh = mesh
        self.video = self.video_worker = None
        self.video_path = resolve_video_path(path, self.src.attrs, video_override)
        self.video_error = ''
        images = self.src.attrs.get('source_type') == 'images'
        kind = 'images' if images else 'video'
        if av is None and not images:
            self.video_error = f'{_pyav_error_message()}\nSource video is unavailable.'
        elif self.video_path is None:
            self.video_error = (f'source {kind} not found: {self.src.attrs.get("source_video_path", "?")} '
                                '(Locate source)')
        else:
            try:
                if images:
                    self.video = ImageSequenceSource(self.video_path, self.src.frame_files, self.src.attrs,
                                                     video_scale)
                    frame = self.video.frame_for_row
                else:
                    self.video = VideoSource(self.video_path, self.src.attrs, video_scale)
                    ts = self.src.timestamps
                    frame = lambda r: self.video.frame_at(float(ts[r]))  # noqa: E731
                self.video_worker = OrderedWorker(frame, 'video', cache_rows=32, on_stop=self.video.close)
            except Exception as exc:
                self.video = None
                self.video_error = f'cannot open the source {kind}: {exc}'
        self.io_pool = ThreadPoolExecutor(4, thread_name_prefix='h5')
        self.accum = UVAccumulator(self.src)
        self.wanted_set = set()
        self.accum.keep_going = lambda row: not self.stopped and row in self.wanted_set
        self.accum_worker = OrderedWorker(self.accum.at, 'accumulate', cache_rows=8)
        self.mesh_worker = None
        self.standard_worker = None
        self.stats = StatsWorker(self.src, self.accum)
        self.stats.start()
        self.raw = {}
        self.raw_lock = threading.Lock()
        self.results = OrderedDict()
        self.in_progress = set()
        self.cond = threading.Condition()
        self.wanted = []
        self.views = []
        self.opts = {}
        self.version = 0
        self.stopped = False
        self.composers = [threading.Thread(target=self._compose_loop, daemon=True, name=f'composer-{i}')
                          for i in range(self.N_COMPOSERS)]
        for thread in self.composers:
            thread.start()

    # -- configuration / requests (main thread)
    def set_config(self, views, opts):
        with self.cond:
            self.views, self.opts = list(dict.fromkeys(views)), dict(opts)
            self.version += 1
            self.results.clear()
            self.cond.notify_all()

    def invalidate(self):
        with self.raw_lock:
            self.raw = {k: v for k, v in self.raw.items() if k[1] not in ('mesh', 'standard_mesh')}
        self.set_config(self.views, self.opts)

    def set_wanted(self, rows):
        rows = [r for r in dict.fromkeys(rows) if r is not None and 0 <= r < self.src.n_rows]
        with self.cond:
            self.wanted = rows
            self.wanted_set = set(rows)
            self.cond.notify_all()
        items = self._items()
        for row in rows[:PREFETCH_AHEAD]:
            for item in items:
                if item in ('video', 'accum') or self.src.ok(row):
                    try:
                        self.raw_future(row, item)
                    except Unavailable:
                        pass
        keep = set(rows)
        for worker in (self.video_worker, self.accum_worker, self.mesh_worker, self.standard_worker):
            if worker is not None:
                worker.keep_only(keep)
        with self.raw_lock:
            if len(self.raw) > 12 * max(1, len(items)):
                self.raw = {k: v for k, v in self.raw.items() if k[0] in keep}

    def result(self, row):
        with self.cond:
            return self.results.get(row)

    def has_result(self, row):
        with self.cond:
            return row in self.results

    def close(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.stats.stop_event.set()
        for worker in (self.video_worker, self.accum_worker, self.mesh_worker, self.standard_worker):
            if worker is not None:
                worker.stop()
        self.io_pool.shutdown(wait=True, cancel_futures=True)
        for thread in self.composers:
            thread.join(timeout=10)
        self.stats.join(timeout=10)
        self.src.close()

    # -- raw data
    def _items(self):
        items = set()
        for key in self.views:
            view = VIEW_BY_KEY.get(key)
            if view is None:
                continue
            items.update(view.items)
            if view.base:
                items.add('video' if self.video is not None else 'crop_skin_rgb')
        if self.video is None:
            items.discard('video')
        if self.mesh.state != 'ready':
            items.discard('mesh')
            items.discard('standard_mesh')
        if not (self.src.has('uv_skin_mask') and self.src.has('uv_skin_rgb')):
            items.discard('accum')
        return {i for i in items if i in ('video', 'mesh', 'standard_mesh', 'accum') or self.src.has(i)}

    def raw_future(self, row, item):
        with self.raw_lock:
            fut = self.raw.get((row, item))
            stale = fut is not None and (fut.cancelled() or (fut.done() and isinstance(fut.exception(), CancelledError)))
            if fut is not None and not stale:
                return fut
            if item == 'video':
                if self.video_worker is None:
                    raise Unavailable(self.video_error)
                fut = self.video_worker.request(row)
            elif item == 'accum':
                fut = self.accum_worker.request(row)
            elif item == 'standard_mesh':
                if self.mesh.state != 'ready':
                    raise Unavailable(self.mesh.message())
                if not all(group in self.src.params for group in ('body_param', 'flame_param')):
                    raise Unavailable('body_param / flame_param are not in this file')
                if self.standard_worker is None:
                    self.standard_worker = OrderedWorker(self._standard_mesh, 'standard-mesh', cache_rows=16)
                fut = self.standard_worker.request(row)
            elif item == 'mesh':
                if self.mesh.state != 'ready':
                    raise Unavailable(self.mesh.message())
                if 'pd_cam' not in self.src.small or 'pear_crop_to_frame' not in self.src.small:
                    raise Unavailable('pd_cam / pear_crop_to_frame are not in this file')
                if self.mesh_worker is None:
                    self.mesh_worker = OrderedWorker(self._mesh_batch, 'mesh', cache_rows=16, batch=8)
                fut = self.mesh_worker.request(row)
            else:
                fut = self.io_pool.submit(self.src.read, item, row)
            self.raw[(row, item)] = fut
            return fut

    def _standard_mesh(self, row):
        params = self.src.row_params(row)
        return self.mesh.model.compute_standard(params['body_param'], params['flame_param'])

    def _mesh_batch(self, rows):
        src, a = self.src, self.src.attrs
        params = [src.row_params(r) for r in rows]
        outs = self.mesh.model.compute_batch(
            [p['body_param'] for p in params], [p['flame_param'] for p in params],
            [src.small['pd_cam'][r] for r in rows], float(a.get('focal_length', 24.0)),
            float(a.get('render_size', 1024)), MESH_RASTER_SIZE)
        for out in outs:
            if 'shade' in out:
                shade, coverage = out.pop('shade'), out.pop('coverage')
                out['rgba'] = cv2.merge([(shade * c).astype(np.uint8) for c in MESH_COLOR]
                                        + [(coverage * 255).astype(np.uint8)])
        return outs

    # -- compose thread
    def _recompose(self, row, version):
        with self.cond:
            if version == self.version:
                self.results.pop(row, None)
                self.cond.notify_all()

    def _compose_loop(self):
        while True:
            with self.cond:
                while True:
                    if self.stopped:
                        return
                    row = next((r for r in self.wanted if r not in self.results and r not in self.in_progress), None)
                    if row is not None:
                        break
                    self.cond.wait()
                self.in_progress.add(row)
                version, views, opts = self.version, list(self.views), dict(self.opts)
            try:
                ctx = RowContext(self, row)
                out = {}
                for key in views:
                    try:
                        out[key] = render_view(key, ctx, opts)
                    except Unavailable as exc:
                        out[key] = Placeholder(str(exc))
                    except Exception as exc:
                        if self.stopped:
                            return
                        traceback.print_exc()
                        out[key] = Placeholder(f'{type(exc).__name__}: {exc}')
            finally:
                with self.cond:
                    self.in_progress.discard(row)
                    self.cond.notify_all()
            with self.cond:
                if version != self.version or self.stopped or ctx.cancelled:
                    continue
                self.results[row] = out
                keep = set(self.wanted)
                for r in [r for r in self.results if r not in keep][:-4]:
                    del self.results[r]
            for fut in ctx.waiting:
                fut.add_done_callback(lambda _, row=row, version=version: self._recompose(row, version))
            try:
                self.composed.emit(row, version)
            except RuntimeError:  # window already destroyed
                return


# ---------------------------------------------------------------------------------------------------------- panels
class ImagePanel(QtWidgets.QFrame):
    viewChanged = Signal()

    def __init__(self, key, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.combo = QtWidgets.QComboBox()
        self.combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(10)
        space = None
        for view in VIEWS:
            category = 'standard' if view.key.startswith('standard') else view.space
            if space is not None and category != space:
                self.combo.insertSeparator(self.combo.count())
            space = category
            self.combo.addItem(view.label, view.key)
            self.combo.setItemData(self.combo.count() - 1, view.label, Qt.ItemDataRole.ToolTipRole)
        fit = QtWidgets.QToolButton(text='Fit', toolTip='Fit the image (or double-click it)')
        fit.clicked.connect(self.fit)
        save = QtWidgets.QToolButton(text='Save', toolTip='Save this panel image at full resolution')
        save.clicked.connect(self.save_image)
        for b in (fit, save):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.combo, 1)
        self.pan_button = QtWidgets.QToolButton()
        self.pan_button.setIcon(QtGui.QIcon.fromTheme('transform-move',
                       self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp)))
        self.pan_button.setCheckable(True)
        self.pan_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pan_button.setToolTip('Pan mode: drag to translate. Shift-drag or middle/right-drag also pans.')
        header.addWidget(self.pan_button)
        header.addWidget(fit)
        header.addWidget(save)

        self.gw = pg.GraphicsLayoutWidget()
        self.gw.ci.setContentsMargins(0, 0, 0, 0)
        self.vb = self.gw.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.vb.addItem(self.img)
        self.gw.scene().sigMouseMoved.connect(self._hover)
        self.gw.scene().sigMouseClicked.connect(self._clicked)
        self.msg = QtWidgets.QLabel(alignment=Qt.AlignmentFlag.AlignCenter, wordWrap=True)
        self.msg.setStyleSheet('color: #bbb; background: #111;')
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.gw)
        self.stack.addWidget(self.msg)
        self.info = QtWidgets.QLabel(' ')
        self.info.setStyleSheet('color: #aaa; font-size: 8pt;')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        layout.addLayout(header)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.info)

        self._image = self._aux = self._shape = None
        self._mesh = None
        self._drag = None
        self._drag_pan = False
        self.gw.viewport().installEventFilter(self)
        self.set_key(key)
        self.combo.currentIndexChanged.connect(lambda _: self._key_changed())

    @property
    def key(self):
        return self.combo.currentData()

    @property
    def space(self):
        return VIEW_BY_KEY[self.key].space

    def set_key(self, key):
        i = self.combo.findData(key if key in VIEW_BY_KEY else 'uv_skin_rgb')
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        self.combo.setToolTip(self.combo.currentText())
        self.pan_button.setVisible(self.space == '3d')
        self.vb.setMouseEnabled(self.space != '3d', self.space != '3d')

    def _key_changed(self):
        self.combo.setToolTip(self.combo.currentText())
        self.pan_button.setVisible(self.space == '3d')
        self._drag = None
        self.vb.setMouseEnabled(self.space != '3d', self.space != '3d')
        self.viewChanged.emit()

    def clear_mesh(self):
        self._mesh = None

    def show_result(self, res):
        if isinstance(res, Placeholder) or res is None:
            self.msg.setText(res.text if res is not None else '...')
            self.stack.setCurrentWidget(self.msg)
            self._image = None
        elif isinstance(res, MeshResult):
            self._mesh = res
            if res.orbit == tuple(MESH_ORBIT) and res.pan == tuple(MESH_PAN):
                self._set_image(res.image)
            else:
                self._render_mesh()
        else:
            self._set_image(res.image, res.aux)

    def _set_image(self, image, aux=None):
        self.img.setImage(image, autoLevels=False)
        self.stack.setCurrentWidget(self.gw)
        if image.shape[:2] != self._shape:
            self._shape = image.shape[:2]
            self.vb.autoRange(padding=0)
        self._image, self._aux = image, aux

    def _render_mesh(self):
        if self._mesh is None:
            return
        try:
            image = self._mesh.renderer.render_orbit(self._mesh.verts, *MESH_ORBIT, texture=self._mesh.texture,
                                                    pan=tuple(MESH_PAN),
                                                    color=(110, 110, 110) if self.key.startswith('standard')
                                                    else (219, 184, 158))
        except Exception as exc:
            self.msg.setText(f'3D view: {exc}')
            self.stack.setCurrentWidget(self.msg)
            return
        self._set_image(image)

    def eventFilter(self, obj, ev):
        """Orbit controls: drag rotates or pans, wheel zooms, double-click resets."""
        if self.space != '3d' or self._mesh is None:
            return False
        t = ev.type()
        if t == QtCore.QEvent.Type.MouseButtonDblClick:
            MESH_ORBIT[:] = [0.0, 0.0, 1.0]
            MESH_PAN[:] = [0.0, 0.0]
            self._drag = None
        elif t == QtCore.QEvent.Type.MouseButtonPress and ev.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._drag = ev.position()
            self._drag_pan = (self.pan_button.isChecked() or ev.button() != Qt.MouseButton.LeftButton
                              or bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            return True
        elif t == QtCore.QEvent.Type.MouseMove and self._drag is not None:
            d = ev.position() - self._drag
            self._drag = ev.position()
            if self._drag_pan:
                scale = self.vb.sceneBoundingRect()
                MESH_PAN[0] += d.x() / max(scale.width(), 1.0)
                MESH_PAN[1] += d.y() / max(scale.height(), 1.0)
            else:
                MESH_ORBIT[0] -= 0.5 * d.x()
                MESH_ORBIT[1] = float(np.clip(MESH_ORBIT[1] + 0.5 * d.y(), -89, 89))
        elif t == QtCore.QEvent.Type.MouseButtonRelease:
            self._drag = None
            return True
        elif t == QtCore.QEvent.Type.Wheel:
            MESH_ORBIT[2] = float(np.clip(MESH_ORBIT[2] * 1.15 ** (ev.angleDelta().y() / 120), 0.2, 20))
        else:
            return False
        self._render_mesh()
        return True

    def fit(self):
        if self.space == '3d':
            MESH_ORBIT[:] = [0.0, 0.0, 1.0]
            MESH_PAN[:] = [0.0, 0.0]
            self._render_mesh()
        if self.stack.currentWidget() is self.gw:
            self.vb.autoRange(padding=0)

    def _clicked(self, ev):
        if ev.double():
            self.fit()

    def _hover(self, pos):
        if self._image is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        x, y = int(math.floor(p.x())), int(math.floor(p.y()))
        h, w = self._image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            self.info.setText(' ')
            return
        text = f'x {x}  y {y}  rgb {tuple(int(v) for v in self._image[y, x])}'
        aux = self._aux or {}
        if 'class_ids' in aux:
            cid = int(aux['class_ids'][y, x])
            names = aux['class_names']
            text += f'  class {cid} {names[cid] if cid < len(names) else ""}'
        if 'age_s' in aux:
            age = aux['age_s'][y, x]
            text += '  never seen' if np.isnan(age) else f'  last seen {age:.2f} s ago'
        self.info.setText(text)

    def current_image(self):
        return self._image

    def save_image(self):
        if self._image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save panel image', f'{self.key}.png', 'PNG (*.png)')
        if path:
            cv2.imwrite(path, cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------------------------------------------ file browser
def scan_file(path, root):
    path = Path(path)
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    info = {'path': str(path), 'name': rel, 'size_mb': path.stat().st_size / 1e6}
    try:
        with h5py.File(path, 'r') as f:
            if 'frame_status' not in f:
                info['error'] = 'incomplete or not a skin-UV file'
                return info
            a = f.attrs
            status = f['frame_status'][:]
            ts = f['source_timestamp_s'][:] if 'source_timestamp_s' in f else None
            fps = float(a.get('effective_fps', np.nan))
            source_attrs = {k: attr_value(a[k]) for k in ('source_video_path', 'source_type') if k in a}
            info.update(rows=len(status), ok=int((status == 1).sum()), fps=fps,
                        mode=attr_value(a.get('frame_selection_mode', '')),
                        duration=float(ts[-1] - ts[0]) if ts is not None and len(ts) > 1 else len(status) / fps,
                        video=resolve_video_path(path, source_attrs) is not None,
                        source_type=source_attrs.get('source_type', 'video'))
    except Exception as exc:
        info['error'] = f'unreadable ({type(exc).__name__})'
    return info


class ScanWorker(QtCore.QThread):
    """Lists *.h5 under root (recursively) and reads each file's summary, off the GUI thread."""
    listed = Signal(int)
    found = Signal(dict)

    def __init__(self, root, extra):
        super().__init__()
        self.root, self.extra = root, extra

    def run(self):
        paths = []
        if self.root is not None:
            for p in self.root.rglob('*.h5'):
                if self.isInterruptionRequested():
                    return
                paths.append(str(p.resolve()))
        paths = sorted(paths) + [p for p in self.extra if p not in paths]
        self.listed.emit(len(paths))
        for p in paths:
            if self.isInterruptionRequested():
                return
            self.found.emit(h5call(scan_file, p, self.root or Path(p).parent))


class SortItem(QtWidgets.QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        a, b = self.data(col, Qt.ItemDataRole.UserRole), other.data(col, Qt.ItemDataRole.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return self.text(col).lower() < other.text(col).lower()


class FileBrowser(QtWidgets.QWidget):
    openRequested = Signal(str)
    rootChanged = Signal(str)
    COLUMNS = ['File', 'Rows', 'OK', 'FPS', 'Mode', 'Dur. (s)', 'MB', 'Source']
    MAX_RECENT = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_edit = QtWidgets.QLineEdit(placeholderText='folder with HDF5 files')
        self.root_edit.returnPressed.connect(self.refresh)
        browse = QtWidgets.QToolButton(text='Browse...', toolTip='Choose the folder to scan (recursively)')
        browse.clicked.connect(self.browse)
        self.recent = []
        self.recent_menu = QtWidgets.QMenu('Recent folders', self)
        recent = QtWidgets.QToolButton(text='Recent', toolTip='Recently opened folders',
                                       popupMode=QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        recent.setMenu(self.recent_menu)
        refresh = QtWidgets.QToolButton(text='Refresh', toolTip='Rescan the folder (F5)')
        refresh.clicked.connect(self.refresh)
        self.filter_edit = QtWidgets.QLineEdit(placeholderText='filter file names')
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(self.COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemActivated.connect(lambda item, _: self._open(item))
        self.tree.currentItemChanged.connect(lambda item, _: self._open(item))
        self.count_label = QtWidgets.QLabel()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.root_edit, 1)
        top.addWidget(browse)
        top.addWidget(recent)
        top.addWidget(refresh)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(top)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.count_label)
        self.worker = None
        self.extra = []
        self.current_path = None
        self.set_recent([])

    def set_root(self, root, extra=()):
        self.root_edit.setText(str(root) if root else '')
        self.extra = [str(Path(p).resolve()) for p in extra]
        self.refresh()

    def browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder with HDF5 files', self.root_edit.text())
        if d:
            self.set_root(d)

    def set_recent(self, folders):
        self.recent = [str(f) for f in folders][:self.MAX_RECENT]
        self.recent_menu.clear()
        for folder in self.recent:
            self.recent_menu.addAction(folder, lambda f=folder: self.set_root(f))
        if self.recent:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction('Clear list', lambda: self.set_recent([]))
        else:
            self.recent_menu.addAction('(none)').setEnabled(False)

    def refresh(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()
            self.worker = None
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        text = self.root_edit.text().strip()
        root = Path(text).resolve() if text else None
        if root is not None and not root.is_dir():
            self.count_label.setText(f'Folder not found: {root}')
            root = None
        elif root is None:
            self.count_label.setText('Choose a folder with HDF5 files (Browse... or File > Open folder)')
        else:
            self.root_edit.setText(str(root))
            self.set_recent([str(root)] + [f for f in self.recent if f != str(root)])
            self.rootChanged.emit(str(root))
            self.count_label.setText('listing files...')
        if root is None and not self.extra:
            return
        self.worker = worker = ScanWorker(root, self.extra)
        # queued signals of a previous scan can still arrive after a refresh: only accept the current worker's
        worker.listed.connect(lambda n: self.worker is worker and self.count_label.setText(f'scanning {n} files...'))
        worker.found.connect(lambda info: self.worker is worker and self._add(info))
        worker.finished.connect(lambda: self.worker is worker and
                                self.count_label.setText(f'{self.tree.topLevelItemCount()} files'))
        worker.start()

    def _add(self, info):
        item = SortItem()
        item.setText(0, info['name'])
        item.setToolTip(0, info['path'])
        item.setData(0, Qt.ItemDataRole.UserRole + 1, info['path'])
        if 'error' in info:
            item.setText(1, info['error'])
            item.setForeground(1, QtGui.QBrush(QtGui.QColor('#c66')))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        else:
            values = [info['rows'], info['ok'], info['fps'], info['mode'], info['duration'], info['size_mb'],
                      ('images' if info['source_type'] == 'images' else 'video') if info['video'] else 'missing']
            texts = [str(info['rows']), str(info['ok']), f'{info["fps"]:.2f}', str(info['mode']),
                     f'{info["duration"]:.1f}', f'{info["size_mb"]:.0f}', values[-1]]
            for col, (text, value) in enumerate(zip(texts, values), start=1):
                item.setText(col, text)
                item.setData(col, Qt.ItemDataRole.UserRole, value)
            if info['ok'] < info['rows']:
                item.setForeground(2, QtGui.QBrush(QtGui.QColor('#da4')))
        self.tree.blockSignals(True)
        self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)
        for col in range(1, len(self.COLUMNS)):
            self.tree.resizeColumnToContents(col)
        self._apply_filter()
        if info['path'] == self.current_path:
            self.mark_current(self.current_path)

    def _apply_filter(self):
        text = self.filter_edit.text().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(bool(text) and text not in item.text(0).lower())

    def _open(self, item):
        if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        path = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if path != self.current_path:
            self.openRequested.emit(path)

    def mark_current(self, path):
        self.current_path = path
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            current = item.data(0, Qt.ItemDataRole.UserRole + 1) == path
            font = item.font(0)
            font.setBold(current)
            item.setFont(0, font)
            if current and self.tree.currentItem() is not item:
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(item)
                self.tree.blockSignals(False)

    def step(self, delta):
        items = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        items = [it for it in items if not it.isHidden() and it.flags() & Qt.ItemFlag.ItemIsEnabled]
        items.sort(key=lambda it: self.tree.indexOfTopLevelItem(it))
        if not items:
            return
        paths = [it.data(0, Qt.ItemDataRole.UserRole + 1) for it in items]
        i = paths.index(self.current_path) + delta if self.current_path in paths else 0
        if 0 <= i < len(items):
            self.tree.setCurrentItem(items[i])


# ------------------------------------------------------------------------------------------------------ export
class Mp4Writer:
    def __init__(self, path, fps, width, height):
        self.size = (width, height)
        self.container = self.stream = self.cv = None
        rate = Fraction(fps).limit_denominator(1001)
        if av is None:
            warnings.warn(f'{_pyav_error_message()}\nMP4 export will use the OpenCV mp4v fallback.',
                          RuntimeWarning, stacklevel=2)
        if av is not None:
            for codec, options in (('libx264', {'crf': '18', 'preset': 'medium'}), ('h264_mf', {}),
                                   ('mpeg4', {'qscale': '3'})):
                try:
                    av.Codec(codec, 'w')
                except Exception:
                    continue
                self.container = av.open(str(path), 'w')
                self.stream = self.container.add_stream(codec, rate=rate)
                self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, 'yuv420p'
                self.stream.options = options
                if codec != 'libx264':
                    self.stream.bit_rate = 20_000_000
                self.codec = codec
                return
        self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height))
        self.codec = 'opencv mp4v'

    def write(self, rgb):
        if rgb.shape[1::-1] != self.size:
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        if self.cv is not None:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
            self.container.mux(packet)

    def close(self):
        if self.cv is not None:
            self.cv.release()
            return
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent, default_path, fps, has_marks):
        super().__init__(parent)
        self.setWindowTitle('Export the panel grid as a video')
        self.range_combo = QtWidgets.QComboBox()
        self.range_combo.addItem('Whole file', 'all')
        if has_marks:
            self.range_combo.addItem('In / out marks', 'marks')
        self.range_combo.addItem('From the current row to the end', 'current')
        if has_marks:
            self.range_combo.setCurrentIndex(1)
        self.fps = QtWidgets.QDoubleSpinBox(decimals=3, minimum=1, maximum=240, value=fps)
        self.path_edit = QtWidgets.QLineEdit(default_path)
        browse = QtWidgets.QToolButton(text='...')
        browse.clicked.connect(self._browse)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form = QtWidgets.QFormLayout(self)
        form.addRow('Rows', self.range_combo)
        form.addRow('Frame rate', self.fps)
        form.addRow('Output', path_row)
        form.addRow(QtWidgets.QLabel('The panels are captured as shown (size, zoom, 3D camera). '
                                     'Do not resize the window during the export.', wordWrap=True))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Output video', self.path_edit.text(), 'MP4 (*.mp4)')
        if path:
            self.path_edit.setText(path)


VIDEO_SCALES = (1.0, 0.75, 0.5, 0.25)


class SettingsDialog(QtWidgets.QDialog):
    """Model and tool locations, and the run-time equivalents of the command-line options (saved between sessions)."""

    def __init__(self, parent, values):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumWidth(720)
        self.assets_edit = QtWidgets.QLineEdit(str(values.get('assets_dir') or ''),
                                               placeholderText='folder holding SMPLX/ and FLAME/ (PEAR "assets")')
        self.assets_status = QtWidgets.QLabel(wordWrap=True)
        self.assets_edit.textChanged.connect(self._check)
        detect = QtWidgets.QToolButton(text='Detect', toolTip='Search $VHRM_ASSETS_DIR, the working directory and '
                                                              'the folders above it')
        detect.clicked.connect(self._detect)
        self.ffmpeg_edit = QtWidgets.QLineEdit(str(values.get('ffmpeg_dir') or ''),
                                               placeholderText='empty: ffmpeg / ffprobe on PATH')
        self.ffmpeg_status = QtWidgets.QLabel(wordWrap=True)
        self.ffmpeg_edit.textChanged.connect(self._check)
        self.checkpoint_edit = QtWidgets.QLineEdit(str(values.get('checkpoint') or ''), placeholderText=(
            'empty: <assets>/../weights/ehm_model_stage1.pt, else downloaded once from Hugging Face'))
        self.yolo_edit = QtWidgets.QLineEdit(str(values.get('yolo_weights') or ''),
                                             placeholderText='empty: yolov8x.pt downloaded once by ultralytics')
        self.mesh_check = QtWidgets.QCheckBox('Enable the SMPL-X mesh views', checked=values.get('mesh_enabled', True))
        self.load_check = QtWidgets.QCheckBox('Load the SMPL-X model at start-up (about 20 s, uses the GPU)',
                                              checked=values.get('load_at_start', False))
        self.scale_combo = QtWidgets.QComboBox()
        for s in VIDEO_SCALES:
            self.scale_combo.addItem(f'{s:g}', s)
        self.scale_combo.setCurrentIndex(max(0, self.scale_combo.findData(values.get('video_scale', 0.5))))

        def row(edit, title, folder=True, extra=()):
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(edit, 1)
            browse = QtWidgets.QToolButton(text='Browse...')
            browse.clicked.connect(lambda: self._browse(edit, title, folder))
            layout.addWidget(browse)
            for widget in extra:
                layout.addWidget(widget)
            return layout

        form = QtWidgets.QFormLayout(self)
        form.addRow('Body-model assets', row(self.assets_edit, 'SMPL-X / FLAME assets folder', extra=(detect,)))
        form.addRow('', self.assets_status)
        form.addRow('ffmpeg folder', row(self.ffmpeg_edit, 'Folder of ffmpeg and ffprobe'))
        form.addRow('', self.ffmpeg_status)
        form.addRow('PEAR checkpoint', row(self.checkpoint_edit, 'PEAR checkpoint (ehm_model_stage1.pt)', False))
        form.addRow('YOLO weights', row(self.yolo_edit, 'YOLOv8x weights (yolov8x.pt)', False))
        form.addRow('Mesh', self.mesh_check)
        form.addRow('', self.load_check)
        form.addRow('Video decode scale', self.scale_combo)
        from .assets import ASSETS_HELP
        form.addRow(QtWidgets.QLabel(ASSETS_HELP + ' The assets are needed by the dataset conversion and the mesh '
                                     'views. Command-line options override these settings for one session.',
                                     wordWrap=True))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._check()

    def _assets_dir(self):
        from .assets import is_assets_dir
        text = self.assets_edit.text().strip()
        return Path(text).resolve() if text and is_assets_dir(text) else None

    def _check(self):
        from .assets import missing_assets
        text = self.assets_edit.text().strip()
        missing = missing_assets(text) if text else None
        if text and not missing:
            self.assets_status.setText('<span style="color:#4a4">SMPL-X / FLAME assets found</span>')
        elif text:
            self.assets_status.setText(f'<span style="color:#c55">Incomplete: missing {", ".join(missing[:4])}'
                                       f'{" ..." if len(missing) > 4 else ""}</span>')
        else:
            self.assets_status.setText('Not set: no dataset conversion and no mesh views')
        from .pipeline import frames
        frames._FFMPEG.clear()
        try:
            found = [frames.find_tool(name, self.ffmpeg_edit.text().strip() or None) for name in ('ffmpeg', 'ffprobe')]
            self.ffmpeg_status.setText(f'<span style="color:#4a4">{found[0]}</span>')
        except RuntimeError as exc:
            self.ffmpeg_status.setText(f'<span style="color:#c55">{exc}</span>')

    def _browse(self, edit, title, folder):
        if folder:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, title, edit.text())
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, edit.text(), 'Weights (*.pt *.pth);;All (*)')
        if path:
            edit.setText(path)

    def _detect(self):
        from .assets import find_assets_dir
        found = find_assets_dir()
        if found is None:
            self.assets_status.setText('<span style="color:#c55">No assets folder found automatically</span>')
        else:
            self.assets_edit.setText(str(found))

    def values(self):
        return {'assets_dir': self._assets_dir(), 'ffmpeg_dir': self.ffmpeg_edit.text().strip(),
                'checkpoint': self.checkpoint_edit.text().strip(), 'yolo_weights': self.yolo_edit.text().strip(),
                'mesh_enabled': self.mesh_check.isChecked(), 'load_at_start': self.load_check.isChecked(),
                'video_scale': float(self.scale_combo.currentData())}


def migrate_settings(settings):
    """Copy the settings of the former Skin-UV viewer once (window layout, folders, panels, ...)."""
    if settings.value('migrated_skin_uv_viewer', False, type=bool):
        return
    old = QtCore.QSettings('vHRM', 'SkinUVViewer')
    for key in old.allKeys():
        if not settings.contains(key) and key != 'pear_dir':
            settings.setValue(key, old.value(key))
    pear_dir = old.value('pear_dir', '') or ''
    if pear_dir and not settings.value('assets_dir', ''):
        from .assets import is_assets_dir
        candidate = Path(pear_dir).parent / 'assets'
        if is_assets_dir(candidate):
            settings.setValue('assets_dir', str(candidate))
    settings.setValue('migrated_skin_uv_viewer', True)


# ------------------------------------------------------------------------------------------------------ main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('vHRM Studio')
        self.settings = QtCore.QSettings('vHRM', 'vHRM Studio')
        migrate_settings(self.settings)
        self.video_overrides = json.loads(self.settings.value('video_overrides', '{}') or '{}')
        from .assets import find_assets_dir, is_assets_dir
        saved_assets = self.settings.value('assets_dir', '') or ''
        self.assets_dir = (args.assets or (saved_assets if saved_assets and is_assets_dir(saved_assets) else None)
                           or find_assets_dir())
        if self.assets_dir and not saved_assets and not args.assets:
            self.settings.setValue('assets_dir', str(self.assets_dir))  # found automatically: keep it
        self.converter = None
        mesh_enabled = not args.no_mesh and self.settings.value('mesh_enabled', True, type=bool)
        self.load_mesh_at_start = self.settings.value('load_mesh_at_start', False, type=bool)
        self.mesh = MeshManager(self.assets_dir, enabled=mesh_enabled)
        self.mesh.stateChanged.connect(self._mesh_state_changed)
        self.session = None
        self.row = 0
        self.shown_row = None
        self.mark_in = self.mark_out = None
        self.playing = False
        self.exporting = False
        self.play_t0 = 0.0
        self.play_steps = 0
        self.video_scale = float(args.video_scale or self.settings.value('video_scale', 0.5))
        self.panel_keys = json.loads(self.settings.value('panels', 'null') or 'null') or list(DEFAULT_PANELS)
        self.panels = []
        self._stats_seen = (-1, -1)

        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._build_shortcuts()
        self.layout_combo.setCurrentText(self.settings.value('layout', '2x3'))
        self._rebuild_grid()

        self.play_timer = QtCore.QTimer(self, timerType=Qt.TimerType.PreciseTimer, interval=4)
        self.play_timer.timeout.connect(self._tick)
        self.stats_timer = QtCore.QTimer(self, interval=400)
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start()

        geometry = self.settings.value('geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1700, 1000)
        state = self.settings.value('window_state')
        if state is not None:
            self.restoreState(state)
        else:
            self.resizeDocks([self.files_dock, self.info_dock], [380, 340], Qt.Orientation.Horizontal)
            self.resizeDocks([self.plots_dock], [230], Qt.Orientation.Vertical)

        files = [Path(p) for p in args.paths if Path(p).is_file()]
        dirs = [Path(p) for p in args.paths if Path(p).is_dir()]
        saved_root = self.settings.value('root', '') or ''
        root = dirs[0] if dirs else (files[0].parent if files else (saved_root if Path(saved_root).is_dir() else None))
        self.browser.set_recent(json.loads(self.settings.value('recent_roots', '[]') or '[]'))
        self.browser.set_root(root, extra=files)
        if files:
            self.open_file(str(files[0].resolve()))
        if args.load_mesh or self.load_mesh_at_start:
            self.mesh.load()
        if args.convert:
            QtCore.QTimer.singleShot(0, self.open_converter)

    # ------------------------------------------------------------------ UI construction
    def _build_toolbar(self):
        tb = self.addToolBar('View')
        tb.setObjectName('view_toolbar')
        tb.addWidget(QtWidgets.QLabel(' Layout '))
        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItems(list(LAYOUTS))
        self.layout_combo.currentTextChanged.connect(lambda _: self._rebuild_grid())
        tb.addWidget(self.layout_combo)
        self.link_check = QtWidgets.QCheckBox('Link zoom', checked=True,
                                              toolTip='Panels showing the same space (frame / crop / UV) share zoom')
        self.link_check.toggled.connect(lambda _: self._apply_links())
        tb.addWidget(self.link_check)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(' Overlay '))
        self.alpha_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=50)
        self.alpha_slider.setFixedWidth(110)
        self.alpha_slider.setToolTip('Overlay opacity')
        self.alpha_slider.valueChanged.connect(lambda _: self._push_config())
        tb.addWidget(self.alpha_slider)
        self.silhouette_check = QtWidgets.QCheckBox('UV silhouette', checked=True,
                                                    toolTip='Draw body texels without data in dark grey')
        self.silhouette_check.toggled.connect(lambda _: self._push_config())
        tb.addWidget(self.silhouette_check)
        tb.addWidget(QtWidgets.QLabel(' Age window (s) '))
        self.age_spin = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=600, value=5.0, decimals=1, singleStep=1.0)
        self.age_spin.setToolTip('Colour scale of the "time since texel last seen" view')
        self.age_spin.valueChanged.connect(lambda _: self._push_config())
        tb.addWidget(self.age_spin)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(' Video scale '))
        self.scale_combo = QtWidgets.QComboBox()
        for s in VIDEO_SCALES:
            self.scale_combo.addItem(f'{s:g}', s)
        self.scale_combo.setCurrentIndex(max(0, self.scale_combo.findData(self.video_scale)))
        self.scale_combo.setToolTip('Decode resolution of the source video (lower = faster playback)')
        self.scale_combo.currentIndexChanged.connect(self._video_scale_changed)
        tb.addWidget(self.scale_combo)
        tb.addSeparator()
        self.mesh_action = tb.addAction('Load SMPL-X', self.mesh.load)
        self.mesh_action.setToolTip('Load the SMPL-X body model (torch, ~20 s) for the mesh views')
        self.mesh_label = QtWidgets.QLabel()
        tb.addWidget(self.mesh_label)
        tb.addSeparator()
        tb.addAction('Snapshot', self.snapshot).setToolTip('Save the panel grid as PNG (Ctrl+S)')
        tb.addAction('Export video', self.export_video).setToolTip('Export the panel grid as MP4 (Ctrl+E)')
        tb.addAction('Locate source', self.locate_video).setToolTip('Choose the source video (or image folder) of '
                                                                    'the current file')
        tb.addSeparator()
        convert = tb.addAction('Convert dataset', self.open_converter)
        convert.setToolTip('Convert a folder of videos / image sequences into skin-UV HDF5 files (Ctrl+Shift+C)')
        tb.addAction('Settings', self.open_settings).setToolTip('Model and ffmpeg locations, mesh views, video scale '
                                                                '(Ctrl+,)')

        menu = self.menuBar()
        file_menu = menu.addMenu('&File')

        def add(menu_, text, slot, shortcut=None):
            action = menu_.addAction(text, slot)
            if shortcut:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            return action

        add(file_menu, 'Open folder...', self._open_folder, 'Ctrl+O')
        self.recent_menu = file_menu.addMenu('Recent folders')
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        add(file_menu, 'Open file...', self._open_dialog, 'Ctrl+Shift+O')
        add(file_menu, 'Refresh file list', lambda: self.browser.refresh(), 'F5')
        add(file_menu, 'Locate source video / images...', self.locate_video)
        file_menu.addSeparator()
        add(file_menu, 'Snapshot (PNG)...', self.snapshot)
        add(file_menu, 'Export video (MP4)...', self.export_video)
        file_menu.addSeparator()
        add(file_menu, 'Settings...', self.open_settings, 'Ctrl+,')
        file_menu.addSeparator()
        add(file_menu, 'Quit', self.close)
        dataset_menu = menu.addMenu('&Dataset')
        add(dataset_menu, 'Convert a dataset...', self.open_converter, 'Ctrl+Shift+C')
        add(dataset_menu, 'Open the dataset folder of this file', self._open_dataset_root)
        mesh_menu = menu.addMenu('&Mesh')
        self.mesh_load_action = mesh_menu.addAction('Load SMPL-X model', self.mesh.load)
        self.mesh_enable_action = mesh_menu.addAction('Enable mesh views')
        self.mesh_enable_action.setCheckable(True)
        self.mesh_enable_action.setChecked(self.mesh.enabled)
        self.mesh_enable_action.toggled.connect(self._set_mesh_enabled)
        mesh_menu.addAction('Set the body-model assets folder...', self.open_settings)
        self._mesh_state_changed()

    def _build_central(self):
        self.grid_widget = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)

        def button(text, slot, tip):
            b = QtWidgets.QToolButton(text=text, toolTip=tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            return b

        self.play_button = button('Play', self.toggle_play, 'Play / pause (Space)')
        self.play_button.setMinimumWidth(50)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(lambda v: self.seek(v) if v != self.row else None)
        self.row_spin = QtWidgets.QSpinBox(keyboardTracking=False)
        self.row_spin.valueChanged.connect(lambda v: self.seek(v) if v != self.row else None)
        self.speed_combo = QtWidgets.QComboBox()
        for s in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f'{s:g}x', s)
        self.speed_combo.setCurrentIndex(3)
        self.speed_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_combo.currentIndexChanged.connect(lambda _: self._restart_clock())
        self.loop_check = QtWidgets.QCheckBox('Loop', checked=True)
        self.realtime_check = QtWidgets.QCheckBox('Real time', checked=True,
                                                  toolTip='Skip rows to keep the playback speed when decoding is slow')
        for w in (self.loop_check, self.realtime_check):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.realtime_check.toggled.connect(lambda _: self._restart_clock())
        self.marks_label = QtWidgets.QLabel()
        self.time_label = QtWidgets.QLabel()
        self.time_label.setMinimumWidth(420)
        self.fps_label = QtWidgets.QLabel()
        self.fps_label.setStyleSheet('color: #888;')

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(button('|<', lambda: self.seek(self._range()[0]), 'First row (Home)'))
        bar.addWidget(button('<', lambda: self.step(-1), 'Previous row (Left)'))
        bar.addWidget(self.play_button)
        bar.addWidget(button('>', lambda: self.step(1), 'Next row (Right)'))
        bar.addWidget(button('>|', lambda: self.seek(self._range()[1]), 'Last row (End)'))
        bar.addWidget(self.slider, 1)
        bar.addWidget(QtWidgets.QLabel('row'))
        bar.addWidget(self.row_spin)
        bar.addWidget(self.speed_combo)
        bar.addWidget(self.loop_check)
        bar.addWidget(self.realtime_check)
        bar.addWidget(button('I', self.set_in, 'Set the in mark (I)'))
        bar.addWidget(button('O', self.set_out, 'Set the out mark (O)'))
        bar.addWidget(button('x', self.clear_marks, 'Clear the marks (Ctrl+I)'))
        bar.addWidget(self.marks_label)
        info = QtWidgets.QHBoxLayout()
        info.addWidget(self.time_label, 1)
        info.addWidget(self.fps_label)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.grid_widget, 1)
        layout.addLayout(bar)
        layout.addLayout(info)
        self.setCentralWidget(central)

    def _build_docks(self):
        self.browser = FileBrowser()
        self.browser.openRequested.connect(self.open_file)
        self.browser.rootChanged.connect(self._root_changed)
        self.files_dock = dock = QtWidgets.QDockWidget('Files', self, objectName='files_dock')
        dock.setWidget(self.browser)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.frame_tree = QtWidgets.QTreeWidget()
        self.frame_tree.setHeaderLabels(['Field', 'Value'])
        self.file_tree = QtWidgets.QTreeWidget()
        self.file_tree.setHeaderLabels(['Name', 'Value'])
        self.info_tabs = QtWidgets.QTabWidget()
        self.info_tabs.addTab(self.frame_tree, 'Row')
        self.info_tabs.addTab(self.file_tree, 'File')
        self.info_tabs.currentChanged.connect(lambda _: self._update_frame_info())
        self.info_dock = QtWidgets.QDockWidget('Details', self, objectName='info_dock')
        self.info_dock.setWidget(self.info_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_dock)

        # Timeline: per-row coverage counts
        self.timeline = pg.PlotWidget()
        self.timeline.setLabel('bottom', 'time', 's')
        self.timeline.setLabel('left', '%')
        self.timeline.showGrid(x=True, y=True, alpha=0.2)
        self.timeline.addLegend(offset=(-10, 5))
        self.tl_curves = {
            'uv_visible': self.timeline.plot(pen=pg.mkPen((70, 120, 230), width=1.5), name='UV visible / body texels'),
            'uv_skin': self.timeline.plot(pen=pg.mkPen((70, 210, 100), width=1.5), name='UV skin / body texels'),
            'crop_skin': self.timeline.plot(pen=pg.mkPen((230, 160, 60), width=1.5), name='crop skin / crop pixels'),
        }
        self.tl_bad = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(220, 60, 60), pen=None, name='failed rows')
        self.timeline.addItem(self.tl_bad)
        # Colour signals: mean UV skin colour per region
        self.signals = pg.PlotWidget()
        self.signals.setLabel('bottom', 'time', 's')
        self.signals.showGrid(x=True, y=True, alpha=0.2)
        self.signals.addLegend(offset=(-10, 5))
        self.sig_curves = [self.signals.plot(pen=pg.mkPen(c, width=1.2), name=n)
                           for c, n in (((230, 70, 70), 'R'), ((70, 200, 90), 'G'), ((80, 120, 240), 'B'))]
        self.region_combo = QtWidgets.QComboBox()
        self.region_combo.addItems(list(SKIN_REGIONS))
        self.region_combo.currentIndexChanged.connect(lambda _: self._refresh_stats(force=True))
        self.sig_mode = QtWidgets.QComboBox()
        self.sig_mode.addItems(['mean level', 'relative (x / mean - 1)', 'texel count'])
        self.sig_mode.currentIndexChanged.connect(lambda _: self._refresh_stats(force=True))
        self.sig_button = QtWidgets.QPushButton('Compute colour signals')
        self.sig_button.setToolTip('Reads uv_skin_rgb, uv_class_ids and the skin mask of every row (background)')
        self.sig_button.clicked.connect(self._start_colour)
        self.sig_status = QtWidgets.QLabel()
        sig_widget = QtWidgets.QWidget()
        sig_layout = QtWidgets.QVBoxLayout(sig_widget)
        sig_layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel('Region'), self.region_combo, QtWidgets.QLabel('Show'), self.sig_mode,
                  self.sig_button, self.sig_status):
            row.addWidget(w)
        row.addStretch(1)
        sig_layout.addLayout(row)
        sig_layout.addWidget(self.signals, 1)

        self.cursors = []
        self.mark_regions = []
        for plot in (self.timeline, self.signals):
            cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((255, 220, 0), width=1.5))
            cursor.sigDragged.connect(lambda line: self.seek(self._row_at_time(line.value())))
            plot.addItem(cursor)
            self.cursors.append(cursor)
            region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 255, 255, 25))
            region.setZValue(-10)
            region.hide()
            plot.addItem(region)
            self.mark_regions.append(region)
            plot.scene().sigMouseClicked.connect(lambda ev, p=plot: self._plot_clicked(p, ev))
        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.addTab(self.timeline, 'Coverage timeline')
        self.plot_tabs.addTab(sig_widget, 'Colour signals')
        self.plots_dock = dock = QtWidgets.QDockWidget('Plots', self, objectName='plots_dock')
        dock.setWidget(self.plot_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_shortcuts(self):
        keys = {
            'Space': self.toggle_play, 'Right': lambda: self.step(1), 'Left': lambda: self.step(-1),
            'Shift+Right': lambda: self.step(10), 'Shift+Left': lambda: self.step(-10),
            'Home': lambda: self.seek(self._range()[0]), 'End': lambda: self.seek(self._range()[1]),
            'I': self.set_in, 'O': self.set_out, 'Ctrl+I': self.clear_marks,
            'PgDown': lambda: self.browser.step(1), 'PgUp': lambda: self.browser.step(-1),
            'F': self.fit_all, 'Ctrl+S': self.snapshot, 'Ctrl+E': self.export_video,
        }
        for key, slot in keys.items():
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)

    # ------------------------------------------------------------------ panels
    def _rebuild_grid(self):
        rows, cols = LAYOUTS.get(self.layout_combo.currentText(), (2, 3))
        n = rows * cols
        while len(self.panel_keys) < n:
            self.panel_keys.append(next((k for k in DEFAULT_PANELS if k not in self.panel_keys), 'uv_skin_rgb'))
        while len(self.panels) < n:
            panel = ImagePanel(self.panel_keys[len(self.panels)])
            panel.viewChanged.connect(self._panel_view_changed)
            self.panels.append(panel)
        for panel in self.panels[n:]:
            self.grid.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self.panels = self.panels[:n]
        for i, panel in enumerate(self.panels):
            self.grid.addWidget(panel, i // cols, i % cols)
        for r in range(4):
            self.grid.setRowStretch(r, 1 if r < rows else 0)
        for c in range(4):
            self.grid.setColumnStretch(c, 1 if c < cols else 0)
        self._apply_links()
        self._push_config()

    def _panel_view_changed(self):
        for i, panel in enumerate(self.panels):
            self.panel_keys[i] = panel.key
        self._apply_links()
        self._push_config()

    def _apply_links(self):
        masters = {}
        for panel in self.panels:
            if not self.link_check.isChecked() or panel.space in ('3d',):
                panel.vb.setXLink(None)
                panel.vb.setYLink(None)
                continue
            master = masters.setdefault(panel.space, panel)
            target = None if master is panel else master.vb
            panel.vb.setXLink(target)
            panel.vb.setYLink(target)

    def fit_all(self):
        for panel in self.panels:
            panel.fit()

    def _opts(self):
        return {'alpha': self.alpha_slider.value() / 100.0, 'silhouette': self.silhouette_check.isChecked(),
                'age_window_s': self.age_spin.value()}

    def _push_config(self):
        keys = [p.key for p in self.panels]
        if any('mesh' in VIEW_BY_KEY[k].items for k in keys) and self.mesh.state == 'off':
            self.mesh.load()
        if self.session is not None:
            self.session.set_config(keys, self._opts())
            self._request()

    def _mesh_state_changed(self):
        state = self.mesh.state
        self.mesh_label.setText({'off': ' mesh: not loaded', 'loading': ' mesh: loading...',
                                 'ready': ' mesh: ready', 'error': ' mesh: error',
                                 'disabled': ' mesh: disabled'}[state])
        self.mesh_label.setToolTip(self.mesh.error)
        self.mesh_action.setEnabled(state == 'off')
        self.mesh_load_action.setEnabled(state == 'off')
        self.mesh_enable_action.blockSignals(True)
        self.mesh_enable_action.setChecked(self.mesh.enabled)
        self.mesh_enable_action.blockSignals(False)
        if self.session is not None:
            self.session.invalidate()
            self._request()

    def _set_mesh_enabled(self, enabled):
        self.settings.setValue('mesh_enabled', enabled)
        self.mesh.configure(self.mesh.assets_dir, enabled)
        self._push_config()

    def converter_config(self):
        """Model / tool locations for the dataset conversion."""
        s = self.settings
        return {'assets_dir': str(self.assets_dir or ''), 'ffmpeg_dir': s.value('ffmpeg_dir', '') or '',
                'checkpoint': s.value('pear_checkpoint', '') or '', 'yolo_weights': s.value('yolo_weights', '') or ''}

    def open_settings(self):
        cfg = self.converter_config()
        dialog = SettingsDialog(self, {**cfg, 'mesh_enabled': self.mesh.enabled, 'load_at_start': self.load_mesh_at_start,
                                       'video_scale': self.video_scale})
        if not dialog.exec():
            return
        v = dialog.values()
        self.assets_dir = v['assets_dir']
        self.settings.setValue('assets_dir', str(v['assets_dir']) if v['assets_dir'] else '')
        self.settings.setValue('ffmpeg_dir', v['ffmpeg_dir'])
        self.settings.setValue('pear_checkpoint', v['checkpoint'])
        self.settings.setValue('yolo_weights', v['yolo_weights'])
        self.settings.setValue('mesh_enabled', v['mesh_enabled'])
        self.settings.setValue('load_mesh_at_start', v['load_at_start'])
        self.load_mesh_at_start = v['load_at_start']
        if self.mesh.configure(v['assets_dir'], v['mesh_enabled']):
            QtWidgets.QMessageBox.information(
                self, 'Restart needed', 'The SMPL-X model of the previous assets folder is already loaded. '
                                        'Restart vHRM Studio to use the new folder in the mesh views.')
        self._push_config()
        if v['load_at_start']:
            self.mesh.load()
        if v['video_scale'] != self.video_scale:
            self.scale_combo.setCurrentIndex(self.scale_combo.findData(v['video_scale']))
        if self.converter is not None:
            self.converter.refresh_models()

    # ------------------------------------------------------------------ dataset conversion
    def open_converter(self):
        if self.converter is None:
            from .convert_window import ConvertWindow
            self.converter = ConvertWindow(self.settings, self.converter_config)
            self.converter.openFolder.connect(self._show_output_folder)
            self.converter.openFile.connect(self._show_output_file)
            self.converter.settingsRequested.connect(self.open_settings)
        self.converter.refresh_models()
        self.converter.show()
        self.converter.raise_()
        self.converter.activateWindow()

    def _show_output_folder(self, folder):
        if folder and Path(folder).is_dir():
            self.files_dock.show()
            self.browser.set_root(folder)
            self.raise_()
            self.activateWindow()

    def _show_output_file(self, path):
        from .dataset import find_manifest
        _, root = find_manifest(path)
        current = self.browser.root_edit.text().strip()
        if not current or not Path(path).resolve().is_relative_to(Path(current).resolve()):
            self.browser.set_root(str(root or Path(path).parent))
        self.open_file(str(Path(path).resolve()))
        self.raise_()
        self.activateWindow()

    def _open_dataset_root(self):
        if self.session is None:
            return
        from .dataset import find_manifest
        _, root = find_manifest(self.session.src.path)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root or self.session.src.path.parent)))

    # ------------------------------------------------------------------ files
    def _open_folder(self):
        self.files_dock.show()
        self.browser.browse()

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        for action in self.browser.recent_menu.actions():
            self.recent_menu.addAction(action)

    def _root_changed(self, root):
        self.settings.setValue('root', root)
        self.settings.setValue('recent_roots', json.dumps(self.browser.recent))
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open skin-UV HDF5', self.browser.root_edit.text(),
                                                        'HDF5 (*.h5 *.hdf5)')
        if path:
            self.open_file(str(Path(path).resolve()))

    def open_file(self, path, row=0):
        self.stop()
        if self.session is not None:
            self.session.close()
            self.session = None
        try:
            session = Session(path, self.mesh, self.video_scale, self.video_overrides.get(path))
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, 'Cannot open file', f'{path}\n\n{type(exc).__name__}: {exc}')
            return
        self.session = session
        session.composed.connect(self._composed)
        src = session.src
        self.browser.mark_current(path)
        self.setWindowTitle(f'vHRM Studio - {src.path.name}')
        self.mark_in = self.mark_out = None
        self._update_marks()
        for w in (self.slider, self.row_spin):
            w.blockSignals(True)
            w.setRange(0, src.n_rows - 1)
            w.blockSignals(False)
        self.shown_row = None
        self._stats_seen = (-1, -1)
        for curve in [*self.tl_curves.values(), *self.sig_curves]:
            curve.setData([], [])
        self.tl_bad.setData([], [])
        self.sig_button.setEnabled(True)
        self.sig_status.setText('')
        for panel in self.panels:
            panel.clear_mesh()
        self._fill_file_info()
        self.fps_label.setText(f'{src.fps:.3f} fps, {src.n_rows} rows'
                               + ('' if session.video else f' | {session.video_error}'))
        self.row = -1
        session.set_config([p.key for p in self.panels], self._opts())
        self.seek(min(max(row, 0), src.n_rows - 1))

    def locate_video(self):
        if self.session is None:
            return
        start = str(self.session.video_path or Path(str(self.session.src.attrs.get('source_video_path', ''))).parent)
        if self.session.src.attrs.get('source_type') == 'images':
            path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Source image folder', start)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Source video', start,
                                                            'Video (*.mp4 *.mov *.mkv *.avi);;All files (*)')
        if path:
            h5 = str(self.session.src.path.resolve())
            self.video_overrides[h5] = path
            self.settings.setValue('video_overrides', json.dumps(self.video_overrides))
            self.open_file(h5, self.row)

    def _video_scale_changed(self):
        self.video_scale = float(self.scale_combo.currentData())
        self.settings.setValue('video_scale', self.video_scale)
        if self.session is not None:
            self.open_file(str(self.session.src.path.resolve()), self.row)

    # ------------------------------------------------------------------ navigation
    def _range(self):
        if self.session is None:
            return 0, 0
        lo = self.mark_in if self.mark_in is not None else 0
        hi = self.mark_out if self.mark_out is not None else self.session.src.n_rows - 1
        return (lo, hi) if lo <= hi else (hi, lo)

    def _advance(self, row, k):
        lo, hi = self._range()
        r = row + k
        if r <= hi:
            return r
        if not self.loop_check.isChecked():
            return None
        return lo + (r - hi - 1) % (hi - lo + 1)

    def step(self, delta):
        if self.session is not None:
            self.stop()
            self.seek(self.row + delta)

    def seek(self, row):
        if self.session is None:
            return
        row = int(min(max(row, 0), self.session.src.n_rows - 1))
        self.row = row
        if self.playing:
            self._restart_clock()
        self._request()
        self._update_position_widgets(row)
        if self.session.has_result(row):
            self._display(row)

    def _request(self, extra=()):
        if self.session is None:
            return
        row = self.row
        if self.playing or self.exporting:
            ahead = [self._advance(row, j) for j in range(1, PREFETCH_AHEAD + 1)]
            self.session.set_wanted([row, *extra, *ahead])
        else:
            self.session.set_wanted([row, row + 1, row - 1, row + 2])

    def _composed(self, row, version):
        if self.session is None or version != self.session.version:
            return
        if row == self.row and not self.playing:
            self._display(row)

    def _display(self, row):
        res = self.session.result(row) if self.session is not None else None
        if res is None:
            return False
        for panel in self.panels:
            panel.show_result(res.get(panel.key))
        self.shown_row = row
        self._update_position_widgets(row)
        self._update_frame_info()
        return True

    def _update_position_widgets(self, row):
        src = self.session.src
        for w in (self.slider, self.row_spin):
            w.blockSignals(True)
            w.setValue(row)
            w.blockSignals(False)
        t = src.timestamps[row] - src.timestamps[0]
        self.time_label.setText(f'row {row} / {src.n_rows - 1}   t = {t:8.3f} s   source frame '
                                f'{int(src.frame_index[row])}   status: {src.status_text(row)}')
        for cursor in self.cursors:
            cursor.blockSignals(True)
            cursor.setValue(t)
            cursor.blockSignals(False)

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.playing:
            self.stop()
        elif self.session is not None:
            if self._advance(self.row, 1) is None:
                self.seek(self._range()[0])
            self.playing = True
            self.play_button.setText('Pause')
            self._restart_clock()
            self._request()
            self.play_timer.start()

    def stop(self):
        if not self.playing:
            return
        self.playing = False
        self.play_button.setText('Play')
        self.play_timer.stop()
        self._request()
        if self.shown_row != self.row and self.session is not None and self.session.has_result(self.row):
            self._display(self.row)
        self._update_frame_info()

    def _restart_clock(self):
        self.play_t0 = time.perf_counter()
        self.play_steps = 0
        self._fps_t0, self._fps_n = self.play_t0, 0

    def _tick(self):
        if not self.playing or self.session is None:
            return
        now = time.perf_counter()
        period = 1.0 / (self.session.src.fps * float(self.speed_combo.currentData()))
        need = int((now - self.play_t0) / period) - self.play_steps
        if need <= 0:
            return
        if self._advance(self.row, 1) is None:
            self.stop()
            return
        if self.realtime_check.isChecked():
            found = None
            for j in range(need, 0, -1):
                r = self._advance(self.row, j)
                if r is not None and self.session.has_result(r):
                    found = (j, r)
                    break
            if found is None:
                if need > 1:  # behind: drop the rows that are already late
                    target = self._advance(self.row, need)
                    if target is not None:
                        self.session.set_wanted([target] + [self._advance(target, j)
                                                            for j in range(1, PREFETCH_AHEAD)])
                return
            j, r = found
            self.play_steps += j
        else:
            r = self._advance(self.row, 1)
            if not self.session.has_result(r):
                self.play_t0 = now - (self.play_steps + 1) * period
                return
            self.play_steps += 1
            if need > 1:
                self.play_t0 = now - self.play_steps * period
        self.row = r
        self._display(r)
        self._request()
        self._fps_n += 1
        if now - self._fps_t0 > 1.0:
            self.fps_label.setText(f'playing {self._fps_n / (now - self._fps_t0):.1f} rows/s '
                                   f'(target {self.session.src.fps * float(self.speed_combo.currentData()):.1f})')
            self._fps_t0, self._fps_n = now, 0

    # ------------------------------------------------------------------ marks
    def set_in(self):
        self.mark_in = self.row
        self._update_marks()

    def set_out(self):
        self.mark_out = self.row
        self._update_marks()

    def clear_marks(self):
        self.mark_in = self.mark_out = None
        self._update_marks()

    def _update_marks(self):
        has = self.mark_in is not None or self.mark_out is not None
        self.marks_label.setText(f'in {self.mark_in if self.mark_in is not None else "-"} / '
                                 f'out {self.mark_out if self.mark_out is not None else "-"}' if has else '')
        for region in self.mark_regions:
            if has and self.session is not None:
                lo, hi = self._range()
                ts = self.session.src.timestamps
                region.setRegion((ts[lo] - ts[0], ts[hi] - ts[0]))
                region.show()
            else:
                region.hide()

    # ------------------------------------------------------------------ plots
    def _row_at_time(self, t):
        ts = self.session.src.timestamps
        return int(np.argmin(np.abs((ts - ts[0]) - t)))

    def _plot_clicked(self, plot, ev):
        if self.session is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.stop()
        self.seek(self._row_at_time(vb.mapSceneToView(ev.scenePos()).x()))

    def _start_colour(self):
        if self.session is not None:
            self.session.stats.want_colour.set()
            self.sig_button.setEnabled(False)

    def _refresh_stats(self, force=False):
        if self.session is None:
            return
        stats, src = self.session.stats, self.session.src
        done = tuple(stats.done)
        if done == self._stats_seen and not force:
            return
        self._stats_seen = done
        t = src.timestamps - src.timestamps[0]
        crop_px = src.crop_size ** 2
        for key, curve in self.tl_curves.items():
            denom = crop_px if key == 'crop_skin' else stats.body_texels
            curve.setData(t, 100.0 * stats.counts[key] / denom, connect='finite')
        bad = np.flatnonzero(src.status != 1)
        self.tl_bad.setData(t[bad], np.zeros(len(bad)))
        j = self.region_combo.currentIndex()
        mode = self.sig_mode.currentIndex()
        if mode == 2:
            for c, curve in enumerate(self.sig_curves):
                curve.setData(t, stats.colour_texels[:, j] if c == 1 else np.full_like(t, np.nan), connect='finite')
            self.signals.setLabel('left', 'skin texels')
        else:
            values = stats.colour[:, j, :]
            if mode == 1:
                with np.errstate(invalid='ignore'):
                    values = values / np.nanmean(values, axis=0) - 1.0 if np.isfinite(values).any() else values
            for c, curve in enumerate(self.sig_curves):
                curve.setData(t, values[:, c], connect='finite')
            self.signals.setLabel('left', 'relative' if mode == 1 else 'mean level (0-255)')
        if stats.error is not None:
            self.sig_status.setText(f'error: {stats.error}')
        elif stats.want_colour.is_set():
            self.sig_status.setText(f'{done[1]} / {src.n_rows} rows' if done[1] < src.n_rows else 'done')
        else:
            self.sig_status.setText(f'coverage {done[0]} / {src.n_rows} rows' if done[0] < src.n_rows else '')

    # ------------------------------------------------------------------ details
    def _fill_file_info(self):
        tree, src, session = self.file_tree, self.session.src, self.session
        tree.clear()
        top = QtWidgets.QTreeWidgetItem(['File', str(src.path)])
        tree.addTopLevelItem(top)
        video = QtWidgets.QTreeWidgetItem(['Source video', str(session.video_path) if session.video_path
                                           else session.video_error])
        tree.addTopLevelItem(video)
        attrs = QtWidgets.QTreeWidgetItem(['Attributes', f'{len(src.attrs)}'])
        for k in sorted(src.attrs):
            child = QtWidgets.QTreeWidgetItem([k, format_value(src.attrs[k])])
            child.setToolTip(1, format_value(src.attrs[k]))
            attrs.addChild(child)
        tree.addTopLevelItem(attrs)
        datasets = QtWidgets.QTreeWidgetItem(['Datasets', f'{len(src.dataset_info)}'])
        for name, text in src.dataset_info:
            datasets.addChild(QtWidgets.QTreeWidgetItem([name, text]))
        tree.addTopLevelItem(datasets)
        attrs.setExpanded(True)
        datasets.setExpanded(True)
        tree.resizeColumnToContents(0)

    def _update_frame_info(self):
        if (self.session is None or self.playing or not self.info_dock.isVisible()
                or self.info_tabs.currentWidget() is not self.frame_tree or self.shown_row is None):
            return
        src, row = self.session.src, self.shown_row
        tree = self.frame_tree
        tree.setUpdatesEnabled(False)
        expanded = {tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
                    if tree.topLevelItem(i).isExpanded()} or {'row', 'geometry'}
        tree.clear()
        general = QtWidgets.QTreeWidgetItem(['row', str(row)])
        for k, v in (('frame_status', f'{int(src.status[row])} ({src.status_text(row)})'),
                     ('source_frame_index', int(src.frame_index[row])),
                     ('source_timestamp_s', f'{src.timestamps[row]:.6f}')):
            general.addChild(QtWidgets.QTreeWidgetItem([k, str(v)]))
        tree.addTopLevelItem(general)
        geometry = QtWidgets.QTreeWidgetItem(['geometry', ''])
        for k, v in src.small.items():
            geometry.addChild(QtWidgets.QTreeWidgetItem([k, format_value(v[row])]))
        tree.addTopLevelItem(geometry)
        for group, values in src.row_params(row).items():
            item = QtWidgets.QTreeWidgetItem([group, f'{len(values)} arrays'])
            for k, v in values.items():
                child = QtWidgets.QTreeWidgetItem([k, f'{v.shape}  {format_value(v)}'])
                child.setToolTip(1, format_value(v))
                item.addChild(child)
            tree.addTopLevelItem(item)
        for i in range(tree.topLevelItemCount()):
            tree.topLevelItem(i).setExpanded(tree.topLevelItem(i).text(0) in expanded)
        tree.resizeColumnToContents(0)
        tree.setUpdatesEnabled(True)

    # ------------------------------------------------------------------ export
    def snapshot(self):
        if self.session is None:
            return
        name = f'{self.session.src.path.stem}_row{self.row:05d}.png'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save snapshot', str(self.session.src.path.parent / name),
                                                        'PNG (*.png)')
        if path:
            self.grid_widget.grab().save(path)

    def export_video(self):
        if self.session is None or self.exporting:
            return
        self.stop()
        src = self.session.src
        has_marks = self.mark_in is not None or self.mark_out is not None
        dialog = ExportDialog(self, str(src.path.parent / f'{src.path.stem}_view.mp4'), src.fps, has_marks)
        if not dialog.exec():
            return
        mode = dialog.range_combo.currentData()
        lo, hi = {'all': (0, src.n_rows - 1), 'marks': self._range(),
                  'current': (self.row, src.n_rows - 1)}[mode]
        rows = list(range(lo, hi + 1))
        progress = QtWidgets.QProgressDialog('Exporting...', 'Cancel', 0, len(rows), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        writer = None
        self.exporting = True
        try:
            for i, row in enumerate(rows):
                self.row = row
                self._request()
                while not self.session.has_result(row):
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
                    if progress.wasCanceled():
                        return
                self._display(row)
                QtWidgets.QApplication.processEvents()
                rgb = qimage_to_rgb(self.grid_widget.grab().toImage())
                if writer is None:
                    h, w = rgb.shape[:2]
                    writer = Mp4Writer(dialog.path_edit.text(), dialog.fps.value(), w - w % 2, h - h % 2)
                writer.write(rgb[:writer.size[1], :writer.size[0]])
                progress.setValue(i + 1)
                if progress.wasCanceled():
                    return
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, 'Export failed', f'{type(exc).__name__}: {exc}')
        finally:
            self.exporting = False
            progress.close()
            if writer is not None:
                writer.close()
                self.statusBar().showMessage(f'Exported {dialog.path_edit.text()} ({writer.codec})', 10000)

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event):
        if self.converter is not None:
            if self.converter.is_running():
                answer = QtWidgets.QMessageBox.question(
                    self, 'Conversion running', 'A dataset conversion is running. Quit and cancel it? '
                                                '(Finished files are kept; Start resumes later.)')
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            self.converter.shutdown()
        self.stop()
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        self.settings.setValue('layout', self.layout_combo.currentText())
        self.settings.setValue('panels', json.dumps(self.panel_keys))
        self.settings.setValue('recent_roots', json.dumps(self.browser.recent))
        if self.browser.worker is not None:
            self.browser.worker.requestInterruption()
            self.browser.worker.wait()
        if self.session is not None:
            self.session.close()
            self.session = None
        super().closeEvent(event)


def main(argv=None):
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:  # console-less launcher (pythonw / gui-scripts)
            setattr(sys, name, open(os.devnull, 'w'))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Every option can also be set inside the application (File > Open folder, File > Settings); '
               'options given here override the saved settings for this session only.')
    parser.add_argument('paths', nargs='*', help='HDF5 files and/or folders (default: the last folder used)')
    parser.add_argument('--assets', type=Path, default=None,
                        help='SMPL-X / FLAME assets folder (holds SMPLX/ and FLAME/; default: the saved setting, '
                             '$VHRM_ASSETS_DIR, then searched upwards from the working directory and this package)')
    parser.add_argument('--convert', action='store_true', help='open the dataset conversion window at start-up')
    parser.add_argument('--no-mesh', action='store_true', help='disable the SMPL-X mesh views')
    parser.add_argument('--load-mesh', action='store_true', help='load the SMPL-X model at start-up')
    parser.add_argument('--video-scale', type=float, default=None, choices=VIDEO_SCALES,
                        help='source video decode scale (default: the saved setting, 0.5)')
    args = parser.parse_args(argv)
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('vHRM Studio')
    window = MainWindow(args)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
