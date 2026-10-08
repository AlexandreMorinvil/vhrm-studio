"""rPPG dataset folders: discovery of the sources, the conversion plan (where each HDF5 file goes) and the manifest.

A source is a video file, or a folder of images holding one frame each (PURE, ...). Outputs go either next to their
source ('in_place') or into a new root that mirrors the source tree ('mirror'), where the other files of the dataset
(ground-truth signals, metadata) can be copied too. vhrm_manifest.json (and a CSV copy) in the output root records
which source every HDF5 file comes from; the viewer uses it to find the source videos.

No torch / Qt imports here: the GUI and the CLI use this module directly.
"""
import csv
import datetime
import fnmatch
import json
import os
import shutil
import subprocess
from pathlib import Path

from .pipeline.frames import IMAGE_EXTENSIONS, natural_key

VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.m4v', '.wmv', '.mpg', '.mpeg', '.webm', '.flv', '.mts',
                    '.m2ts', '.mxf', '.y4m')
MIN_SEQUENCE_IMAGES = 30
H5_SUFFIX = '_skin_uv.h5'
PARTIAL_SUFFIX = '.partial'
MANIFEST_NAME = 'vhrm_manifest.json'
MANIFEST_CSV = 'vhrm_manifest.csv'
MANIFEST_FORMAT = 'vhrm-studio-manifest'
OWN_PATTERNS = (f'*{H5_SUFFIX}', f'*{PARTIAL_SUFFIX}', MANIFEST_NAME, MANIFEST_CSV, '*_accumulated_skin_uv.png',
                'vhrm_conversion_log.txt', 'vhrm_conversion.lock')
MODES = ('in_place', 'mirror')


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def rel_posix(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def _is_own_file(name):
    return any(fnmatch.fnmatch(name, pattern) for pattern in OWN_PATTERNS)


def _inside(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- discovery
class SourceItem:
    """One source of a dataset. kind: 'video' (path = file) or 'images' (path = folder, files = frame images)."""

    def __init__(self, kind, path, rel, files=None):
        self.kind, self.path, self.rel = kind, Path(path), rel
        self.files = files
        self.output = None          # absolute HDF5 path (plan)
        self.output_rel = None      # relative to the output root
        self.frames = None          # source frames (probe)
        self.fps = None
        self.duration = None
        self.size = None            # (width, height)
        self.probe_error = None

    @property
    def stem(self):
        return self.path.stem if self.kind == 'video' else self.path.name

    def estimated_rows(self, target_fps=30.0):
        """Rows the pipeline will write (~target_fps), for progress estimates."""
        if not self.frames:
            return 0
        if self.fps and self.fps > target_fps * 1.01:
            return int(round(self.frames * target_fps / self.fps))
        return int(self.frames)

    def to_job(self):
        return {'kind': self.kind, 'path': str(self.path), 'rel': self.rel, 'output': str(self.output),
                'output_rel': self.output_rel, 'estimated_frames': self.frames, 'fps': self.fps}


def discover(root, video_extensions=VIDEO_EXTENSIONS, min_images=MIN_SEQUENCE_IMAGES, exclude=None,
             should_stop=None):
    """Sources under root (recursively), in natural path order. Folders whose name starts with '.' and the
    `exclude` folder (an output root inside the source tree) are skipped."""
    root = Path(root).resolve()
    video_extensions = tuple(e.lower() for e in video_extensions)
    items = []
    for folder, dirs, files in os.walk(root):
        if should_stop is not None and should_stop():
            break
        dirs[:] = sorted((d for d in dirs if not d.startswith('.')
                          and not (exclude and _inside(Path(folder) / d, exclude))), key=natural_key)
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS) and not _is_own_file(f)]
        if len(images) >= min_images:
            items.append(SourceItem('images', folder, rel_posix(folder, root), sorted(images, key=natural_key)))
        for name in sorted(files, key=natural_key):
            if name.lower().endswith(video_extensions) and not _is_own_file(name):
                path = Path(folder) / name
                items.append(SourceItem('video', path, rel_posix(path, root)))
    items.sort(key=lambda item: natural_key(item.rel))
    return items


def probe_item(item, ffmpeg_dir=None):
    """Fills frames / fps / duration / size (fast: stream headers only for videos, the first image otherwise)."""
    from .pipeline.frames import find_tool, image_timestamps, read_image_rgb
    try:
        if item.kind == 'images':
            item.frames = len(item.files)
            ts, _ = image_timestamps(item.files)
            item.duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            item.fps = (len(ts) - 1) / item.duration if item.duration > 0 else None
            first = read_image_rgb(item.path / item.files[0])
            item.size = (first.shape[1], first.shape[0])
        else:
            cmd = [find_tool('ffprobe', ffmpeg_dir), '-v', 'error', '-select_streams', 'v:0', '-of', 'json',
                   '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration',
                   str(item.path)]
            data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout)
            stream = (data.get('streams') or [{}])[0]
            if 'width' not in stream:
                raise RuntimeError('no video stream')

            def rate(text):
                num, _, den = (text or '0/1').partition('/')
                return float(num) / float(den) if float(den or 0) else 0.0

            item.fps = rate(stream.get('avg_frame_rate')) or rate(stream.get('r_frame_rate')) or None
            item.duration = float(stream.get('duration') or data.get('format', {}).get('duration') or 0.0) or None
            item.frames = int(stream.get('nb_frames') or 0) or (
                int(round(item.duration * item.fps)) if item.duration and item.fps else None)
            item.size = (int(stream['width']), int(stream['height']))
    except Exception as exc:
        item.probe_error = f'{type(exc).__name__}: {exc}'
    return item


# ---------------------------------------------------------------- plan
def plan_outputs(items, source_root, mode, output_root=None, suffix=H5_SUFFIX):
    """Sets item.output / item.output_rel. Returns the output root (the source root in place)."""
    source_root = Path(source_root).resolve()
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if mode == 'mirror':
        if not output_root:
            raise ValueError('mirror mode needs an output root')
        output_root = Path(output_root).resolve()
        if output_root == source_root:
            raise ValueError('the output root must differ from the source root (or use "next to the sources")')
    else:
        output_root = source_root
    used = {}
    for item in items:
        # next to the video; for an image folder, next to the folder (inside it when it is the root itself)
        parent = item.path.parent if item.kind == 'video' or item.path != source_root else item.path
        rel_parent = Path(os.path.relpath(parent, source_root))
        name = f'{item.stem}{suffix}'
        key = (rel_parent / name).as_posix().lower()
        if key in used:  # a.avi and a.mp4 (or a video and an image folder) in one folder
            name = f'{item.stem}_{item.path.suffix.lstrip(".") or "images"}{suffix}'
            key = (rel_parent / name).as_posix().lower()
        used[key] = item
        item.output_rel = (rel_parent / name).as_posix().removeprefix('./')
        item.output = output_root / item.output_rel
    return output_root


def companion_files(source_root, items, exclude=None):
    """Files of the dataset that are neither sources nor vHRM Studio outputs (signals, metadata, ...)."""
    source_root = Path(source_root).resolve()
    sources = {item.path.resolve() for item in items if item.kind == 'video'}
    image_dirs = {item.path.resolve() for item in items if item.kind == 'images'}
    out = []
    for folder, dirs, files in os.walk(source_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (exclude and _inside(Path(folder) / d, exclude)))
        folder = Path(folder)
        in_sequence = folder.resolve() in image_dirs
        for name in sorted(files):
            path = folder / name
            if _is_own_file(name) or path.resolve() in sources:
                continue
            if in_sequence and name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            out.append(path)
    return out


def output_complete(path):
    """True when path is a finished HDF5 file of the pipeline."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return 'frame_status' in f and 'saved_frame_count' in f.attrs and bool(f.attrs.get('complete', True))
    except Exception:
        return False


def copy_if_newer(src, dst):
    """Copies src to dst unless dst has the same size and modification time. Returns True when copied."""
    src, dst = Path(src), Path(dst)
    try:
        s, d = src.stat(), dst.stat()
        if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
            return False
    except FileNotFoundError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------- manifest
def manifest_path(output_root):
    return Path(output_root) / MANIFEST_NAME


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if data.get('format') == MANIFEST_FORMAT else None


def new_manifest(source_root, output_root, mode, settings=None):
    return {'format': MANIFEST_FORMAT, 'version': 1, 'created': now_iso(), 'updated': now_iso(),
            'mode': mode, 'source_root': str(Path(source_root).resolve()),
            'output_root': str(Path(output_root).resolve()), 'settings': settings or {}, 'items': []}


def manifest_item(manifest, source_rel):
    for entry in manifest['items']:
        if entry.get('source') == source_rel:
            return entry
    entry = {'source': source_rel}
    manifest['items'].append(entry)
    return entry


CSV_COLUMNS = ('source', 'output', 'kind', 'status', 'rows', 'ok_rows', 'effective_fps', 'frame_selection_mode',
               'seconds', 'finished', 'source_copied', 'error')


def save_manifest(manifest):
    """Writes vhrm_manifest.json and vhrm_manifest.csv into the manifest's output root (atomically)."""
    root = Path(manifest['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    manifest['updated'] = now_iso()
    manifest['items'].sort(key=lambda entry: natural_key(entry.get('source', '')))
    target = root / MANIFEST_NAME
    tmp = target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, target)
    tmp = root / (MANIFEST_CSV + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for entry in manifest['items']:
            writer.writerow({k: entry.get(k, '') for k in CSV_COLUMNS})
    os.replace(tmp, root / MANIFEST_CSV)


_MANIFEST_CACHE = {}


def find_manifest(h5_path):
    """(manifest, its folder) of the nearest vhrm_manifest.json above h5_path, or (None, None)."""
    for parent in Path(h5_path).resolve().parents:
        path = parent / MANIFEST_NAME
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cached = _MANIFEST_CACHE.get(path)
        if cached is None or cached[0] != mtime:
            cached = _MANIFEST_CACHE[path] = (mtime, load_manifest(path))
        if cached[1] is not None:
            return cached[1], parent
    return None, None


def source_from_manifest(h5_path):
    """Source path recorded for h5_path in its dataset manifest (the original, else a copy in the output tree)."""
    manifest, folder = find_manifest(h5_path)
    if manifest is None:
        return None
    rel = rel_posix(Path(h5_path).resolve(), folder)
    for entry in manifest['items']:
        if entry.get('output') == rel:
            candidates = [Path(manifest['source_root']) / entry['source'], folder / entry['source']]
            return next((c for c in candidates if c.exists()), None)
    return None
