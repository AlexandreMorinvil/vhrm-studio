"""Model files that cannot ship with vHRM Studio, and where they are found.

* SMPL-X / FLAME body-model assets (license of the Max Planck Institute, registration required): a folder holding
  `SMPLX/` and `FLAME/`, laid out like PEAR's `assets` folder. Needed by the dataset conversion and the mesh views.
* PEAR checkpoint `ehm_model_stage1.pt`: a configured file, `<assets>/../weights/`, else downloaded from Hugging Face.
* YOLOv8x person detector: a configured file, else downloaded by ultralytics into the vHRM Studio cache folder.

The DeepLab ONNX model and the packed-UV layout are package data (vhrm_studio/pipeline/data).
"""
import os
import sys
from pathlib import Path

ENV_ASSETS = 'VHRM_ASSETS_DIR'
SMPLX_FILES = (
    'SMPLX_NEUTRAL_2020.npz', 'flame_generic_model.pkl', 'smplx_tex.obj', 'smplx_uv.obj', 'smplx_extra_joints.yaml',
    'SMPLX_to_J14.pkl', 'SMPL-X__FLAME_vertex_ids.npy', 'flame_l_eyelid.npy', 'flame_r_eyelid.npy',
    'mediapipe_landmark_embedding.npz', 'MANO_SMPLX_vertex_ids.pkl', '203_landmark_embeding.npz',
    'uv_masks/uv_mask512_with_faceid_smplx.npy', 'smplx_faces.npy', 'lbs_map_smplx_512.npy',
)
FLAME_FILES = (
    'FLAME2020/generic_model.pkl', 'FLAME_masks/FLAME_masks.pkl', 'l_eyelid.npy', 'r_eyelid.npy',
    'landmark_embedding.npy', 'mediapipe_landmark_embedding.npz', '203_landmark_embeding.npz',
    'selected_lowerhead.npy', 'head_template.obj',
)
REQUIRED_FILES = tuple(f'SMPLX/{f}' for f in SMPLX_FILES) + tuple(f'FLAME/{f}' for f in FLAME_FILES)
LICENSED_FILES = ('SMPLX/SMPLX_NEUTRAL_2020.npz', 'SMPLX/flame_generic_model.pkl', 'FLAME/FLAME2020/generic_model.pkl')
ASSETS_HELP = (
    'The body-model folder holds SMPLX/ and FLAME/ as in PEAR\'s "assets" folder: download the bundle linked in the '
    'PEAR README (https://github.com/Pixel-Talk/PEAR), or get SMPLX_NEUTRAL_2020.npz from smpl-x.is.tue.mpg.de and '
    'generic_model.pkl (FLAME 2020) from flame.is.tue.mpg.de and add the other files from the PEAR repository.')


def missing_assets(assets_dir):
    """Required files missing from assets_dir (all of them when it is not a folder)."""
    if not assets_dir or not Path(assets_dir).is_dir():
        return list(REQUIRED_FILES)
    root = Path(assets_dir)
    return [f for f in REQUIRED_FILES if not (root / f).is_file()]


def is_assets_dir(path):
    return bool(path) and not missing_assets(path)


def find_assets_dir(*starts):
    """$VHRM_ASSETS_DIR, else an `assets` or `PEAR/assets` folder in or above the starts (default: the working
    directory and this package)."""
    env = os.environ.get(ENV_ASSETS)
    if env and is_assets_dir(env):
        return Path(env).resolve()
    for start in starts or (Path.cwd(), Path(__file__)):
        start = Path(start).resolve()
        for parent in [start, *start.parents]:
            for candidate in (parent, parent / 'assets', parent / 'PEAR' / 'assets'):
                if (candidate / 'SMPLX').is_dir() and is_assets_dir(candidate):
                    return candidate
    return None


def cache_dir():
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'vHRM Studio'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'vhrm_studio'
    path = base / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(configured=None, assets_dir=None, download=True):
    """PEAR checkpoint: the configured file, <assets>/../weights/ehm_model_stage1.pt, else the Hugging Face copy
    (downloaded once into the Hugging Face cache when download=True)."""
    from .third_party.pear import CHECKPOINT_FILE, CHECKPOINT_REPO
    if configured:
        if not Path(configured).is_file():
            raise FileNotFoundError(f'PEAR checkpoint not found: {configured}')
        return Path(configured)
    if assets_dir:
        local = Path(assets_dir).parent / 'weights' / CHECKPOINT_FILE
        if local.is_file():
            return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE, repo_type='model',
                                local_files_only=not download))


def yolo_path(configured=None):
    """YOLOv8x weights: the configured file, else <cache>/yolov8x.pt (ultralytics downloads it there if missing)."""
    if configured:
        if not Path(configured).is_file():
            raise FileNotFoundError(f'YOLO weights not found: {configured}')
        return Path(configured)
    return cache_dir() / 'yolov8x.pt'
