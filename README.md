# vHRM Studio

vHRM (Visual Heart Rate Monitoring) Studio prepares and inspects remote-photoplethysmography (rPPG) datasets:

- **Convert** a dataset folder (videos, or folders of images such as PURE) into one skin-UV HDF5 file per source:
  every frame's visible skin mapped onto the SMPL-X UV atlas (PEAR body/face fit + DeepLabV3+ human parsing with
  a temporal skin-colour refinement), plus the crops, masks, class maps and body parameters.
- **View** the HDF5 files: step through or play the rows of a file and show any mix of the stored UV / crop
  products, the source video or images, coverage and colour plots, and the SMPL-X mesh rebuilt from the stored
  parameters.

The conversion is the pipeline of `pear_deeplab_skin_uv_video.ipynb`, packaged so that it runs on whole datasets;
on the same video it writes bit-identical datasets.

## Install

```
pip install -r requirements.txt        # viewer
pip install -e .                       # optional: the vhrm-studio, vhrm-studio-gui and vhrm-studio-convert commands
```

### Dataset conversion (and mesh views)

Needs an NVIDIA GPU with about 10 GB of free memory (batch 8; lower the batch size otherwise) and:

1. **PyTorch with CUDA** for your driver, from https://pytorch.org, e.g.
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
2. **PyTorch3D** built for that torch / CUDA version. `pip install pytorch3d` is not a reliable route for recent
   PyTorch, especially on Windows: follow the
   [PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
   ([Windows build record](INSTALL_PYTORCH3D_WINDOWS.md)). Do not copy PyTorch3D from another environment.
3. `pip install chumpy --no-build-isolation` (reads the FLAME / SMPL-X model pickles).
4. `pip install -r requirements-pipeline.txt` (ONNX Runtime GPU, ultralytics, ...).
5. **ffmpeg and ffprobe** on `PATH`, or their folder set in *Settings*.
6. **SMPL-X / FLAME body-model assets** (not redistributable: license of the Max Planck Institute for Intelligent
   Systems). One folder holding `SMPLX/` and `FLAME/`, laid out like PEAR's `assets` folder: download the bundle
   linked in the [PEAR README](https://github.com/Pixel-Talk/PEAR), or get `SMPLX_NEUTRAL_2020.npz` from
   https://smpl-x.is.tue.mpg.de and the FLAME 2020 `generic_model.pkl` from https://flame.is.tue.mpg.de and add
   the other files of PEAR's `assets` folder. Set the folder in *Settings* (it is detected automatically when it is
   named `assets` or `PEAR/assets` above the working directory, or given by `$VHRM_ASSETS_DIR`). The settings
   dialog lists any missing file.

Downloaded once at first use: the PEAR checkpoint `ehm_model_stage1.pt` (Hugging Face `BestWJH/PEAR_models`;
`<assets>/../weights/ehm_model_stage1.pt` or a file chosen in *Settings* is used instead when present) and
YOLOv8x (`yolov8x.pt`, by ultralytics, into the vHRM Studio cache folder). Shipped with the package: the
DeepLabV3+ ResNet-50 ONNX model (from `keras-io/deeplabv3p-resnet50`, with an in-graph argmax), the packed semantic
UV layout and the PEAR model code (`src/vhrm_studio/third_party/pear`, Apache 2.0).

Optional: `pip install "triton-windows<3.5"` (Windows) or `triton` (Linux) lets the backbone run through
`torch.compile`.

### Windows OpenMP conflict

`OMP: Error #15` occurs when conda's MKL-backed NumPy and PyTorch initialize different copies of
`libiomp5md.dll`. On Windows vHRM Studio defaults `MKL_THREADING_LAYER` to `SEQUENTIAL` before NumPy is imported
(an explicitly set value is kept), so MKL CPU operations run single-threaded; PyTorch threading and CUDA are
unaffected. Do not use `KMP_DUPLICATE_LIB_OK=TRUE` as a fix.

## Run

```
vhrm-studio                 # or vhrm-studio-gui (no console), or python -m vhrm_studio (with PYTHONPATH=src)
vhrm-studio --convert       # open the dataset conversion window at start-up
```

### Converting a dataset

*Dataset > Convert a dataset* (Ctrl+Shift+C, or *Convert dataset* in the toolbar):

1. **Dataset**: choose the dataset root and *Scan*. Every video file (`.avi`, `.mp4`, `.mov`, `.mkv`, ...) in the
   folder and its subfolders is a source, and so is every folder holding at least 30 images (one image per frame,
   e.g. PURE's `01-01/01-01/Image<ns>.png`). The table lists frames, frame rate, duration and size; untick the
   sources to leave out.
2. **Output**, one of:
   - *Next to each source*: `subject1/vid.avi` gives `subject1/vid_skin_uv.h5`; an image folder `01-01/01-01/`
     gives `01-01/01-01_skin_uv.h5`.
   - *In a new dataset folder with the same structure*: the HDF5 files replace the sources in a copy of the tree.
     The other files of the dataset (ground-truth signals, metadata, readme, ...) are copied too (option), the
     videos / image folders only when *Also copy the source videos* is ticked.
3. **Processing**: batch size, a row limit per source for quick tests, the frame-selection target (30 fps), the
   frame rate of image sequences whose names hold no time stamps, and the optional outputs.
4. **Conversion**: *Start*. The window shows the stage, the dataset progress (sources, rows, rows/s, elapsed,
   estimated remaining time), the current source (row, fps, rows without a person) and the full log. *Stop after
   this source* finishes the current file; *Cancel now* abandons it.

The conversion runs in a separate process (`python -m vhrm_studio.convert`), so the viewer stays usable. The models
load once for the whole dataset (about 30 s, plus the first-use downloads). Expect roughly 20-60 frames/s depending
on the GPU; other GPU programs that hold memory (a notebook kernel, the mesh views) slow it down a lot.

Output root contents:

| File | Content |
|---|---|
| `<name>_skin_uv.h5` | one per source (written as `.h5.partial`, renamed when complete) |
| `vhrm_manifest.json`, `vhrm_manifest.csv` | dataset root, source and output root, and per source: relative source path, relative output path, status, rows, rows with a person, frame rate, time, error |
| `vhrm_conversion_log.txt` | the log of every run |
| `vhrm_conversion.lock` | present while a conversion writes into the folder (one conversion per folder) |

Runs resume: sources whose HDF5 file is complete are skipped unless *Redo sources that are already converted* is
ticked, and partial files of an interrupted run are deleted. The viewer finds the source of an HDF5 file from its
`source_video_path` attribute, then from the manifest of its dataset (source root + relative path, or the copy of
the source in the mirrored dataset), then a file of the same name next to it; *Locate source* picks it by hand.

The same conversion from a terminal (also handy on a headless GPU machine):

```
vhrm-studio-convert C:\dataset\UBFC_DATASET\DATASET_2 --dry-run                  # list sources and outputs
vhrm-studio-convert C:\dataset\UBFC_DATASET\DATASET_2                            # next to the sources
vhrm-studio-convert C:\dataset\UBFC_DATASET\DATASET_2 --out D:\UBFC2_skin_uv     # mirrored dataset
vhrm-studio-convert --help                                                       # all options
```

### Frame sources and frame selection

- Videos are decoded by ffmpeg into NV12 (NVDEC when the codec supports it) and converted on the GPU with the
  stream's colour matrix and range, as in the notebook. Videos stored as RGB (e.g. UBFC-rPPG's uncompressed
  `bgr24` AVI files) are piped as RGB instead, so that their colours are not chroma-subsampled.
- Image sequences are read with OpenCV (8/16-bit, grey, BGR(A)). Their time stamps come from the last number of the
  file names when it is a time stamp in ns, us or ms (PURE), from frame numbers in the names, or from the
  *Image sequence fps* setting; the HDF5 attribute `source_timestamp_source` records which.
- Rows target 30 fps: every frame of sources up to ~30 fps; an exact stride for (near) integer multiples of 30 with
  regular time stamps; otherwise the frame nearest each multiple of 1/30 s.

## HDF5 layout

| Dataset | Shape / dtype | Content |
|---|---|---|
| `uv_skin_rgb` | (N, S, S, 3) uint8 | RGB where visible and skin, 0 elsewhere (S = 1024) |
| `uv_skin_mask_packed`, `uv_visible_mask_packed` | (N, S, S/8) uint8 | `np.packbits(..., axis=-1)` |
| `uv_class_ids` | (N, S, S) uint8 | CIHP id at each visible texel, 0 elsewhere |
| `packed_uv_skin_rgb` | (N, P, P, 3) uint8 | `uv_skin_rgb` in the packed semantic layout |
| `crop_skin_rgb` | (N, C, C, 3) uint8 | skin crop, refined non-skin = 0 (C = 512) |
| `crop_skin_mask_packed` | (N, C, C/8) uint8 | refined skin mask (before erosion) |
| `crop_class_ids` | (N, C, C) uint8 | CIHP labels of the skin crop |
| `crop_to_frame`, `pear_crop_to_frame` | (N, 2, 3) float32 | crop pixel -> frame pixel affines |
| `crop_box_xyxy` | (N, 4) float32 | skin crop box in the frame |
| `body_param/*`, `flame_param/*`, `pd_cam` | float32 | temporally stabilized PEAR parameters |
| `source_frame_index` (int64), `source_timestamp_s` (float64), `frame_status` (uint8) | (N,) | per row; status 1 = ok, 2 = no person, 3 = invalid crop |
| `source_frame_files` | (N,) str | image sequences: the image file of every row |

Static: `uv_body_mask`, `texel_fids`, `texel_bary`, `accumulated_skin_rgb`, `accumulated_skin_mask_packed`,
`packed_uv_body_mask`, `packed_uv_texel_fids`, `packed_uv_texel_bary`, `packed_uv_texel_chart`. The attributes
record the source (`source_video_path`, `source_type`, `source_relative_path`, `dataset_source_root`), the frame
selection, every pipeline setting and the model files. Rows with `frame_status != 1` read as zeros.

## Viewer

- **Folders:** *Browse...* / *Recent* in the Files panel, or *File > Open folder* (Ctrl+O). Folders are scanned
  recursively for `*.h5`; the last folder is reopened at the next start. The *Source* column says whether the source
  video / images were found.
- **Settings** (*File > Settings*, Ctrl+,): body-model assets folder (with *Detect*), ffmpeg folder, PEAR checkpoint
  and YOLO weights (empty = automatic), mesh views on/off, loading the SMPL-X model at start-up, video decode scale.
  Settings of the former Skin-UV viewer are imported once.

```
vhrm-studio [FILE_OR_FOLDER ...] [--assets DIR] [--no-mesh] [--load-mesh] [--video-scale 0.5] [--convert]
```

### Keys

Space play/pause, Left/Right step, Shift+Left/Right 10 rows, Home/End, I/O in/out marks, Ctrl+I clear marks,
PgUp/PgDn previous/next file, F fit all panels, Ctrl+S snapshot, Ctrl+E export video, Ctrl+Shift+C convert a dataset.
In the 3D view: left-drag rotates, wheel zooms, and double-click resets. Enable the arrow-icon **Pan mode** button
beside the view selector to translate with left-drag; Shift-left-drag or middle/right-drag also pans.

### Packed UV view

**Packed UV: skin RGB** displays the row's `packed_uv_skin_rgb` atlas. The UV silhouette toggle uses
`packed_uv_body_mask` when available; without a packed skin mask, nonzero RGB texels are treated as data for this
background display (an observed pure-black texel cannot be distinguished from missing data). Turn the silhouette off
to display the exact stored RGB. Packed and standard UV panels have separate linked zoom/pan groups.

### Standard-pose views

The **Standard pose** views show the whole fitted SMPL-X body in its neutral T-pose (the row's identity and
head/hand scales, without pose and expression), with skin RGB, masks, observed surfaces, coverage, CIHP classes or
accumulated skin attached through the SMPL-X UV topology. **Projected skin crop RGB / refined skin mask / crop CIHP
classes** sample the stored crop using the recorded-pose geometry and crop affines, gated by the saved texel
visibility. Unobserved surfaces remain grey; nothing is filled or inferred. These views need the mesh dependencies.

## Tests

```
python -m unittest discover -s tests -v
```

## Licenses

- `src/vhrm_studio/third_party/pear`: subset of [PEAR](https://github.com/Pixel-Talk/PEAR), Apache License 2.0;
  see its `NOTICE.md` for the files and changes.
- The SMPL-X and FLAME models are not included; their own licenses apply.
- The YOLOv8 detector is used through [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- The DeepLabV3+ weights come from the Hugging Face model `keras-io/deeplabv3p-resnet50`.
