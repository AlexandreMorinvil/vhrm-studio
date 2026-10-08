# PEAR subset

Copied from PEAR (https://github.com/Pixel-Talk/PEAR), Copyright the PEAR authors (International Digital Economy
Academy), licensed under the Apache License 2.0 (`LICENSE.txt` in this folder). Only the parts needed to run the
pretrained model for inference are included.

| File | Changes |
|---|---|
| `configs/infer.yaml` | none |
| `models/backbones/vit.py` | imports `drop_path`, `to_2tuple`, `trunc_normal_` from `timm.layers` when available |
| `models/smplx/smplx_head.py`, `pose_transformer.py`, `t_cond_mlp.py` | unused imports removed; the SMPL mean parameters are read from `data/smpl_mean_params.npz` next to this package instead of the working directory |
| `models/modules/ehm/EHM_v2.py` | unused imports (MANO, SMPLX v2) removed; package-relative import of `rotation_converter` |
| `models/modules/flame/*`, `models/modules/smplx/*`, `utils/rotation_converter.py` | none |
| `data/smpl_mean_params.npz` | from PEAR's `assets/SMPLX` |
| `__init__.py` | new: configuration loading and `Ehm_Pipeline` without the PyTorch Lightning base class |

The local copy includes the modifications made in this workspace's PEAR checkout (for example the GPU-side
`pd_cam` rotation in `smplx_head.py`). The pretrained weights are downloaded from the Hugging Face repository
`BestWJH/PEAR_models` at first use. The SMPL-X and FLAME model files are not part of this package: they are
licensed separately by the Max Planck Institute for Intelligent Systems.
