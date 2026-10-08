"""Subset of PEAR (https://github.com/Pixel-Talk/PEAR, Apache License 2.0) used by vHRM Studio: the ViT backbone,
the SMPL-X transformer head and the EHM_v2 body model (SMPL-X body + FLAME head). See NOTICE.md for the list of
files and changes. The SMPL-X / FLAME model files themselves are not included (see vhrm_studio.assets).
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / 'configs' / 'infer.yaml'
CHECKPOINT_REPO = 'BestWJH/PEAR_models'
CHECKPOINT_FILE = 'ehm_model_stage1.pt'


def load_config(path=CONFIG_PATH):
    """infer.yaml as the read-only OmegaConf node PEAR's ConfigDict exposes."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(path))
    OmegaConf.set_readonly(cfg, True)
    return cfg


def build_ehm_pipeline(cfg):
    """PEAR's Ehm_Pipeline without its Lightning training scaffolding: backbone, head and input normalization."""
    import torch
    import torchvision

    from .models.backbones import ViT
    from .models.smplx.smplx_head import SMPLXTransformerDecoderHead

    class EhmPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ViT(**cfg.BACKBONE)
            self.head = SMPLXTransformerDecoderHead(cfg.HEAD, cfg.TRAIN.batch_size)
            self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def forward(self, x):
            x = self.normalize(x)
            return self.head(self.backbone(x[:, :, :, 32:-32]))

    return EhmPipeline()


def build_ehm(assets_dir):
    """EHM_v2 (SMPL-X body + FLAME head) on the CPU, without its training-only dense Laplacians (~1.7 GB)."""
    from .models.modules.ehm import EHM_v2
    assets_dir = Path(assets_dir)
    ehm = EHM_v2(str(assets_dir / 'FLAME'), str(assets_dir / 'SMPLX')).eval()
    for module in (ehm, ehm.smplx):
        for name in ('laplacian_matrix', 'laplacian_matrix_negate_diag'):
            module._buffers.pop(name, None)
    return ehm
