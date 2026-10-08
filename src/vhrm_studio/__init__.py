"""vHRM Studio (Visual Heart Rate Monitoring): converts rPPG video datasets into per-video skin-UV HDF5 files
(PEAR SMPL-X UV mapping + DeepLabV3+ skin segmentation) and views them."""
import os
import sys

__version__ = '0.2.0'

if sys.platform == 'win32':
    # conda's MKL numpy and PyTorch otherwise load two OpenMP runtimes (OMP: Error #15); see README.
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
