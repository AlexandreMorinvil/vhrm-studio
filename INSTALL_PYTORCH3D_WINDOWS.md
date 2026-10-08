# PyTorch3D installation in vHRM_pretransformation

Installation verified on 2026-10-05, on Alexandre's Windows computer.

This records the installation used to enable the skin-UV viewer's PEAR mesh
views. It is not a general PEAR environment installation or a recommendation to
upgrade an already working environment.

## What Is Confirmed

PyTorch3D was built from the official source and installed in
`C:\Users\Alexandre\anaconda3\envs\vHRM_pretransformation`.
Its compiled extension imports, the PEAR model loads, and CUDA rasterization
produces a nonblank image.

| Component | Verified value | Scope |
| --- | --- | --- |
| Python | 3.12.13, 64-bit, conda | This environment |
| PyTorch | 2.5.1+cu124 | Existing version, preserved |
| PyTorch3D | 0.7.9 | Installed from source |
| Source revision | `v0.7.9` | Recorded in installed package metadata |
| Source commit | `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` | Exact installed revision |
| CUDA toolkit used for compilation | 12.6 | Already installed on this computer |
| C++ compiler selected during troubleshooting | MSVC 14.42, Visual Studio 2022 | This computer's usable compiler |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU | This computer |
| CUDA compute capability | 8.9 | This GPU |
| NumPy | 1.26.4 | Restored compatibility pin |
| SciPy | 1.17.1 | Original conda version, repaired |
| Chumpy | 0.70 | Added for PEAR's legacy model pickles |
| iopath | 0.1.10 | Available PyTorch3D dependency |
| OpenCV Python distribution | 4.9.0.80 (`cv2.__version__` is 4.9.0) | Verified import after repairs |

**Evidence limit:** the original successful compiler command is not available
as a complete transcript in the retained session evidence. The compiler setup
and build commands below are reconstructed equivalents of the recorded steps,
not a claim that every character was typed exactly this way. The source commit,
final versions, dependency repairs, and successful mesh test are directly
verified. The retained `PEAR/pytorch3d_build.log` describes the older
`vHRM_PEAR`/Python 3.9 installation, not this Python 3.12 build.

## 1. Diagnose the Missing Dependency

The viewer's `requirements.txt` installs its core GUI dependencies. PyTorch3D
appears only in comments; it is not installed by `pip install -r requirements.txt`.
The viewer's `mesh` extra also does not install PyTorch3D.

PEAR requires PyTorch3D while constructing the SMPL-X/EHM model, including
Laplacians and pose conversions. It is not merely an optional rasterizer once
mesh views are enabled.

**General:** packages in one conda environment are not shared with another.
The working PyTorch3D installation in `vHRM_PEAR` could not simply be reused or
copied into `vHRM_pretransformation`: its compiled extension targets a different
Python/PyTorch combination.

## 2. Target the Correct Interpreter

The interpreter was explicitly selected rather than relying on terminal PATH.
The following PowerShell variable is used throughout this document:

```powershell
$python = 'C:\Users\Alexandre\anaconda3\envs\vHRM_pretransformation\python.exe'
& $python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

**Machine-specific:** replace the interpreter path on another computer.
**General:** using `python -m pip` through this exact executable prevents
accidentally installing into another environment. PyTorch was already installed
and CUDA worked; there was no need to reinstall PyTorch, torchvision, or torchaudio.

## 3. Prepare the Windows Source-Build Tools

The build used the existing Visual Studio C++ tools and CUDA toolkit 12.6.
Ninja was made available as a build runner. Its exact installation command is
not retained, and the final environment has no pip distribution metadata for
`ninja`; do not infer that it was necessarily installed with pip.

**General for a CUDA source build on Windows:** a supported C++ compiler, the
Windows SDK headers/libraries, CUDA's `nvcc`, Git for the VCS install, and the
build-time Python packages must be available. Ninja speeds up the build but is
not a universal requirement for every PyTorch extension installation.

**Machine-specific troubleshooting:** initial attempts could not find `cl.exe`.
MSVC 14.39 directories were present but did not contain the compiler executable;
the usable compiler was MSVC 14.42. A directory's existence is not proof of an
installed compiler. After the compiler was found, another attempt lacked CRT
headers, so the include/library paths needed correction.

The following is a reconstructed setup for a **fresh PowerShell session**. It
discovers the Visual Studio root and supplies compiler, SDK, and CUDA paths to
the same process that launches pip. These are process-local settings, not
permanent machine environment changes.

```powershell
$python = 'C:\Users\Alexandre\anaconda3\envs\vHRM_pretransformation\python.exe'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw 'No Visual Studio C++ tools installation found.' }

Import-Module "$vsRoot\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'

$msvc = Get-ChildItem "$vsRoot\VC\Tools\MSVC" -Directory |
    Where-Object { $_.Name -like '14.42.*' -and (Test-Path "$($_.FullName)\bin\Hostx64\x64\cl.exe") } |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $msvc) { throw 'MSVC 14.42 with cl.exe was not found.' }

$kits = "${env:ProgramFiles(x86)}\Windows Kits\10"
$sdk = Get-ChildItem "$kits\Include" -Directory |
    Where-Object {
        (Test-Path "$($_.FullName)\ucrt\stdio.h") -and
        (Test-Path "$kits\Lib\$($_.Name)\ucrt\x64") -and
        (Test-Path "$kits\Lib\$($_.Name)\um\x64")
    } | Sort-Object Name -Descending | Select-Object -First 1
if (-not $sdk) { throw 'A complete Windows SDK was not found.' }

$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$($msvc.FullName)\bin\Hostx64\x64;$env:CUDA_HOME\bin;$env:PATH"
$env:INCLUDE = @(
    "$($msvc.FullName)\include"
    "$($sdk.FullName)\ucrt"
    "$($sdk.FullName)\shared"
    "$($sdk.FullName)\um"
    "$($sdk.FullName)\winrt"
    "$env:CUDA_HOME\include"
    $env:INCLUDE
) -join ';'
$env:LIB = @(
    "$($msvc.FullName)\lib\x64"
    "$kits\Lib\$($sdk.Name)\ucrt\x64"
    "$kits\Lib\$($sdk.Name)\um\x64"
    "$env:CUDA_HOME\lib\x64"
    $env:LIB
) -join ';'

Get-Command cl.exe, nvcc.exe, ninja.exe
```

If Ninja is missing, a possible setup command is
`& $python -m pip install --no-deps ninja`; this is a suggested equivalent,
not the recovered historical command. Likewise, install any missing PyTorch3D
runtime dependencies deliberately rather than disabling dependencies in a new,
otherwise empty environment.

**General compatibility caution:** the CUDA toolkit used to compile an extension
is different from the CUDA runtime supplied with the PyTorch wheel. This machine
used toolkit 12.6 with PyTorch's CUDA 12.4 build, and the final rendering test
passed. That result does not establish that arbitrary toolkit/PyTorch/compiler
combinations work; matching the toolkit to PyTorch is preferable when setting up
a new machine. No unsupported-compiler override is prescribed here.

## 4. Build PyTorch3D from the Official Source

PyTorch3D `v0.7.9` was selected, with dependency upgrades and isolated build
environments disabled. This lets the extension build against the already
installed PyTorch rather than a different build-time installation.

Reconstructed build command, in the compiler-configured PowerShell session:

```powershell
$env:TORCH_CUDA_ARCH_LIST = '8.9'
& $python -m pip install --no-deps --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch3D installation failed.' }
```

- **Machine-specific:** `8.9` targets this RTX 4080 Laptop GPU and reduces the
  architectures compiled. Choose the target GPU's capability on another machine,
  or configure multiple architectures when distributing the build.
- **General for this existing environment:** `--no-deps` protects existing package
  versions; it does not install missing runtime requirements. `--no-build-isolation`
  requires build dependencies to be present in the target environment.
- **Reproducibility:** the installed package's `direct_url.json` records commit
  `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`. The Git URL can use that commit
  instead of the tag to select exactly the recorded source.

The installed extension was verified at
`C:\Users\Alexandre\anaconda3\envs\vHRM_pretransformation\Lib\site-packages\pytorch3d\_C.cp312-win_amd64.pyd`.
Importing the Python package alone was not treated as sufficient verification.

## 5. Install Chumpy and Repair Dependency Changes

The first full PEAR mesh test then failed with
`ModuleNotFoundError: No module named 'chumpy'` while unpickling a model asset.
The package-install tool was used to install `chumpy==0.70` into the selected
environment. A command-line equivalent of that request is:

```powershell
& $python -m pip install chumpy==0.70
```

**PEAR/asset-specific:** Chumpy is needed for these legacy model pickles, not for
PyTorch3D itself. That package-install request unexpectedly changed transitive
dependencies. NumPy became 2.5.3 and OpenCV failed to import. NumPy was restored
to 1.26.4 through the package-install tool; its command-line equivalent is:

```powershell
& $python -m pip install numpy==1.26.4
```

The next test exposed mixed SciPy files: distribution metadata reported 1.13.1,
while conda's original record was 1.17.1, and importing SciPy failed with a
missing `_promote` symbol. The following terminal command was actually run to
repair SciPy without changing its dependencies:

```powershell
& 'C:/Users/Alexandre/anaconda3/envs/vHRM_pretransformation/python.exe' -m pip install --no-deps --force-reinstall scipy==1.17.1
```

**Specific to this installation incident:** the SciPy force-reinstall and NumPy
restoration were repairs, not normal prerequisites for installing PyTorch3D.
They should not be performed indiscriminately on another working environment.

**Recommended for a repeat installation, not the original command:** if Chumpy's
requirements are already satisfied, use
`& $python -m pip install --no-deps chumpy==0.70` to avoid the dependency changes
encountered here. Verify imports afterward; legacy Chumpy compatibility can vary
with Python and NumPy. No Chumpy source patch or NumPy downgrade below 1.26.4 was
needed for the final PEAR mesh test in this environment.

## 6. Verify the Extension and the Actual Mesh

The following minimal import check confirms the compiled extension is loadable:

```powershell
& $python -c "import torch, pytorch3d; from pytorch3d import _C; print(torch.__version__); print(pytorch3d.__version__); print(_C.__file__)"
```

The final verification also constructed the real PEAR model and rasterized it
on CUDA. Run this Python code in `vHRM_pretransformation`, with the working
directory set to `C:\Projects\vHRM2_develop`:

```python
import sys
from pathlib import Path

import numpy as np
import torch
import pytorch3d
import cv2
import scipy
import scipy.optimize

print('torch:', torch.__version__, 'numpy:', np.__version__,
      'scipy:', scipy.__version__, 'pytorch3d:', pytorch3d.__version__)
sys.path.insert(0, str(Path('vhrm_studio/src').resolve()))
from vhrm_studio.mesh import SmplxMesh

model = SmplxMesh('PEAR/assets', device='cuda:0')  # the SMPL-X / FLAME assets folder
assert model.can_rasterize
image = model.render_orbit(model.ehm.v_template.detach().cpu().numpy(), size=64)
assert image.shape == (64, 64, 3)
assert np.any(image != np.array((22, 22, 26), dtype=np.uint8))
torch.cuda.synchronize()
print('PASS: PEAR mesh loaded and CUDA rendered a nonblank image')
```

Actual successful output included:

```text
torch: 2.5.1+cu124 numpy: 1.26.4 scipy: 1.17.1 pytorch3d: 0.7.9
Using generic FLAME model
Using generic FLAME model
PASS: PEAR mesh loaded and CUDA rendered a nonblank image
```

**Project-specific:** this test requires the local PEAR source and its existing
FLAME/SMPL-X assets. They were not downloaded or regenerated during this install.
It tests model loading and CUDA rasterization, not every PEAR inference workflow.

## 7. Restart the Viewer

The already running viewer was not forcibly terminated. Restart it so it uses
the completed environment, then load the mesh again. The explicit-interpreter
launch equivalent is:

```powershell
Set-Location 'C:\Projects\vHRM2_develop\vhrm_studio'
& $python -m vhrm_studio
```

(This record was written for the former `skin_uv_viewer` package, now vHRM Studio; the commands above use the
new names.)

The viewer was already installed in this environment. On a new setup, install
the core requirements and the editable viewer package separately as described
in [README.md](README.md). Do not copy a compiled PyTorch3D extension from another
environment or assume that the compiler setup is necessary on every launch;
the build tools are needed for compilation, not for normal viewer use.

## Reference

For a different platform or version combination, consult the official
[PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/INSTALL.md)
instead of treating this machine's paths, compiler version, or GPU architecture
as universal requirements.