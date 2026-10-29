# Environment setup: one complete install or capability profiles

Use Python 3.12 and a dedicated writable environment. Setup installs the selected dependencies and builds CUDA extensions for rendering profiles. Download models and example data separately using the [asset guide](../assets/README.md). Run the commands from the repository root and choose your environment and cache paths.

## Complete environment

The reference GPU environment uses Python 3.12, NumPy 1.26.4, PyTorch 2.9.1+cu130, torchvision 0.24.1, CUDA toolkit 13.2, PyTorch3D 0.7.9 and nvdiffrast 0.4.0 on Blackwell `sm_120`. A compatible NVIDIA driver, CUDA toolkit/compiler and build toolchain must already be available. Setup does not install drivers, CUDA or system packages.

```bash
python3.12 environment/setup.py --profile all --prefix /path/to/env/bfsm \
  --cache-dir /path/to/cache/bfsm --cuda-arch 12.0 --jobs 4
source /path/to/env/bfsm/bin/activate
export BFSM_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/path/to/cache/bfsm/xdg
export TORCH_HOME=/path/to/cache/bfsm/torch
export TORCH_EXTENSIONS_DIR=/path/to/cache/bfsm/extensions
export MPLCONFIGDIR=/path/to/cache/bfsm/matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export VT_PYTORCH3D_SOURCE=/path/to/cache/bfsm/sources/pytorch3d
export VT_REGISTRATION_ASSETS="$PWD/assets/external/registration"
export ALBUMENTATIONS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
python -m pip check
```

Change `--cuda-arch` and `--torch-index` for another supported GPU only after verifying its compiler/wheels. `12.0` is the verified hardware, not a universal GPU default. `CUDA_HOME` must identify the actual toolkit when automatic discovery is insufficient. View commands without installation with `--plan`.

Setup creates a dedicated venv and installs pip from a SHA-256-verified PyPI wheel when pip is absent. Dependency caches, cloned provider sources and temporary files stay below the selected cache directory. Editable-package metadata and the in-place CPU renderer build can write ignored files under the checkout; the compiled renderer stays local and is not distributed with the source. Fixed source commits are defined in `setup.py`; CUDA 13 symbol-visibility flags are explicit there. Network access is needed for dependency installation.

## Select local workflow packages

Setup installs the root `bfsm_core` and all five local workflow packages in editable mode by default. `--modules` selects code modules independently of dependency capability profiles:

```bash
python3.12 environment/setup.py --profile all --modules 03 04 \
  --prefix /path/to/env/bfsm --cache-dir /path/to/cache/bfsm --cuda-arch 12.0
```

Module 04 includes its module 01 and module 03 dependencies automatically. Selecting code does not install every external capability: the complete photo-plus-Plan workflow needs registration and image capabilities, or `--profile all`. Omit `--modules` to install all local packages. Editable packages locate their checkout automatically. Standard venv activation does not set `BFSM_ROOT`; explicitly use `export BFSM_ROOT="$PWD"` from the repository root when selecting that checkout's assets/data. Set it to the checkout path for wheel installations or an explicit project location when running elsewhere.

For an already compatible environment, manually install the root plus chosen module directories with `python -m pip install -e . -e modules/03_image_reconstruction`. Include module 01 and module 03 when manually installing module 04. Installing the editable code is separate from downloading model/provider assets.

## Smaller environments by capability

Profiles correspond to dependencies, not the number of project modules. Each command can use the same dedicated prefix when combining capabilities; `all` is simplest if disk/build time permits.

| Profile | Capability | Project operations |
| --- | --- | --- |
| `cpu` | NumPy/SciPy/trimesh, sklearn, matplotlib, Plotly | Module 1 PCA construction/synthetic check; NumPy native FLAME identity decode/fits; region OBJ export |
| `torch` | Base plus Torch/torchvision and image I/O | Shared Torch FSMM decoding/fitting on CPU or GPU; DECA with explicit landmarks; combine with `cpu` for Module 2 mode generation |
| `render` | Torch plus fixed PyTorch3D | Scan ARAP and postoperative grids after frontend preparation; geometry rasterization; parent of registration/image profiles |
| `registration` | Render plus FAN/Open3D/ray/NICP dependencies | Module 1 original/CT-only registration, external Plan registration and Module 4 scan frontend |
| `image` | Render plus nvdiffrast and image frontend dependencies | Module 3 reconstruction, Module 4 photograph frontend and original TMM demonstration; add `registration` for the complete external-Plan postoperative route |
| `all` | Union of above | All implemented operations |

For PCA construction on your own registered face/skull pairs:

```bash
python3.12 environment/setup.py --profile cpu --modules 01 --prefix /path/to/env/bfsm-cpu --cache-dir /path/to/cache/bfsm-cpu
source /path/to/env/bfsm-cpu/bin/activate
python -m bfsm_topology.construction.synthetic_smoke --output data/output/runs/synthetic_pca
```

For registration or image reconstruction, select the matching profile and module, set the CUDA architecture, and apply the cache and asset exports from the complete setup. For mode generation, install `cpu` and then `torch` into the **same dedicated prefix** (or use `all`). This adds fitting dependencies without building unused CUDA rasterizers. Native FLAME fitting is SciPy-based and does not need a GPU. DECA without supplied landmarks additionally needs FAN; use `all` or `registration`, or install `face-alignment==1.4.1` into the Torch profile for automatic detection.

## Reproducibility

The profile files pin application dependencies. Transitive dependencies can vary by platform and resolver date; record `python -m pip freeze` for your execution. The shared FSMM decoder requires the Torch capability even when using a CPU device. The NumPy native FLAME identity decoder is separate. The public `bfsm_applications.decode` command includes the original 5000-step tissue-face fit and requires the Torch capability as well.

For deterministic CUDA registration, see the [runtime backend guide](../common/README.md#deterministic-pytorch3d-cuda-rasterization). Results can differ across GPU/toolchain versions and between CPU and CUDA execution.

Model and checkpoint installation is described in the [asset guide](../assets/README.md).
