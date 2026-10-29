# Shared BFSM core

The root editable package provides `bfsm_core` from `common/src/`. It contains the shared face/skull decoder, validated mesh I/O, model/asset paths, display colors, geometry renderers and deterministic runtime adapter. Scientific workflows and their configurations belong to the five numbered [modules](../README.md#modules).

Editable installation locates the checkout automatically. To select a project explicitly, set `BFSM_ROOT` to its absolute path; standard venv activation alone does not set this variable.

After installing the required [environment capabilities](../environment/README.md), install the core from the repository root:

```bash
python -m pip install -e .
```

Load a model with the shared decoder (requires the Torch capability):

```python
from bfsm_core.model import FaceSkullModel
model = FaceSkullModel('assets/models/fsmm_200.npz', device='cpu')
print(model.face_count, model.skull_count)
```

Display an existing mesh with the shared fixed-camera renderer:

```bash
python -m bfsm_core.visualize \
  --mesh before_skull=data/output/reference/01_topology/sample_001/registration/posed/topology_skull.obj \
  --output data/output/runs/skull_views --device cpu --neutral
```

Rendering additionally needs the render capability. `bfsm_core.paper_figure` rearranges actual renderer panels without changing mesh geometry or scientific pixels. The runtime section below documents deterministic CUDA settings and compiler/source-header requirements. [Model formats](../assets/README.md#geometry-model-formats) define geometry and coefficient conventions.

## Deterministic PyTorch3D CUDA rasterization

The implementation is in `common/src/bfsm_core/runtime/`. This opt-in adapter uses PyTorch3D 0.7.9 coarse/fine forward operators and
differentiation formulas. In strict deterministic mode, it sorts coarse face
candidates by packed face index before fine rasterization, making equal-depth
visibility ties reproducible. It removes only the common all-padding tail after
verifying every bin contains a packed valid prefix; no candidate is dropped.
Depths are never perturbed. Non-tied forward values are unchanged; tied pixels
may select a different face from upstream scheduling-dependent selection.
CPU, unbinned rasterization and non-strict forward calls remain upstream.

It replaces unordered atomic additions in mesh raster and attribute
interpolation backward with independent contributions and PyTorch deterministic
`index_add_`. The accumulation order changes, so small float32 differences from
upstream CUDA are expected. It does not change losses, optimizer settings,
iteration counts, camera projection or mesh topology.

Get the public source headers at the validated revision:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git /path/to/cache/sources/pytorch3d
git -C /path/to/cache/sources/pytorch3d checkout 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
export VT_PYTORCH3D_SOURCE="/path/to/cache/sources/pytorch3d"
export TORCH_EXTENSIONS_DIR="/path/to/cache/torch_extensions"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Install PyTorch3D 0.7.9, a matching CUDA-enabled PyTorch, ninja and the CUDA
compiler. The validated runtime is PyTorch 2.9.1+cu130 with CUDA toolkit 13.2 on
`sm_120`. Build flags restore CUDA 13 template symbol visibility used by the
validated upstream build. Older compiler support has not been verified.

With the core editable package installed, initialize before running the pipeline in each process:

```python
import torch
from bfsm_core.runtime.deterministic_pytorch3d import install

torch.use_deterministic_algorithms(True)
install()
```

First use compiles the two shipped source files into the chosen cache. The
headers come from the pinned PyTorch3D source checkout. `uninstall()` restores upstream functions in that process.
The adapter requires CUDA float32 raster inputs, the upstream raster output
contract and nonnegative packed face indices or -1 padding. It preserves the
upstream perspective-correction and barycentric-clipping formulas, including
existing CUDA/CPU differences when both flags are true. It does not claim
double backward or bitwise agreement across different hardware/software stacks.

Raster contribution scratch memory is `N * H * W * K * 9 * 4` bytes, before
valid-index filtering and reduction temporaries. Coarse-candidate sorting and prefix validation also use temporary GPU memory.
This is a reproducibility backend with additional work; removing trailing
padding can reduce fine rasterization time without changing its arithmetic.

`pixel_contributions.cu` derives from the Meta PyTorch3D raster backward kernel;
its nine per-face atomic additions are replaced by nine writes to a unique
pixel row. See the [PyTorch3D license](../THIRD_PARTY_NOTICES.md#pytorch3d). The Python shim and bindings
implement local integration and require the same upstream API version.

## Adding a module

Add `modules/06_name/` with `README.md`, `figures/`, `src/bfsm_newmodule/` and `pyproject.toml`. Setup discovers numbered modules automatically. Declare package dependencies in `[project].dependencies`; also list local module dependencies in `[tool.bfsm] requires-modules` so setup installs them together. Keep reusable code in `common/`, models in `assets/`, and case inputs/results in `data/`.
