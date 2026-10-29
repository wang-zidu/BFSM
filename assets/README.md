# Download models and example data

This page lists assets for the Face–Skull Morphable Model (FSMM), Tissue Morphable Model (TMM), and their registration and reconstruction workflows.

## Installed asset layout

Installed files are grouped by module below. Preparation inputs are listed with their download commands.

```text
assets/
├── README.md
├── models/
│   ├── fsmm_200.npz                                      # M1 topology/pairing template; M2–M4; M5 model build
│   ├── tmm_200.npz                                       # M2 tissue modes; M5 TMM construction
│   ├── flame_fsmm_200.npz                                # M5; built locally with FLAME assets
│   └── flame_tmm_200.npz                                 # M5; built locally with FLAME assets
├── templates/
│   ├── muFACE.obj                                        # M1/M4; fixed-order face template
│   └── muSKULL.obj                                       # M1/M4; fixed-order skull template
└── external/
    ├── image/                                            # M3; M4 photograph frontend
    │   ├── face_model.npy                                # BFM geometry/texture data
    │   ├── epoch_20_deep3d_closeeye.pth                  # Deep3D reconstruction checkpoint
    │   ├── large_base_net.pth                            # Face-alignment network checkpoint
    │   ├── retinaface_resnet50_2020-07-20_old_torch.pth  # Face detector
    │   └── similarity_Lm3D_all.mat                       # Five-point alignment reference
    ├── registration/                                     # M1; M4 scan and external-Plan paths
    │   ├── BFM/
    │   │   ├── BFM09_model_info.mat                      # Prepared BFM09 model, including albedo
    │   │   └── ldm68.npy                                 # Ordered 68 template landmark indices
    │   ├── config/
    │   │   ├── coarse_grain.toml                         # Coarse NICP settings
    │   │   └── fine_grain.toml                           # Fine NICP settings
    │   └── template/
    │       ├── muFACE.obj                                # Registration template copy
    │       └── muSKULL.obj                               # Registration template copy
    ├── flame/
    │   └── flame_neutral.npz                             # M5; FLAME identity, expression and skinning
    └── DECA/                                             # M5 DECA image frontend only
        ├── decalib/
        │   ├── __init__.py
        │   └── models/
        │       ├── encoders.py                           # External DECA encoder implementation
        │       └── resnet.py                             # External DECA ResNet implementation
        └── data/
            └── deca_model.tar                            # External DECA checkpoint
```

Download the [BFSM model bundle](https://huggingface.co/datasets/Zidu-Wang/BFSM) for geometry models and the exact registration templates. Example inputs and selected reference outputs are included in this GitHub repository. Obtain third-party models and checkpoints separately from their providers, using the paths and commands on this page.

Run the commands below from the BFSM **repository root** after configuring the environment and required modules. The download installs the models and templates at their repository-relative paths.

## 1. Download the BFSM bundle

Install `huggingface_hub` if it is not already available, then download the four model and template files into the checkout:

```bash
python -m pip install huggingface_hub
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Zidu-Wang/BFSM", repo_type="dataset", local_dir=".", allow_patterns=["assets/models/fsmm_200.npz", "assets/models/tmm_200.npz", "assets/templates/muFACE.obj", "assets/templates/muSKULL.obj"])'
```

The BFSM bundle contains:

| Installed path | Contents |
| --- | --- |
| `assets/models/fsmm_200.npz`, `tmm_200.npz` | Geometry-only models, 200 components each |
| `assets/templates/muFACE.obj`, `muSKULL.obj` | Exact fixed-order registration template pair |


## 2. Prepare assets for your workflow

| Workflow | Required assets and preparation |
| --- | --- |
| FSMM/TMM visualization and tissue variation | BFSM bundle |
| PCA model construction | Your own registered face/skull pairs, topology/pairing template and construction configuration; see the prerequisites below |
| Textured scan registration; scan-based postoperative prediction | [Registration inputs](#registration-inputs) and [detector cache](#detector-cache) |
| CT-only registration with known XYZ landmarks | [Registration inputs](#registration-inputs) |
| Image reconstruction | [Image reconstruction files](#image-reconstruction-files) |
| Image-based postoperative prediction with an external Plan | [Image reconstruction files](#image-reconstruction-files) and [registration inputs](#registration-inputs) |
| BFSM-in-FLAME | [FLAME correspondence, masks and model build](#flame-correspondence-and-masks) |
| External FLAME fitting | [FLAME model build](#flame-correspondence-and-masks) and [neutral FLAME2020](#neutral-flame2020) |
| DECA image frontend | [FLAME model build](#flame-correspondence-and-masks), [neutral FLAME2020](#neutral-flame2020), [DECA](#deca-source-and-checkpoint) and [detector cache](#detector-cache) |

We provide our FSMM and TMM geometry models with 200 principal components each. Institutions interested in developing their own private, internal bidirectional face–skull models can use our registration and PCA construction code to build customized face–skull and tissue models, provided they have the capability to acquire paired 3D face and skull data with anatomically consistent spatial correspondence.

To build your own models, prepare corresponding face/skull pairs using the [textured-scan registration](../modules/01_topology_and_modeling/README.md#original-textured-scan-path) or [CT-only registration](../modules/01_topology_and_modeling/README.md#ct-only-registration) workflow. Our registration workflows produce meshes with consistent template topology and vertex order in a normalized coordinate frame. Use these processed meshes as inputs to PCA model construction. Configure the face–skull correspondences, templates, regions and sample manifest as described in the [construction input requirements](../modules/01_topology_and_modeling/docs/PCA_CONSTRUCTION.md#inputs), and choose the component count according to the number of paired cases and numerical rank. Your models do not need to use 200 components. See the [PCA construction command](../modules/01_topology_and_modeling/docs/PCA_CONSTRUCTION.md#run) to build the models.

### Image reconstruction files

Place these files directly under `assets/external/image/`:

| Target filename | Download |
| --- | --- |
| `face_model.npy` | [3DDFA-V3 provider mirror](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/ffedf42ca91a072beaaf3bc858fcd752ab3b1502/assets/face_model.npy); contains BFM-derived geometry/texture data, so first obtain permission under the [BFM09 terms](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) |
| `epoch_20_deep3d_closeeye.pth` | Download the closed-eye `epoch_20.pth` from the [original Deep3D model folder](https://drive.google.com/drive/folders/1grs8J4vu7gOhEClyKjWU-SNxfonGue5F), then rename it to the target filename |
| `large_base_net.pth` | [3DDFA-V3 provider mirror](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/ffedf42ca91a072beaaf3bc858fcd752ab3b1502/assets/large_base_net.pth) |
| `retinaface_resnet50_2020-07-20_old_torch.pth` | [3DDFA-V3 provider mirror](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/ffedf42ca91a072beaaf3bc858fcd752ab3b1502/assets/retinaface_resnet50_2020-07-20_old_torch.pth) |
| `similarity_Lm3D_all.mat` | [Original Deep3D alignment reference](https://raw.githubusercontent.com/sicxu/Deep3DFaceRecon_pytorch/9167c7136aebef0c54d3ac74f4b0396222a491c1/BFM/similarity_Lm3D_all.mat) |

### Registration inputs

1. Apply for [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads), accept the provider's terms and obtain `01_MorphableModel.mat`. Put it in `assets/external/bfm_inputs/`. The BFM09 terms prohibit redistribution of model data; neither this original file nor `BFM09_model_info.mat` belongs in the BFSM bundle.
2. Read the [expression-model provider's terms](https://github.com/Juyong/3DFace). The commands below download its `Exp_Pca.bin` and four pinned fitting-topology inputs, then run the BFSM numerical converter locally. They do not apply for or grant BFM access.

```bash
mkdir -p assets/external/bfm_inputs
for asset in BFM_front_idx.mat BFM_exp_idx.mat facemodel_info.mat std_exp.txt; do
  curl --fail --location \
    "https://raw.githubusercontent.com/ascust/3DMM-Fitting-Pytorch/8ca87fc3ce42555d376dc505858e96536068896d/BFM/$asset" \
    --output "assets/external/bfm_inputs/$asset"
done
curl --fail --location \
  https://raw.githubusercontent.com/Juyong/3DFace/2b19c9d36cb5e07de9d4dc1203071161abd66110/Exp_Pca.bin \
  --output assets/external/bfm_inputs/Exp_Pca.bin

python -m bfsm_topology.provision_assets \
  --source assets/external/bfm_inputs --templates assets/templates \
  --output assets/external/registration
export VT_REGISTRATION_ASSETS="$PWD/assets/external/registration"
```

Before running the converter, the source folder must contain `01_MorphableModel.mat`, `Exp_Pca.bin`, `BFM_front_idx.mat`, `BFM_exp_idx.mat`, `facemodel_info.mat` and `std_exp.txt`. The converter validates its source versions and produces:

```text
assets/external/registration/
├── BFM/
│   ├── BFM09_model_info.mat
│   └── ldm68.npy
├── template/
│   ├── muFACE.obj
│   └── muSKULL.obj
└── config/
    ├── coarse_grain.toml
    └── fine_grain.toml
```

The command also installs the two BFSM templates and NICP settings into the `template/` and `config/` subdirectories shown above.

### Detector cache

FAN and SFD are downloaded from the original provider, outside the BFSM bundle. Set a writable local Torch cache:

```bash
export TORCH_HOME="$PWD/.cache/torch"
mkdir -p "$TORCH_HOME/hub/checkpoints"
```

| Required by | Exact filenames under `$TORCH_HOME/hub/checkpoints/` |
| --- | --- |
| Textured scan registration and scan postoperative frontend | `3DFAN4-4a694010b9.zip`, `depth-6c4283c0e0.zip`, `s3fd-619a316812.pth` |
| DECA automatic face/landmark detection | `2DFAN4-cd938726ad.zip`, `s3fd-619a316812.pth` |

Download only the files needed by your workflow, or all four for both:

```bash
for asset in 3DFAN4-4a694010b9.zip depth-6c4283c0e0.zip 2DFAN4-cd938726ad.zip s3fd-619a316812.pth; do
  curl --fail --location "https://www.adrianbulat.com/downloads/python-fan/$asset" \
    --output "$TORCH_HOME/hub/checkpoints/$asset"
done
```

Keep the exact filenames; the ZIP-named files are TorchScript archives loaded directly and must not be extracted. Use the documented `face-alignment==1.4.1` environment. The image reconstruction frontend uses its RetinaFace checkpoint.

### FLAME correspondence and masks

Download the correspondence and masks below, then build the FLAME-topology models.

```bash
mkdir -p assets/external/flame_inputs
curl --fail --location \
  https://raw.githubusercontent.com/TimoBolkart/BFM_to_FLAME/44a920347e508f1aedac6aca36cf98f4581358ab/data/BFM_to_FLAME_corr.npz \
  --output assets/external/flame_inputs/BFM_to_FLAME_corr.npz
curl --fail --location \
  https://raw.githubusercontent.com/sicxu/Deep3DFaceRecon_pytorch/9167c7136aebef0c54d3ac74f4b0396222a491c1/BFM/BFM_front_idx.mat \
  --output assets/external/flame_inputs/BFM_front_idx.mat
curl --fail --location https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip \
  --output assets/external/flame_inputs/FLAME_masks.zip
unzip -j assets/external/flame_inputs/FLAME_masks.zip '*FLAME_masks.pkl' \
  -d assets/external/flame_inputs
python - <<'PY'
from pathlib import Path
import numpy as np
from scipy.io import loadmat
p = Path("assets/external/flame_inputs")
idx = loadmat(p / "BFM_front_idx.mat")["idx"]
np.save(p / "indices_53215_35709.npy", idx.astype(np.int32) - 1)
PY
python -m bfsm_applications.build_models \
  --correspondence assets/external/flame_inputs/BFM_to_FLAME_corr.npz \
  --crop-indices assets/external/flame_inputs/indices_53215_35709.npy \
  --anatomical-masks assets/external/flame_inputs/FLAME_masks.pkl \
  --fsmm assets/models/fsmm_200.npz --tmm assets/models/tmm_200.npz \
  --output assets/models
```

The builder adds `flame_fsmm_200.npz` and `flame_tmm_200.npz`; it does not replace the two original models. Review the [correspondence provider notice](https://github.com/TimoBolkart/BFM_to_FLAME) and [FLAME terms](https://flame.is.tue.mpg.de/modellicense.html) before obtaining or sharing inputs/derivatives.

### Neutral FLAME2020

External FLAME fitting and DECA additionally need the original neutral FLAME2020 model. Register at [FLAME](https://flame.is.tue.mpg.de/), accept its terms and extract `generic_model.pkl` into `assets/external/flame/`. The reference file SHA-256 is `efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b`. Prepare its NPZ locally:

```bash
python -m bfsm_applications.prepare_flame \
  --model assets/external/flame/generic_model.pkl \
  --output assets/external/flame/flame_neutral.npz
```

Use the neutral FLAME2020 model required by DECA. The original model and derived files remain subject to the [FLAME academic model license](https://flame.is.tue.mpg.de/modellicense.html).

### DECA source and checkpoint

For the DECA frontend only, obtain [the original DECA source and model](https://github.com/yfeng95/DECA) under [its own terms](https://github.com/yfeng95/DECA/blob/master/LICENSE). Clone the source locally:

```bash
git clone https://github.com/yfeng95/DECA.git assets/external/DECA
git -C assets/external/DECA checkout a11554ae2a2b0f3998cf1fa94dd4db03babb34a2
```

Download `deca_model.tar` through the provider's [model registration/download page](https://deca.is.tue.mpg.de/) and place it at `assets/external/DECA/data/deca_model.tar`. Keep `decalib/__init__.py`, `decalib/models/encoders.py` and `decalib/models/resnet.py` from that source. The reference checkpoint SHA-256 is `e714ed293054cba5eea9c96bd3b6b57880074cd84b3fd00d606cbaf0bee7c5c2`.

Pass `--deca-root assets/external/DECA` when invoking the frontend. Its automatic detection also requires the 2D detector cache above. The frontend uses the coarse encoder and FLAME identity, expression and skinning arrays prepared by the command above. Image overlays use the predicted global/jaw pose, weak-perspective camera and crop transform.

## Geometry model formats

| File | Representation |
| --- | --- |
| `fsmm_200.npz` | 35709 face and 23207 skull vertices; 200 shared geometry coefficients |
| `tmm_200.npz` | 76393 paired skull-minus-face vectors; 200 independent tissue coefficients |
| `flame_fsmm_200.npz` | 1657 facial FLAME-topology vertices and 23207 skull vertices; 200 shared coefficients |
| `flame_tmm_200.npz` | 1088 directly supported tissue vertices; 200 coefficients and weighted skull anchors; complete fitted display uses 1657 facial vertices |

Load NPZ files with `allow_pickle=False`. `shapeMU` stores the mean and `shapeBase` stores principal components scaled by their standard deviations. Linear decoding is `shapeMU + shapeBase @ alpha`, before the fitter's recentering and pose transforms. Each basis column describes a population variation direction. FSMM face and skull share one coefficient vector; TMM uses a separate coefficient vector.

The supplied FSMM and TMM contain geometry only. The skull uses 23,207 vertices and 44,940 triangles. FSMM files store `recenter_offset` to retain the shared model frame; use it rather than recomputing a center from the changed topology. Keep each model with its matching templates, pair indices and region labels.

To construct a model on your own topology-consistent face/skull pairs, follow [PCA construction](../modules/01_topology_and_modeling/docs/PCA_CONSTRUCTION.md) and choose a separate output directory for the resulting models.

## Small region labels and generated viewers

`modules/02_morphable_models/src/bfsm_models/skull_regions.npz` ships with the source. It contains four boolean labels, a vertex count and a topology fingerprint, not skull coordinates. It is optional for region visualization and is not required by the generic editor or predictor.

`PLOTLY_LICENSE.txt` is the Plotly copyright and MIT license notice accompanying the HTML viewers; it is not model data.

Generate `fsmm_interactive.html`, `tmm_interactive.html` and their Plotly notice with the [module 2 model-mode command](../modules/02_morphable_models/README.md#principal-components). The HTML files contain population-mode geometry and embedded Plotly JavaScript and open directly in a browser. Their reference PNG/PDF figures are under `data/output/reference/02_models/modes/`; new generation writes to your chosen output directory.
