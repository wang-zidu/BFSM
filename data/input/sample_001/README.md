# Original sample_001 face/skull example

This is the supplied sample_001 example. Its original surfaces and texture are included for reproducing registration and full-surface postoperative demonstration.

## Input

| File | Role |
| --- | --- |
| `data/input/sample_001/raw/face_scan.obj`, `face_scan.mtl`, `face_scan.png` | Original textured 3D face scan: 64513 vertices, 128493 triangles |
| `data/input/sample_001/raw/face.obj` | Original CT face: 45741 vertices, 90696 triangles |
| `data/input/sample_001/raw/skull.obj` | Original CT skull: 54781 vertices, 106724 triangles |
| `data/input/sample_001/ct_only/landmarks3d.toml` | 68 XYZ correspondences on that original CT face, in CT-world coordinates |
| `modules/04_postoperative/configs/scan_demo.toml` | Synthetic mandibular editing plan for the scan example |

The three original surfaces are already aligned. In the textured-scan route, scanner RGB/depth and the scanner surface drive face fitting/NICP; the CT face is transformed with the other inputs but is not its fitting target. The CT-only route instead uses the original CT face and skull. Its 68 XYZ annotations were selected directly on CT geometry with Agent assistance and multiview review. See the [manual annotation guide](../../../modules/01_topology_and_modeling/docs/CT_ONLY.md#annotating-the-68-landmarks) and review the brow, lip and cropped lower-chin correspondences before use.

## Run from original inputs

```bash
python -m bfsm_topology.demo --input data/input/sample_001/raw \
  --output data/output/runs/sample_001/registration --device cuda:0 --deterministic --seed 0
python -m bfsm_topology.preprocess_pipeline.ct_only \
  --face data/input/sample_001/raw/face.obj \
  --skull data/input/sample_001/raw/skull.obj \
  --landmarks3d data/input/sample_001/ct_only/landmarks3d.toml \
  --output-dir data/output/runs/ct_only --device cuda:0
python -m bfsm_postoperative.demo scan --example data/input/sample_001 \
  --output data/output/runs/postoperative_scan --device cuda:0 --deterministic --seed 0
```

Follow [the registration guide](../../../modules/01_topology_and_modeling/README.md) and [postoperative guide](../../../modules/04_postoperative/README.md) for required assets, environments and stages. Default commands generate results in a new destination and do not read the bundled reference predictions.

## Reference output

- `data/output/reference/01_topology/sample_001/registration/posed/topology_face.obj` and `topology_skull.obj`: corresponding 35709/23207-vertex meshes actually generated from these original scanner/CT inputs. They are outputs, not substitutes for original input scans. `coordinate_transform.npz` provides the common original-world/posed transform.
- `data/output/reference/01_topology/sample_001/ct_only/`: corresponding world/canonical meshes from the original CT face/skull, with fitted parameters and world transform in `fit_parameters.npz`.
- `data/output/reference/02_models/sample_001/tissue/`: four tissue variations around the registered fixed skull; `id1/id2` refer to sampled tissue coefficients, not other patients.
- `data/output/reference/04_postoperative/sample_001/topology/`: same-topology edited skulls, fitted faces, edit support and coefficients.
- `data/output/reference/04_postoperative/sample_001/world/`: complete ARAP-predicted textured faces and corresponding edited model-topology skulls, returned to the original coordinate frame.

![Original scanner input and registered surfaces](../../output/reference/01_topology/sample_001/registration/preview/comparison.png)

![Original CT surfaces and registered surfaces](../../output/reference/01_topology/sample_001/ct_only/preview/comparison.png)

The [HFM photograph-only demonstration](../image_clinical/1.png) is a separate publicly sourced example without CT. Synthetic edits are not observed surgery outcomes. Configuration uses TOML and numerical arrays use NPZ.
