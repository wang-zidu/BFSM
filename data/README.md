# Inputs, reference results and local runs

The BFSM models and example files are hosted at [Hugging Face](https://huggingface.co/datasets/Zidu-Wang/BFSM). Download them into the repository root using the command in [asset setup](../assets/README.md#1-download-the-bfsm-bundle); it installs this `data/` directory beside `assets/`.

Inputs and generated geometry are separated by role:

```text
data/
  input/
    sample_001/                 # original aligned scanner/CT surfaces and known 3D correspondences
    image_clinical/             # published photograph, canonical edited Plan and independent pieces
    image_portrait/             # public-domain Newton portrait
  output/
    reference/
      01_topology/              # scanner and CT-only registration
      02_models/                # population modes and fixed-skull tissue variations
      03_image/                 # image reconstruction
      04_postoperative/         # scan and direct-FSMM photograph predictions
    runs/                      # ignored outputs of new local executions
```

| Input / operation | Guide | Reference location |
| --- | --- | --- |
| Original sample_001 scanner/CT surfaces | [sample_001](input/sample_001/README.md) | `data/output/reference/01_topology/sample_001/` |
| FSMM/TMM population modes and tissue variation | [models](../modules/02_morphable_models/README.md) | `data/output/reference/02_models/` |
| Published HFM photograph | [image 1](input/image_clinical/1.png) | `data/output/reference/03_image/image_clinical/` |
| Public-domain Newton portrait | [image 2](input/image_portrait/2.jpg) | `data/output/reference/03_image/image_portrait/` |
| Postoperative software predictions | [module4](../modules/04_postoperative/README.md) | `data/output/reference/04_postoperative/` |
| Facial FLAME applications | [module5](../modules/05_applications/README.md) | Generated locally in `data/output/runs/` |

Workflow configurations live under their owning module. The scan example uses `modules/04_postoperative/configs/scan_demo.toml`; the photograph example uses `image_demo.toml`, which binds its supplied Plan to the example photograph, model and reconstructed canonical pair. For another photograph, provide its matching `--plan` or configuration. Numerical arrays use NPY/NPZ; configurations use TOML. New runs write to `data/output/runs/` and do not overwrite or implicitly reuse reference outputs.

The supplied edited `data/input/image_clinical/Plan.obj` is an input. Its NICP-fitted `topology_Plan.obj` and predicted faces are outputs. Other simple scan edits are generated during execution and stay in that run's output directory. Model weights and provider assets are installed separately under [assets](../assets/README.md).
