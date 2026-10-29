# Reproduce the example skull edit

These independent pieces belong to the photograph-derived example skull. Their preparation used the supplied template upper/lower region references to keep all tooth surfaces with their corresponding jaw, followed by geometric cuts and a separate chin piece. They are not a measured CT or another patient's template geometry.

The generic editor reads these OBJ pieces and explicit rigid transforms. It does not load `skull_mask_all.npy` or require a segmentation file. Other institutions can supply their own independently prepared pieces or use an external 3D editor to create a Plan.

From the repository root:

```bash
python -m bfsm_postoperative.plan_edit --config data/input/image_clinical/plan_edit/edit.toml --output data/output/runs/Plan.obj
```

`edit.toml` specifies piece order, axes, pivots, angles and translations in canonical model units. `chin_bridge.npz` contains only the 262 triangle indices joining the lower/chin cut boundaries in that piece order. This optional bridge is an explicitly artificial demonstration surface; it is not recovered or observed bone. It leaves the positioned bone vertices unchanged.

The supplied `../Plan.obj` is the selected input to the postoperative demonstration. Its arbitrary edited topology is registered back to the fixed BFSM skull order before the original tissue transport and fitting. See the [postoperative workflow](../../../../modules/04_postoperative/README.md) for the registered Plan and prediction outputs.
