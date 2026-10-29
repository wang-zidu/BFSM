from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from bfsm_core.paths import ROOT
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.environ.get("VT_REGISTRATION_ASSETS", ROOT / "assets/external/registration")).expanduser().resolve()
BFM_MODEL_PATH = ASSET_ROOT / "BFM" / "BFM09_model_info.mat"
BFM_LDM68_PATH = ASSET_ROOT / "BFM" / "ldm68.npy"
TEMPLATE_ROOT = ASSET_ROOT / "template"
DEFAULT_FACE_TEMPLATE_PATH = TEMPLATE_ROOT / "muFACE.obj"
DEFAULT_SKULL_TEMPLATE_PATH = TEMPLATE_ROOT / "muSKULL.obj"
DEFAULT_NICP_CONFIG = ASSET_ROOT / "config" / "fine_grain.toml"


def check_registration_output(raw, output, *, nicp_config=None, skull_template_dir=None):
    """Reject stage/world exports that alias original surfaces or runtime assets."""
    from bfsm_core.paths import check_output_inputs
    raw = Path(raw).resolve()
    output_path = Path(output).resolve()
    # Retain the legacy raw/output layout, but reject redirected stage folders.
    if output_path != raw and raw not in output_path.parents:
        for stage in ("standardized", "posed", "world"):
            target = (output_path / stage).resolve()
            if target == raw or raw in target.parents:
                raise ValueError("Registration stage output must not redirect into the raw input directory")
    inputs = [raw / name for name in ("face_scan.obj", "face_scan.mtl", "face_scan.png", "face.obj", "skull.obj")]
    inputs.extend((BFM_MODEL_PATH, BFM_LDM68_PATH, DEFAULT_FACE_TEMPLATE_PATH,
                   DEFAULT_SKULL_TEMPLATE_PATH, Path(nicp_config) if nicp_config else DEFAULT_NICP_CONFIG))
    if skull_template_dir is not None:
        inputs.extend(Path(skull_template_dir) / name for name in
                      ('skull_template.obj', 'up_skull_template.npy',
                       'middle_skull_template.npy', 'down_skull_template.npy'))
    torch_home = Path(os.environ.get('TORCH_HOME',
        Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'torch')).expanduser()
    inputs.extend(torch_home / 'hub/checkpoints' / name for name in
                  ('3DFAN4-4a694010b9.zip', 'depth-6c4283c0e0.zip', 's3fd-619a316812.pth'))
    standardized = ("face_scan.obj", "face_scan.mtl", "face_scan.png", "face.obj", "skull.obj", "scale.npy",
                    "render.png", "render_depth.npy", "render_depth_mask.npy", "render_coeffs.npy",
                    "render_mesh.obj", "render_composed_img.jpg")
    posed = ("face_scan.obj", "face_scan.mtl", "face_scan.png", "face.obj", "skull.obj", "render_mesh.obj",
             "face_scan_topology_points.obj", "face_nicp.obj", "topology_face.obj", "topology_skull.obj",
             "up_skull.npy", "middle_skull.npy", "down_skull.npy")
    outputs = [f"standardized/{name}" for name in standardized]
    outputs.extend(f"posed/{name}" for name in posed)
    outputs.extend(("coordinate_transform.npz", "world/topology_face.obj", "world/topology_skull.obj"))
    check_output_inputs(output, outputs, inputs)


@dataclass(frozen=True)
class SamplePaths:
    root: Path
    output_dir: Path | None = None

    @property
    def scan_obj(self) -> Path:
        return self.root / "face_scan.obj"

    @property
    def scan_mtl(self) -> Path:
        return self.root / "face_scan.mtl"

    @property
    def scan_png(self) -> Path:
        return self.root / "face_scan.png"

    @property
    def face_obj(self) -> Path:
        return self.root / "face.obj"

    @property
    def skull_obj(self) -> Path:
        return self.root / "skull.obj"

    @property
    def output(self) -> Path:
        return self.output_dir if self.output_dir is not None else self.root / "output"

    def stage(self, name: str) -> Path:
        return self.output / name

    @property
    def standardized(self) -> Path:
        return self.stage("standardized")

    @property
    def posed(self) -> Path:
        return self.stage("posed")


def iter_samples(data_root: str | Path, include: Iterable[str] | None = None) -> list[SamplePaths]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    include_set = set(include or [])
    if any((root / name).exists() for name in ("face_scan.obj", "face.obj", "skull.obj")):
        samples = [root]
    else:
        samples = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if include_set:
        samples = [p for p in samples if p.name in include_set]
        missing = include_set - {p.name for p in samples}
        if missing:
            raise ValueError("requested subjects not found: " + ", ".join(sorted(missing)))
    if not samples:
        raise ValueError(f"no subjects found under {root}")
    return [SamplePaths(p) for p in samples]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
