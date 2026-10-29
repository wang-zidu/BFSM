"""Build registration BFM arrays from separately obtained upstream model inputs."""
from __future__ import annotations

import argparse
from bfsm_core.paths import ROOT
import hashlib
import io
import json
import tomllib
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

SPEC_PATH = Path(__file__).with_name("bfm_asset_spec.toml")
FIELDS = ("meanshape", "meantex", "idBase", "exBase", "texBase", "tri", "point_buf",
          "tri_mask2", "keypoints", "frontmask2_idx", "skinmask")


def _atomic_write(path, data):
    """Replace one output only after a complete write; reject linked destinations."""
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Output must not use symbolic links: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def convert(source):
    """Crop the licensed identity model and expression basis to the fitting topology."""
    source = Path(source)
    identity = loadmat(source / "01_MorphableModel.mat")
    expression_indices = loadmat(source / "BFM_front_idx.mat")["idx"].astype(np.int32).reshape(-1) - 1
    shape_indices = loadmat(source / "BFM_exp_idx.mat")["trimIndex"].astype(np.int32).reshape(-1) - 1
    shape_indices = shape_indices[expression_indices]
    with (source / "Exp_Pca.bin").open("rb") as stream:
        dimensions = np.fromfile(stream, dtype="<i4", count=1)
        if dimensions.tolist() != [79]:
            raise ValueError("Expected the 79-component, 53,215-vertex expression model")
        mean = np.fromfile(stream, dtype="<f4", count=53215 * 3)
        expression = np.fromfile(stream, dtype="<f4", count=53215 * 3 * 79)
        if mean.size != 53215 * 3 or expression.size != 53215 * 3 * 79 or stream.read(1):
            raise ValueError("Unexpected expression binary size")
    expression = expression.reshape(79, -1).T
    standard_deviation = np.loadtxt(source / "std_exp.txt").reshape(1, -1)

    def crop_basis(basis, deviation, indices, count, divisor=1):
        weighted = basis * deviation.reshape(1, -1)
        if divisor != 1:
            weighted = weighted / divisor
        return weighted[:, :count].reshape(-1, 3, count)[indices].reshape(-1, count)

    output = {
        "meanshape": (identity["shapeMU"].reshape(-1, 3) / 1e5)[shape_indices].reshape(1, -1),
        "meantex": identity["texMU"].reshape(-1, 3)[shape_indices].reshape(1, -1),
        "idBase": crop_basis(identity["shapePC"], identity["shapeEV"], shape_indices, 80, 1e5),
        "exBase": crop_basis(expression, standard_deviation, expression_indices, 64, 1e5),
        "texBase": crop_basis(identity["texPC"], identity["texEV"], shape_indices, 80),
    }
    topology = loadmat(source / "facemodel_info.mat")
    output.update((key, topology[key]) for key in FIELDS[5:])
    return output


def verify_arrays(model, spec):
    for name, expected in spec["arrays"].items():
        array = model[name]
        actual = {"shape": list(array.shape), "dtype": str(array.dtype),
                  "sha256": digest(np.ascontiguousarray(array).tobytes())}
        if actual != expected:
            raise ValueError(f"Incompatible model array: {name}; use the documented input versions")


def build(source, output, templates=None):
    spec = tomllib.loads(SPEC_PATH.read_text())
    arrays = convert(source)
    verify_arrays(arrays, spec)
    buffer = io.BytesIO()
    savemat(buffer, {name: arrays[name] for name in FIELDS}, do_compression=False)
    encoded = bytearray(buffer.getvalue())
    # The MAT-v5 descriptive header is fixed for byte-reproducible builds.
    # Its historical date identifies the reference format, not this build's time.
    encoded[:116] = spec["reference_mat_header"].encode("ascii").ljust(116, b"\0")
    if digest(encoded) != spec["outputs"]["BFM/BFM09_model_info.mat"]:
        raise ValueError("Arrays match but MAT serialization differs; use the documented SciPy environment")
    landmarks = (arrays["keypoints"].reshape(-1) - 1).astype(np.int64)
    buffer = io.BytesIO()
    np.save(buffer, landmarks, allow_pickle=False)
    encoded_landmarks = buffer.getvalue()
    if digest(encoded_landmarks) != spec["outputs"]["BFM/ldm68.npy"]:
        raise ValueError("Landmark serialization does not match the reference")

    copied = {}
    if templates is not None:
        templates = Path(templates)
        for name, expected in spec["templates"].items():
            data = (templates / name).read_bytes()
            if digest(data) != expected:
                raise ValueError(f"Wrong BFSM registration template: {name}")
            copied[name] = data
    output = Path(output)
    target = output / "BFM"
    _atomic_write(target / "BFM09_model_info.mat", encoded)
    _atomic_write(target / "ldm68.npy", encoded_landmarks)
    if copied:
        for name, data in copied.items():
            _atomic_write(output / "template" / name, data)
    config = Path(__file__).with_name("preprocess_pipeline") / "assets" / "config"
    for name in ("fine_grain.toml", "coarse_grain.toml"):
        _atomic_write(output / "config" / name, (config / name).read_bytes())
    return {"array_validation": "exact", "bfm_sha256": digest(encoded),
            "landmark_sha256": digest(encoded_landmarks),
            "templates_copied": sorted(copied),
            "complete_registration_assets": bool(copied)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing the six separately licensed/downloaded source files")
    root = ROOT
    parser.add_argument("--output", type=Path, default=root / "assets/external/registration",
                        help="Registration asset root (default: assets/external/registration)")
    parser.add_argument("--templates", type=Path, default=root / "assets/templates",
                        help="Directory containing the released muFACE.obj and muSKULL.obj")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.templates), indent=2))


if __name__ == "__main__":
    main()
