"""Register raw textured scan/CT surfaces; explicitly validate any reused results."""
import argparse
from bfsm_core.paths import ROOT
import hashlib
from pathlib import Path
import shutil

import numpy as np

from .coordinates import RegistrationTransform


RAW_FILES = ("face_scan.obj", "face_scan.mtl", "face_scan.png", "face.obj", "skull.obj")


def _obj(path):
    vertices, faces = [], []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields and fields[0] == "f":
                faces.append([int(x.split("/")[0]) for x in fields[1:]])
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Expected finite mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 1 or faces.max() > len(vertices):
        raise ValueError(f"Expected valid OBJ triangles: {path}")
    return vertices, faces


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raw_files(raw):
    raw = Path(raw).expanduser().resolve()
    for name in RAW_FILES:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    return raw


def finalize_existing(raw, registration_output):
    """Verify raw→standardized→posed geometry before exporting a world-space pair.

    This is an explicit reuse operation. It validates mesh connectivity, all raw
    vertices, scan texture pixels, and the fitted topology; no optimization runs.
    Existing stage outputs are read only. Only the transform and world exports
    are written under registration_output.
    """
    from PIL import Image
    from bfsm_core.mesh import replace_obj_vertices
    raw = _raw_files(raw)
    out = Path(registration_output).expanduser().resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError("Registration output and raw input directories must not contain one another")
    from .preprocess_pipeline.paths import check_registration_output
    check_registration_output(raw, out)
    standardized, posed = out / "standardized", out / "posed"
    scan_vertices, _ = _obj(raw / "face_scan.obj")
    # Match the original PyTorch3D float32 input/bounding-box arithmetic.
    scan32 = scan_vertices.astype(np.float32)
    low, high = scan32.min(0), scan32.max(0)
    center = (low + high) / np.float32(2)
    expected_scale = np.float32(2) / (high[:2] - low[:2]).max()
    scale = float(np.load(standardized / "scale.npy", allow_pickle=False))
    if not np.isclose(scale, expected_scale, rtol=2e-6, atol=0):
        raise ValueError("Saved scale does not match the raw scanner bounding box")
    coeffs = np.load(standardized / "render_coeffs.npy", allow_pickle=False).reshape(-1)
    if coeffs.shape != (257,) or not np.isfinite(coeffs).all():
        raise ValueError("Expected finite original 257 fitting coefficients")
    from .preprocess_pipeline.pose_normalize import compute_rotation_matrix
    import torch
    rotation = compute_rotation_matrix(torch.tensor(coeffs[224:227], dtype=torch.float32)[None])[0].numpy()
    transform = RegistrationTransform(center, scale, rotation, coeffs[254:257])
    for name in ("face_scan.obj", "face.obj", "skull.obj"):
        original, original_faces = _obj(raw / name)
        normalized, normalized_faces = _obj(standardized / name)
        canonical, canonical_faces = _obj(posed / name)
        if not np.array_equal(original_faces, normalized_faces) or not np.array_equal(original_faces, canonical_faces):
            raise ValueError(f"Raw/registered connectivity mismatch: {name}")
        expected = (original - center) * scale
        if normalized.shape != original.shape or not np.allclose(normalized, expected, atol=1e-5, rtol=0):
            raise ValueError(f"Standardized surface does not correspond to the supplied raw input: {name}")
        # Pose is computed on the serialized standardized vertices in the original code.
        expected = (normalized - transform.translation) @ np.linalg.inv(rotation)
        if canonical.shape != original.shape or not np.allclose(canonical, expected, atol=1e-5, rtol=0):
            raise ValueError(f"Posed surface does not correspond to the supplied raw input: {name}")
    texture = np.asarray(Image.open(raw / "face_scan.png").convert("RGB"))
    for stage in (standardized, posed):
        if not np.array_equal(texture, np.asarray(Image.open(stage / "face_scan.png").convert("RGB"))):
            raise ValueError("Registered scan texture does not match the raw scan")
    from .preprocess_pipeline.paths import DEFAULT_SKULL_TEMPLATE_PATH
    skull_template_vertices, skull_template_faces = _obj(DEFAULT_SKULL_TEMPLATE_PATH)
    _, face_template_faces = _obj(posed / "render_mesh.obj")
    pair = []
    for name, vertex_count, expected_faces in (("face", 35709, face_template_faces),
                                                ("skull", len(skull_template_vertices), skull_template_faces)):
        source = posed / f"topology_{name}.obj"
        vertices, faces = _obj(source)
        if len(vertices) != vertex_count or not np.array_equal(faces, expected_faces):
            raise ValueError(f"Registered {name} does not preserve the required template topology")
        pair.append((source, vertices))
    hashes = np.asarray([_sha(raw / name) for name in RAW_FILES])
    transform.save(out / "coordinate_transform.npz", raw_files=np.asarray(RAW_FILES), raw_sha256=hashes)
    for source, vertices in pair:
        destination = out / "world" / source.name
        replace_obj_vertices(source, destination, transform.to_world(vertices))
    return transform


def run(raw, output, *, device="cuda:0", deterministic=False, seed=0,
        registration_dir=None, stages=None):
    """Run the original eight stages, or explicitly validate/reuse a prior run.

    Outputs are OUTPUT/{standardized,posed,world}; the raw directory is never
    written. Partial stages are for explicit continuation and return no transform
    until all stage outputs are present and a complete validation is requested.
    """
    raw = _raw_files(raw)
    out = Path(output).expanduser().resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError("Registration output and raw input directories must not contain one another")
    from .preprocess_pipeline.paths import check_registration_output
    check_registration_output(raw, out)
    if registration_dir is not None:
        if stages:
            raise ValueError("Choose explicit reuse or stage execution, not both")
        source = Path(registration_dir).expanduser().resolve()
        if source != out:
            if source in out.parents or out in source.parents:
                raise ValueError("Reuse source and output must not contain one another")
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"Refusing to mix reused registration with existing files: {out}")
            # Copy first: validation must not write into the reused reference run.
            shutil.copytree(source, out, symlinks=False, dirs_exist_ok=True)
        return finalize_existing(raw, out)
    from .preprocess_pipeline.pipeline import PipelineOptions, run_pipeline, DEFAULT_STAGES
    selected = tuple(stages) if stages else None
    run_pipeline(PipelineOptions(data_root=raw, device=device, deterministic=deterministic,
                                 seed=seed, output_dir=out), stages=selected)
    if selected is None or "all" in selected or set(selected) == set(DEFAULT_STAGES):
        return finalize_existing(raw, out)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", "--input", dest="raw", default=str(ROOT / "data/input/sample_001/raw"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used with --deterministic; native mode retains the original RNG behavior")
    parser.add_argument("--stage", action="append", choices=("standardize", "render", "fit3dmm", "pose",
                                                           "texture", "face_nicp", "skull_controls", "skull_nicp", "all"))
    parser.add_argument("--registration-dir", help="explicitly reuse and validate a completed output directory")
    args = parser.parse_args(argv)
    run(args.raw, args.output, device=args.device, deterministic=args.deterministic,
        seed=args.seed, registration_dir=args.registration_dir, stages=args.stage)


if __name__ == "__main__":
    main()
