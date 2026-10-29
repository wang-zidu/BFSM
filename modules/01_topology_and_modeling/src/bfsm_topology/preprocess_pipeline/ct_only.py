"""CT surfaces + explicitly corresponding 3D landmarks; independent geometry-only route."""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Real
import tomllib
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mesh(path):
    import trimesh
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh", skip_materials=True)
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh must have finite vertices and triangles: {path}")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError(f"Invalid mesh vertex indices: {path}")
    return mesh


def landmark_errors(vertices, xyz, indices):
    """Report semantic landmark error separately from surface proximity."""
    residual = np.asarray(vertices)[np.asarray(indices)] - np.asarray(xyz)
    distance = np.linalg.norm(residual, axis=1)
    return {"rms": float(np.sqrt(np.mean(distance ** 2))),
            "maximum": float(distance.max())}


def read_inputs(face_path, skull_path, landmark_path):
    """Validate actual XYZ landmarks in the same world frame as both CT meshes."""
    from scipy.spatial import cKDTree
    face, skull = _mesh(face_path), _mesh(skull_path)
    landmark_path = Path(landmark_path)
    if not landmark_path.is_file():
        raise FileNotFoundError(landmark_path)
    annotation = (tomllib.loads(landmark_path.read_text()) if landmark_path.suffix == ".toml"
                  else json.loads(landmark_path.read_text()))
    if annotation.get("schema_version") != 1 or annotation.get("coordinate_space") != "ct_mesh_world":
        raise ValueError("Landmarks require schema_version=1 and coordinate_space=ct_mesh_world")
    if not isinstance(annotation.get("units"), str) or not annotation["units"].strip():
        raise ValueError("Landmarks must declare the units shared by both meshes")
    points = np.asarray(annotation.get("points"), dtype=float)
    indices = np.asarray(annotation.get("template_landmark_indices"))
    if points.shape != (68, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 68 finite XYZ points; 2D image landmarks are not accepted")
    if indices.shape != (68,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Expected 68 ordered zero-based template_landmark_indices")
    if len(np.unique(points, axis=0)) < 4 or np.linalg.matrix_rank(points-points.mean(0)) < 2:
        raise ValueError("3D landmarks must contain noncollinear distinct points")
    spread = float(np.ptp(points, axis=0).max())
    if spread <= 0:
        raise ValueError("Landmark spatial extent is zero")
    distances, target_indices = cKDTree(face.vertices).query(points)
    if distances.max() > 0.02 * spread:
        raise ValueError("3D landmarks are not near the CT face surface; check coordinate frame/units")
    ratio = np.ptp(skull.vertices, axis=0).max()/np.ptp(face.vertices, axis=0).max()
    if not 0.1 < ratio < 10:
        raise ValueError("CT face/skull extents suggest inconsistent coordinate units")
    return face, skull, points, indices.astype(np.int64), target_indices, annotation, distances


def similarity(source, target):
    """Row-vector, proper similarity: world = scale * canonical @ rotation + translation."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    x, y = source-source.mean(0), target-target.mean(0)
    u, singular, vh = np.linalg.svd(x.T @ y / len(x))
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vh))
    rotation = (u * signs) @ vh
    scale = float((singular*signs).sum() / np.square(x).sum(axis=1).mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot estimate a positive CT-to-template landmark similarity")
    translation = target.mean(0)-scale*source.mean(0)@rotation
    return scale, rotation, translation


def _save_mesh(path, vertices, faces):
    from pytorch3d.io import save_obj
    import torch
    save_obj(str(path), torch.as_tensor(vertices, dtype=torch.float32, device="cpu"),
             torch.as_tensor(faces, dtype=torch.int64, device="cpu"))


def _validate_options(args):
    for name, minimum in (('coarse_steps', 1), ('depth_steps', 1), ('depth_size', 32)):
        value = getattr(args, name)
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name in ('learning_rate', 'landmark_weight', 'depth_weight',
                 'id_regularization', 'expression_regularization'):
        value = getattr(args, name)
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')


def run(args):
    _validate_options(args)
    # Surface/annotation validation precedes CUDA setup or optional asset imports.
    face, skull, xyz, indices, target_indices, annotation, snapping = read_inputs(
        args.face, args.skull, args.landmarks3d)
    from .paths import BFM_LDM68_PATH, BFM_MODEL_PATH, DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH, DEFAULT_NICP_CONFIG
    required = [BFM_MODEL_PATH, BFM_LDM68_PATH, DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH,
                Path(args.nicp_config) if args.nicp_config else DEFAULT_NICP_CONFIG]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required external registration asset: {path}")
    expected_indices = np.load(BFM_LDM68_PATH).reshape(-1)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Landmark semantic order must match the external BFM ldm68.npy indices exactly")
    from bfsm_core.paths import check_output_inputs
    inputs = [args.face, args.skull, args.landmarks3d, *required]
    if args.skull_template_dir:
        inputs.extend(Path(args.skull_template_dir) / name for name in
                      ("skull_template.obj", "up_skull_template.npy", "middle_skull_template.npy", "down_skull_template.npy"))
    outputs = ["fit_parameters.npz", "topology_face.obj", "topology_skull.obj",
               "canonical/face_target.obj", "canonical/skull.obj", "canonical/landmarks3d.npy",
               "canonical/fitted_face.obj", "canonical/topology_face.obj", "canonical/topology_skull.obj"]
    outputs.extend(f"canonical/{name}_skull.npy" for name in ("up", "middle", "down"))
    check_output_inputs(args.output_dir, outputs, inputs)
    import random, torch
    if not args.native:
        from .reproducibility import configure
        configure(args.device, args.seed)
    else:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from .core import get_recon_model
    from .rendering import get_depth
    from .nicp_core import io3d
    from .nicp_core.utils import normalize_mesh
    from .nicp_core.nicp import non_rigid_icp_mesh2mesh
    from .topology import _restore_nicp_scale, make_skull_control_points, run_skull_nicp, _load_template_controls
    from pytorch3d.structures import Meshes
    from tqdm import trange
    import sys, time

    start = time.time()
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    canonical = out/"canonical"; canonical.mkdir(exist_ok=True)
    model = get_recon_model(device=device)
    neutral = model.get_vs(model.get_id_tensor(), model.get_exp_tensor()).detach().cpu().numpy()[0]
    scale, rotation, translation = similarity(neutral[indices], xyz)
    to_canonical = lambda vertices: (vertices-translation)@rotation.T/scale
    face_v, skull_v, landmarks = map(to_canonical, [face.vertices, skull.vertices, xyz])
    _save_mesh(canonical/"face_target.obj", face_v, face.faces)
    _save_mesh(canonical/"skull.obj", skull_v, skull.faces)
    np.save(canonical/"landmarks3d.npy", landmarks)
    transform = {"equation": "world = scale * canonical @ rotation + translation",
                 "scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
                 "units": annotation["units"], "coordinate_space": "ct_mesh_world"}
    index_tensor = torch.as_tensor(indices, dtype=torch.int64, device=device)
    lm = torch.as_tensor(landmarks[None], dtype=torch.float32, device=device)
    delta_angle = torch.zeros((1, 3), device=device, requires_grad=True)
    translation_delta = torch.zeros((1, 3), device=device, requires_grad=True)
    log_scale = torch.zeros((), device=device, requires_grad=True)
    parameters = [model.get_id_tensor(), model.get_exp_tensor(), delta_angle, translation_delta, log_scale]

    def vertices():
        local = model.get_vs(model.get_id_tensor(), model.get_exp_tensor())
        return log_scale.exp()*(local@model.compute_rotation_matrix(delta_angle))+translation_delta[:, None]

    def objective(v):
        landmark = (v[:, index_tensor]-lm).square().sum(-1).mean()
        prior = args.id_regularization*model.get_id_tensor().square().sum()
        prior = prior+args.expression_regularization*model.get_exp_tensor().square().sum()
        return args.landmark_weight*landmark+prior, landmark

    history = {}
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    for _ in trange(args.coarse_steps, desc="CT 3D landmarks"):
        optimizer.zero_grad(); loss, landmark = objective(vertices())
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite CT landmark objective; check scale, weights and learning rate')
        loss.backward(); optimizer.step()
    history["coarse_landmark_rmse_canonical"] = float(landmark.detach().sqrt())
    # No images, FAN, texture, Phong shading, or photometric objective enter either fit phase.
    target = Meshes([torch.as_tensor(face_v, dtype=torch.float32, device=device)],
                    [torch.as_tensor(face.faces, dtype=torch.int64, device=device)])
    with torch.no_grad():
        target_depth, target_mask = get_depth(target, device=str(device), image_size=args.depth_size)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate*0.1)
    for _ in trange(args.depth_steps, desc="CT geometric depth"):
        optimizer.zero_grad(); v = vertices(); base, landmark = objective(v)
        if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(base)):
            raise ValueError('Non-finite CT fitted geometry or objective; check scale, weights and learning rate')
        prediction = Meshes(verts=v, faces=model.tri[None])
        predicted_depth, predicted_mask = get_depth(prediction, device=str(device), image_size=args.depth_size)
        valid = (target_mask>0)&(predicted_mask>0)
        if not bool(valid.any()):
            raise ValueError("No overlapping frontal CT/model depth; check 3D landmark semantics")
        depth_loss = (predicted_depth[valid]-target_depth[valid]).square().mean()
        loss = base+args.depth_weight*depth_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite CT depth objective; check scale, weights and learning rate')
        loss.backward(); optimizer.step()
    with torch.no_grad():
        fitted = vertices().detach()
    if not bool(torch.isfinite(fitted).all()):
        raise ValueError('Non-finite final CT geometry; check scale, weights and learning rate')
    history.update(fine_landmark_rmse_canonical=float(((fitted[:, index_tensor]-lm).square().sum(-1).mean()).sqrt()),
                   depth_mse_last_preupdate=float(depth_loss.detach()))
    _save_mesh(canonical/"fitted_face.obj", fitted[0].cpu().numpy(), model.tri.cpu().numpy())
    np.savez(out/"fit_parameters.npz", identity=model.get_id_tensor().detach().cpu().numpy(),
             expression=model.get_exp_tensor().detach().cpu().numpy(), delta_angle=delta_angle.detach().cpu().numpy(),
             translation=translation_delta.detach().cpu().numpy(), log_scale=log_scale.detach().cpu().numpy(),
             world_scale=np.asarray(scale), world_rotation=rotation, world_translation=translation,
             world_units=np.asarray(annotation["units"]), coordinate_space=np.asarray("ct_mesh_world"))
    initial = io3d.load_obj_as_mesh(canonical/"fitted_face.obj", device=device)
    target_mesh = io3d.load_obj_as_mesh(canonical/"face_target.obj", device=device)
    with torch.no_grad():
        target_normalized, norm = normalize_mesh(target_mesh)
        initial_normalized, _ = normalize_mesh(initial)
    config = (tomllib.loads(required[-1].read_text()) if required[-1].suffix == ".toml"
              else json.loads(required[-1].read_text()))
    registered = non_rigid_icp_mesh2mesh(initial_normalized, target_normalized,
        index_tensor[None], torch.as_tensor(target_indices[None], dtype=torch.int64, device=device), config, device=device)
    registered = _restore_nicp_scale(registered, norm)
    io3d.save_meshes_as_objs([canonical/"topology_face.obj"], registered, save_textures=False)
    make_skull_control_points(canonical)
    _, source_controls = _load_template_controls(args.skull_template_dir)
    target_controls = [np.load(canonical/f"{name}_skull.npy") for name in ["up", "middle", "down"]]
    source_points, target_points = np.concatenate(source_controls), np.concatenate(target_controls)
    good = ~((source_points == -1).all(1)|(target_points == -1).all(1))
    if good.sum()<3 or np.linalg.matrix_rank(target_points[good]-target_points[good].mean(0))<2:
        raise ValueError("Insufficient noncollinear CT skull ray controls")
    run_skull_nicp(canonical, template_dir=args.skull_template_dir, config_path=args.nicp_config, device=str(device))
    geometry = {}
    for name in ["face", "skull"]:
        result = _mesh(canonical/f"topology_{name}.obj")
        world = scale*result.vertices@rotation+translation
        _save_mesh(out/f"topology_{name}.obj", world, result.faces)
        exported = _mesh(out/f"topology_{name}.obj")
        geometry[name] = {"vertices": len(exported.vertices), "faces": len(exported.faces),
            "finite": bool(np.isfinite(exported.vertices).all()), "zero_area_faces": int((exported.area_faces==0).sum()),
            "sha256": _sha(out/f"topology_{name}.obj")}
    report = {"route": "ct_surfaces_and_corresponding_3d_landmarks", "status": "completed",
        "original_eight_stage_pipeline_modified": False, "coordinate_transform": transform,
        "landmark_provenance": annotation.get("provenance", {}), "seed": args.seed,
        "coarse_steps": args.coarse_steps, "depth_steps": args.depth_steps, "depth_size": args.depth_size,
        "nicp_config": config, "skull_control_pairs": int(good.sum()), "landmark_snap_max_world": float(snapping.max()),
        "photometric_terms": [], "face_alignment_imported": "face_alignment" in sys.modules,
        "history": history, "geometry": geometry,
        "final_landmark_errors_world": {**landmark_errors(
            _mesh(out/"topology_face.obj").vertices, xyz, indices),
            "units": annotation.get("units", "unspecified")},
        "seconds": time.time()-start,
        "inputs": {k: {"path": str(Path(v).resolve()), "sha256": _sha(v)} for k,v in
                   [("face", args.face), ("skull", args.skull), ("landmarks3d", args.landmarks3d)]},
        "assets": {str(p): _sha(p) for p in required}}
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", required=True, help="CT-derived skin/face triangle mesh")
    parser.add_argument("--skull", required=True, help="CT skull in the same coordinate system and units")
    parser.add_argument("--landmarks3d", required=True, help="Corresponding 68-point XYZ TOML/JSON, not photo landmarks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native", action="store_true", help="Disable the default strict deterministic backend")
    parser.add_argument("--coarse-steps", type=int, default=500)
    parser.add_argument("--depth-steps", type=int, default=1000)
    parser.add_argument("--depth-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--landmark-weight", type=float, default=100.)
    parser.add_argument("--depth-weight", type=float, default=30.)
    parser.add_argument("--id-regularization", type=float, default=1e-3)
    parser.add_argument("--expression-regularization", type=float, default=0.8e-3)
    parser.add_argument("--nicp-config")
    parser.add_argument("--skull-template-dir")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
