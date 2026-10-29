from __future__ import annotations

import json
import tomllib
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from .nicp_core import io3d
from .nicp_core.nicp import non_rigid_icp_mesh2mesh
from .nicp_core.utils import normalize_mesh
from .paths import BFM_LDM68_PATH, DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_NICP_CONFIG, DEFAULT_SKULL_TEMPLATE_PATH
from .texture_transfer import transfer_final_face_texture


def find_nearest_indices(a, b):
    tree = cKDTree(a)
    _, indices = tree.query(b, k=1)
    return indices


def _load_config(config_path: str | Path | None):
    path = Path(config_path or DEFAULT_NICP_CONFIG)
    return (tomllib.loads(path.read_text()) if path.suffix == ".toml"
            else json.loads(path.read_text()))


def _restore_nicp_scale(registered_mesh, norm_param):
    offset, scale = norm_param
    offset = -offset
    scale = 1.0 / scale
    offset = offset[0:1, :].repeat(registered_mesh.verts_padded().shape[1], 1)
    registered_mesh.scale_verts_(scale)
    return registered_mesh.offset_verts(offset)


def run_face_nicp(
    stage_dir: str | Path,
    source_color_stage_dir: str | Path,
    config_path: str | Path | None = None,
    device: str = "cuda:0",
) -> None:
    stage_dir = Path(stage_dir)
    mesh_target = trimesh.load(stage_dir / "face_scan_topology_points.obj", process=False, maintain_order=True)
    mesh_source = trimesh.load(stage_dir / "render_mesh.obj", process=False, maintain_order=True)
    v_target = mesh_target.vertices
    v_source = mesh_source.vertices
    bad = np.array([-1, -1, -1])
    mask_all = np.all(v_target == bad.copy(), axis=1) | np.all(v_source == bad.copy(), axis=1)
    v_target = v_target[~mask_all]
    v_source = v_source[~mask_all]

    template = trimesh.load(stage_dir / "render_mesh.obj", process=False, maintain_order=True)
    target = trimesh.load(stage_dir / "face_scan.obj", process=False, maintain_order=True)
    template_lm_index = torch.from_numpy(find_nearest_indices(template.vertices, v_source)).unsqueeze(0)
    target_lm_index = torch.from_numpy(find_nearest_indices(target.vertices, v_target)).unsqueeze(0)

    meshes = io3d.load_obj_as_mesh(stage_dir / "face_scan.obj", device=torch.device(device))
    initial_mesh = io3d.load_obj_as_mesh(stage_dir / "render_mesh.obj", device=torch.device(device))
    with torch.no_grad():
        norm_meshes, norm_param = normalize_mesh(meshes)
        initial_norm_meshes, _ = normalize_mesh(initial_mesh)
    registered_mesh = non_rigid_icp_mesh2mesh(
        initial_norm_meshes,
        norm_meshes,
        template_lm_index,
        target_lm_index,
        _load_config(config_path),
        device=torch.device(device),
    )
    registered_mesh = _restore_nicp_scale(registered_mesh, norm_param)
    io3d.save_meshes_as_objs([stage_dir / "face_nicp.obj"], registered_mesh, save_textures=False)
    transfer_final_face_texture(stage_dir, source_color_stage_dir)


def _get3p(pts68):
    up = (pts68[19] + pts68[24]) / 2
    middle = (pts68[0] + pts68[16]) / 2
    down = pts68[8]
    lz = pts68[8, 2] - middle[2]
    z = pts68[8, 2] - 1.35 * lz
    up[2] = z
    middle[2] = z
    down[2] = z
    return [up, middle, down]


@lru_cache(maxsize=1)
def _landmark_indices():
    return np.load(BFM_LDM68_PATH)


def _get_segment_intersections(a, b, mesh, batch_size: int = 8192):
    a = np.asarray(a)
    b = np.asarray(b).reshape(1, 3)
    ray_origins = a
    ray_directions = b - a
    ray_lengths = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    ray_directions_normalized = ray_directions / ray_lengths
    intersections = np.full_like(a, fill_value=-1.0)

    for start in range(0, len(ray_origins), batch_size):
        stop = min(start + batch_size, len(ray_origins))
        index_tri, index_ray, locations = mesh.ray.intersects_id(
            ray_origins=ray_origins[start:stop],
            ray_directions=ray_directions_normalized[start:stop],
            return_locations=True,
            multiple_hits=False,
        )
        if len(index_tri) == 0:
            continue
        global_index = start + index_ray
        dists = np.linalg.norm(locations - ray_origins[global_index], axis=1)
        valid = dists <= (ray_lengths[global_index, 0] + 1e-6)
        intersections[global_index[valid]] = locations[valid]

    return intersections


def _skull_control_points(face_file: str | Path, skull_file: str | Path):
    face_mesh = trimesh.load(face_file, process=False, maintain_order=True)
    skull_mesh = trimesh.load(skull_file, process=False, maintain_order=True)
    p68 = face_mesh.vertices[_landmark_indices(), :].copy()
    end_list = _get3p(p68)
    a = face_mesh.vertices.copy()
    if a.shape[0] != 35709:
        raise ValueError(f"topology face must have 35709 vertices, got {a.shape[0]} in {face_file}")
    return [_get_segment_intersections(a, end_p, skull_mesh) for end_p in end_list]


@lru_cache(maxsize=1)
def _default_template_controls():
    return _skull_control_points(DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH)


def make_skull_control_points(stage_dir: str | Path, template: bool = False) -> None:
    stage_dir = Path(stage_dir)
    if template:
        export_template_controls(stage_dir)
        return
    face_file = stage_dir / "topology_face.obj"
    skull_file = stage_dir / "skull.obj"
    intersections = _skull_control_points(face_file, skull_file)
    for name, points in zip(("up", "middle", "down"), intersections):
        np.save(stage_dir / f"{name}_skull.npy", points)


def export_template_controls(
    output_dir: str | Path,
    face_template: str | Path = DEFAULT_FACE_TEMPLATE_PATH,
    skull_template: str | Path = DEFAULT_SKULL_TEMPLATE_PATH,
) -> None:
    output_dir = Path(output_dir)
    from bfsm_core.paths import check_output_inputs
    destinations = [f"{name}_skull_template.npy" for name in ("up", "middle", "down")]
    if Path(skull_template).resolve() != (output_dir / "skull_template.obj").resolve():
        destinations.append("skull_template.obj")
    check_output_inputs(output_dir, destinations, [face_template, skull_template])
    output_dir.mkdir(parents=True, exist_ok=True)
    intersections = _skull_control_points(face_template, skull_template)
    skull_template = Path(skull_template)
    skull_template_out = output_dir / "skull_template.obj"
    if skull_template.resolve() != skull_template_out.resolve():
        shutil.copy2(skull_template, skull_template_out)
    for name, points in zip(("up", "middle", "down"), intersections):
        np.save(output_dir / f"{name}_skull_template.npy", points)


def _load_template_controls(template_dir: str | Path | None):
    if template_dir is None:
        return DEFAULT_SKULL_TEMPLATE_PATH, _default_template_controls()

    template_dir = Path(template_dir)
    template_mesh_path = template_dir / "skull_template.obj"
    control_paths = [
        template_dir / "up_skull_template.npy",
        template_dir / "middle_skull_template.npy",
        template_dir / "down_skull_template.npy",
    ]
    missing = [p for p in (template_mesh_path, *control_paths) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing skull template assets: " + ", ".join(str(p) for p in missing))
    return template_mesh_path, [np.load(path) for path in control_paths]


def run_skull_nicp(
    stage_dir: str | Path,
    template_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "cuda:0",
) -> None:
    stage_dir = Path(stage_dir)
    template_mesh_path, source_controls = _load_template_controls(template_dir)
    bad = np.array([-1, -1, -1])
    v_source = np.concatenate(source_controls, axis=0)
    v_target = np.concatenate(
        [
            np.load(stage_dir / "up_skull.npy"),
            np.load(stage_dir / "middle_skull.npy"),
            np.load(stage_dir / "down_skull.npy"),
        ],
        axis=0,
    )
    mask_all = np.all(v_target == bad.copy(), axis=1) | np.all(v_source == bad.copy(), axis=1)
    v_target = v_target[~mask_all]
    v_source = v_source[~mask_all]
    template = trimesh.load(template_mesh_path, process=False, maintain_order=True)
    target = trimesh.load(stage_dir / "skull.obj", process=False, maintain_order=True)
    template_lm_index = torch.from_numpy(find_nearest_indices(template.vertices, v_source)).unsqueeze(0)
    target_lm_index = torch.from_numpy(find_nearest_indices(target.vertices, v_target)).unsqueeze(0)

    meshes = io3d.load_obj_as_mesh(stage_dir / "skull.obj", device=torch.device(device))
    initial_mesh = io3d.load_obj_as_mesh(template_mesh_path, device=torch.device(device))
    with torch.no_grad():
        norm_meshes, norm_param = normalize_mesh(meshes)
        initial_norm_meshes, _ = normalize_mesh(initial_mesh)
    registered_mesh = non_rigid_icp_mesh2mesh(
        initial_norm_meshes,
        norm_meshes,
        template_lm_index,
        target_lm_index,
        _load_config(config_path),
        device=torch.device(device),
    )
    registered_mesh = _restore_nicp_scale(registered_mesh, norm_param)
    io3d.save_meshes_as_objs([stage_dir / "topology_skull.obj"], registered_mesh, save_textures=False)
