from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes

from .paths import SamplePaths, ensure_dir


def load_obj_as_mesh(path: str | Path, device=None, load_textures: bool = True) -> Meshes:
    return load_objs_as_meshes([str(path)], device=device, load_textures=load_textures)


def save_mesh_as_obj(path: str | Path, mesh: Meshes, save_textures: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.verts_padded()[0]
    faces = mesh.faces_padded()[0]
    if save_textures:
        if mesh.textures.isempty():
            raise ValueError("cannot save untextured mesh with save_textures=True")
        texture_map = mesh.textures.maps_padded()[0]
        verts_uvs = mesh.textures.verts_uvs_padded()[0]
        faces_uvs = mesh.textures.faces_uvs_padded()[0]
        save_obj(
            str(path),
            verts,
            faces,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            texture_map=texture_map,
        )
    else:
        save_obj(str(path), verts, faces)


def standardize_scan_space(in_mesh: Meshes):
    assert in_mesh.verts_padded().shape[0] == 1
    aabb = in_mesh.get_bounding_boxes()
    distance = aabb[:, :2, 1] - aabb[:, :2, 0]
    max_distance, _ = torch.max(distance, dim=1)
    mesh_verts = in_mesh.verts_padded()
    center = torch.zeros((1, mesh_verts.shape[1], 3)).to(in_mesh.device)
    center[0, :, 0] = aabb[:, 0, :].mean()
    center[0, :, 1] = aabb[:, 1, :].mean()
    center[0, :, 2] = aabb[:, 2, :].mean()
    offset = -1 * center.reshape(-1, 3)
    scale = 2 / max_distance
    out_mesh = in_mesh.offset_verts(offset)
    out_mesh.scale_verts_(scale)
    return out_mesh, (offset, scale)


def apply_offset_scale(in_mesh: Meshes, offset: torch.Tensor, scale: torch.Tensor) -> Meshes:
    offset = offset[0:1, :].repeat(in_mesh.verts_padded().shape[1], 1)
    out_mesh = in_mesh.offset_verts(offset)
    out_mesh.scale_verts_(scale)
    return out_mesh


def standardize_sample(sample: SamplePaths) -> None:
    out_dir = ensure_dir(sample.standardized)
    scan = load_obj_as_mesh(sample.scan_obj, load_textures=True, device="cpu")
    scan, trans = standardize_scan_space(scan)
    np.save(out_dir / "scale.npy", trans[1].squeeze().detach().cpu().numpy())
    save_mesh_as_obj(out_dir / "face_scan.obj", scan, save_textures=True)

    for src, name in ((sample.face_obj, "face.obj"), (sample.skull_obj, "skull.obj")):
        mesh = load_obj_as_mesh(src, load_textures=False, device="cpu")
        mesh = apply_offset_scale(mesh, trans[0], trans[1])
        save_mesh_as_obj(out_dir / name, mesh, save_textures=False)
