from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .mesh_ops import load_obj_as_mesh, save_mesh_as_obj
from .paths import ensure_dir


def compute_rotation_matrix(angles):
    n_b = angles.size(0)
    sinx = torch.sin(angles[:, 0])
    siny = torch.sin(angles[:, 1])
    sinz = torch.sin(angles[:, 2])
    cosx = torch.cos(angles[:, 0])
    cosy = torch.cos(angles[:, 1])
    cosz = torch.cos(angles[:, 2])
    rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)
    rotXYZ[0, :, 1, 1] = cosx
    rotXYZ[0, :, 1, 2] = -sinx
    rotXYZ[0, :, 2, 1] = sinx
    rotXYZ[0, :, 2, 2] = cosx
    rotXYZ[1, :, 0, 0] = cosy
    rotXYZ[1, :, 0, 2] = siny
    rotXYZ[1, :, 2, 0] = -siny
    rotXYZ[1, :, 2, 2] = cosy
    rotXYZ[2, :, 0, 0] = cosz
    rotXYZ[2, :, 0, 1] = -sinz
    rotXYZ[2, :, 1, 0] = sinz
    rotXYZ[2, :, 1, 1] = cosz
    rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])
    return rotation.permute(0, 2, 1)


def split_coeffs(coeffs):
    return coeffs[:, :80], coeffs[:, 80:144], coeffs[:, 144:224], coeffs[:, 224:227], coeffs[:, 227:254], coeffs[:, 254:]


def back_rigid_transform(vs, rot, trans):
    vs_t = vs - trans.view(-1, 1, 3)
    return torch.matmul(vs_t, torch.linalg.pinv(rot))


def back_forward_coeffs(coeffs, vs_list):
    _, _, _, angles, _, translation = split_coeffs(coeffs)
    rotation = compute_rotation_matrix(angles)
    return [back_rigid_transform(v, rotation.clone(), translation.clone()) for v in vs_list]


def normalize_pose(stage_dir: str | Path, output_dir: str | Path, device: str = "cpu") -> None:
    stage_dir = Path(stage_dir)
    output_dir = ensure_dir(output_dir)
    coeffs = np.load(stage_dir / "render_coeffs.npy")
    coeffs = torch.tensor(coeffs, dtype=torch.float32, requires_grad=False, device=device).unsqueeze(0)
    items = [
        ("face_scan.obj", True),
        ("face.obj", False),
        ("skull.obj", False),
        ("render_mesh.obj", False),
    ]
    meshes = [load_obj_as_mesh(stage_dir / name, load_textures=tex, device=device) for name, tex in items]
    res = back_forward_coeffs(coeffs.clone(), [m.verts_padded() for m in meshes])
    meshes = [mesh.update_padded(verts) for mesh, verts in zip(meshes, res)]
    for (name, tex), mesh in zip(items, meshes):
        save_mesh_as_obj(output_dir / name, mesh, save_textures=tex)
