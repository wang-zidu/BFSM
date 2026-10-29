from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes
from scipy.spatial import cKDTree


def _ray_batch_size(mesh, requested: int | None):
    if requested is not None:
        return requested
    return 8192 if "ray_pyembree" in type(mesh.ray).__module__ else 8


def face_vertices(vertices, faces):
    bs, nv = vertices.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))
    return vertices[faces.long()]


def sample_colors(vertices, colors, face_indices, points):
    triangle_vertices = vertices[face_indices]
    triangle_colors = colors[face_indices]
    distances = np.linalg.norm(triangle_vertices - points[:, None, :], axis=2)
    distances[distances == 0] = 1e-10
    weights = 1 / distances
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, triangle_colors)


def _read_vertex_colors(mesh_path: str | Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if colors.size == 0:
        raise ValueError(f"mesh has no vertex colors: {mesh_path}")
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _closest_ray_hits(mesh, origins, directions, batch_size: int):
    points = np.full_like(origins, np.nan, dtype=np.float64)
    faces = np.full(len(origins), -1, dtype=np.int64)
    has_hit = np.zeros(len(origins), dtype=bool)

    for start in range(0, len(origins), batch_size):
        stop = min(start + batch_size, len(origins))
        batch_origins = origins[start:stop]
        index_tri, index_ray, locations = mesh.ray.intersects_id(
            ray_origins=batch_origins,
            ray_directions=directions[start:stop],
            return_locations=True,
            multiple_hits=False,
        )
        if len(locations) == 0:
            continue
        ray_indices = start + index_ray
        points[ray_indices] = locations
        faces[ray_indices] = index_tri
        has_hit[ray_indices] = True

    return points, faces, has_hit


def _closest_bidirectional_ray_hits(mesh, origins, normals, batch_size: int):
    points, faces, has_hit = _closest_ray_hits(mesh, origins, normals, batch_size)
    missing = np.flatnonzero(~has_hit)
    if len(missing) == 0:
        return points, faces, has_hit

    back_points, back_faces, back_has_hit = _closest_ray_hits(
        mesh,
        origins[missing],
        -normals[missing],
        batch_size,
    )
    hit_indices = missing[back_has_hit]
    points[hit_indices] = back_points[back_has_hit]
    faces[hit_indices] = back_faces[back_has_hit]
    has_hit[hit_indices] = True
    return points, faces, has_hit


def find_corresponding_points_and_colors(
    mesh_A,
    mesh_B,
    mesh_B_temp_vertices,
    mesh_B_temp_texture,
    mesh_vertices_texture,
    geometry_mode: str = "nearest",
    texture_mode: str = "ray",
    ray_batch_size: int | None = None,
):
    if geometry_mode not in {"nearest", "ray"}:
        raise ValueError(f"unsupported geometry_mode: {geometry_mode}")
    if texture_mode not in {"ray"}:
        raise ValueError(f"unsupported texture_mode: {texture_mode}")
    vertices = np.asarray(mesh_A.vertices)
    normals = np.asarray(mesh_A.vertex_normals)
    color_results = mesh_vertices_texture.copy()
    kdtree_B = cKDTree(mesh_B.vertices)
    _, nearest_indices = kdtree_B.query(vertices)
    nearest_points = mesh_B.vertices[nearest_indices]
    ray_points, ray_faces, ray_hits = _closest_bidirectional_ray_hits(
        mesh_B,
        vertices,
        normals,
        batch_size=_ray_batch_size(mesh_B, ray_batch_size),
    )
    if texture_mode == "ray" and np.any(ray_hits):
        color_results[ray_hits] = sample_colors(
            mesh_B_temp_vertices,
            mesh_B_temp_texture,
            ray_faces[ray_hits],
            ray_points[ray_hits],
        )
    results = nearest_points
    if geometry_mode == "ray":
        results = nearest_points.copy()
        results[ray_hits] = ray_points[ray_hits]
    return results, color_results


def transfer_render_texture(
    stage_dir: str | Path,
    source_color_stage_dir: str | Path | None = None,
    geometry_mode: str = "nearest",
    texture_mode: str = "ray",
    ray_batch_size: int | None = None,
) -> None:
    stage_dir = Path(stage_dir)
    source_color_stage_dir = Path(source_color_stage_dir) if source_color_stage_dir else stage_dir
    mesh_A = trimesh.load(stage_dir / "render_mesh.obj", process=False, maintain_order=True)
    mesh_B = trimesh.load(stage_dir / "face_scan.obj", process=False, maintain_order=True)
    mesh_B_temp = load_objs_as_meshes([str(stage_dir / "face_scan.obj")], device="cpu")
    mesh_B_temp_vertices = face_vertices(mesh_B_temp.verts_padded(), mesh_B_temp.faces_padded()).squeeze().cpu().numpy()
    mesh_B_temp_texture = mesh_B_temp.textures.faces_verts_textures_packed().cpu().numpy()
    mesh_vertices_texture = _read_vertex_colors(source_color_stage_dir / "render_mesh.obj")
    vertex_array, color_results = find_corresponding_points_and_colors(
        mesh_A,
        mesh_B,
        mesh_B_temp_vertices,
        mesh_B_temp_texture,
        mesh_vertices_texture,
        geometry_mode=geometry_mode,
        texture_mode=texture_mode,
        ray_batch_size=ray_batch_size,
    )
    trimesh.Trimesh(
        vertices=vertex_array,
        faces=mesh_A.faces,
        vertex_colors=color_results,
        process=False,
        maintain_order=True,
    ).export(stage_dir / "face_scan_topology_points.obj")


def transfer_final_face_texture(stage_dir: str | Path, source_color_stage_dir: str | Path) -> None:
    stage_dir = Path(stage_dir)
    mesh_A = trimesh.load(stage_dir / "face_nicp.obj", process=False, maintain_order=True)
    mesh_B = trimesh.load(stage_dir / "face_scan.obj", process=False, maintain_order=True)
    mesh_B_temp = load_objs_as_meshes([str(stage_dir / "face_scan.obj")], device="cpu")
    mesh_B_temp_vertices = face_vertices(mesh_B_temp.verts_padded(), mesh_B_temp.faces_padded()).squeeze().cpu().numpy()
    mesh_B_temp_texture = mesh_B_temp.textures.faces_verts_textures_packed().cpu().numpy()
    mesh_vertices_texture = _read_vertex_colors(source_color_stage_dir / "render_mesh.obj")
    _, color_results = find_corresponding_points_and_colors(
        mesh_A,
        mesh_B,
        mesh_B_temp_vertices,
        mesh_B_temp_texture,
        mesh_vertices_texture,
        geometry_mode="nearest",
        texture_mode="ray",
        ray_batch_size=None,
    )
    trimesh.Trimesh(
        vertices=mesh_A.vertices,
        faces=mesh_A.faces,
        vertex_colors=color_results,
        process=False,
        maintain_order=True,
    ).export(stage_dir / "topology_face.obj")
