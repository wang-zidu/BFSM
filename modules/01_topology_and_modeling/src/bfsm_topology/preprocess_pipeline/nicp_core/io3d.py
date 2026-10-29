from __future__ import annotations

from pathlib import Path

from pytorch3d.io import load_objs_as_meshes, save_obj


def load_obj_as_mesh(path: str | Path, device=None, load_textures: bool = True):
    return load_objs_as_meshes([str(path)], device=device, load_textures=load_textures)


def save_meshes_as_objs(paths, mesh, save_textures: bool = True):
    for idx, path in enumerate(paths):
        verts = mesh.verts_padded()[idx]
        faces = mesh.faces_padded()[idx]
        if save_textures:
            if mesh.textures.isempty():
                raise ValueError("cannot save untextured mesh with save_textures=True")
            texture_map = mesh.textures.maps_padded()[idx]
            verts_uvs = mesh.textures.verts_uvs_padded()[idx]
            faces_uvs = mesh.textures.faces_uvs_padded()[idx]
            save_obj(str(path), verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=texture_map)
        else:
            save_obj(str(path), verts, faces)
