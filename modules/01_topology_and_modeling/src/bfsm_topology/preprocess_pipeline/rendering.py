from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    blending,
    look_at_view_transform,
)

from .mesh_ops import load_obj_as_mesh
from .rasterization import MeshRasterizer


def _fov_from_focal(image_size: int, focal: float = 1015.0) -> float:
    return 2 * np.arctan(image_size / 2 / focal) * 180.0 / np.pi


def get_renderer(img_size: int = 224, device: str = "cuda:0"):
    r, t = look_at_view_transform(10, 0, 0)
    cameras = FoVPerspectiveCameras(
        device=device,
        R=r,
        T=t,
        znear=0.01,
        zfar=50,
        fov=_fov_from_focal(img_size),
    )
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 1e5]],
        ambient_color=[[1, 1, 1]],
        specular_color=[[0.0, 0.0, 0.0]],
        diffuse_color=[[0.0, 0.0, 0.0]],
    )
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    blend_params = blending.BlendParams(background_color=[1, 1, 1])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        ),
    )


def get_depth(meshes_world, device: str = "cuda:0", image_size: int = 1024):
    r, t = look_at_view_transform(10, 0, 0)
    cameras = FoVPerspectiveCameras(
        device=device,
        R=r,
        T=t,
        znear=0.01,
        zfar=50,
        fov=_fov_from_focal(image_size),
    )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    fragments = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)(meshes_world)
    mask = -fragments.dists
    mask[mask < 0.1] = 0
    mask = 1 - mask
    return fragments.zbuf, mask


def render_face_scan(stage_dir: str | Path, device: str = "cuda:0") -> None:
    stage_dir = Path(stage_dir)
    mesh = load_obj_as_mesh(stage_dir / "face_scan.obj", load_textures=True, device=device)
    renderer = get_renderer(device=device)
    rendered = renderer(mesh)
    image = (rendered[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(stage_dir / "render.png"), image[:, :, ::-1])
    depth, mask = get_depth(mesh, device=device)
    np.save(stage_dir / "render_depth.npy", depth.detach().cpu().numpy())
    np.save(stage_dir / "render_depth_mask.npy", mask.detach().cpu().numpy())
