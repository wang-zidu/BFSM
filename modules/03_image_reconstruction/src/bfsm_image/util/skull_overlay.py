"""Project the complete inferred skull back to the original photograph."""
from pathlib import Path
import numpy as np

from bfsm_core.rendering.image_space import rasterize_pixels


def overlay_skull(results, original_bgr, trans_params, *, segmentation=False):
    from .io import back_resize_ldms
    vertices = results['v3d_skull'][0]
    faces = results['skull_tri']
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or (vertices[:, 2] <= 0).any()):
        raise ValueError('Skull camera coordinates must be finite and in front of the camera')
    pixels = 1015. * vertices[:, :2] / vertices[:, 2:]
    pixels[:, 0] += 112.
    pixels[:, 1] = 223. - (pixels[:, 1] + 112.)
    if trans_params is not None:
        pixels = back_resize_ldms(pixels, trans_params)
    if 'skull_colors' in results:
        colors = results['skull_colors'][0]
    else:
        # Replay the original five-light skull shading for older saved results.
        physical = vertices.astype(float).copy(); physical[:, 2] = 10. - physical[:, 2]
        triangles = physical[faces]
        normal = np.zeros_like(physical)
        area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        for corner in (1, 2, 0):
            np.add.at(normal, faces[:, corner], area)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        directions = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1.]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        colors = np.repeat(np.maximum(normal @ directions.T, 0.).mean(1)[:, None], 3, axis=1)
    features = colors
    if segmentation:
        from bfsm_models.regions import load_regions, REGIONS
        import bfsm_models.regions as module
        masks = load_regions(Path(module.__file__).with_name('skull_regions.npz'), faces, len(vertices))
        features = np.column_stack((colors, *[masks[name] for name in REGIONS]))
    rendered, ids, (left, top) = rasterize_pixels(pixels, faces, vertices[:, 2], features,
                                                original_bgr.shape[:2], perspective=True)
    output = original_bgr.copy()
    labels = np.zeros(original_bgr.shape[:2], np.uint8) if segmentation else None
    height, width = ids.shape
    selected = ids >= 0
    region = np.s_[top:top + height, left:left + width]
    output[region][selected] = np.rint(np.clip(rendered[..., :3][selected][:, ::-1], 0., 1.) * 255).astype(np.uint8)
    if segmentation:
        labels[region][selected] = rendered[..., 3:][selected].argmax(-1) + 1
    return output, labels
