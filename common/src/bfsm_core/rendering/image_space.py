"""Depth-tested mesh rasterization in original-image pixels, without crop clipping."""
import numpy as np


def shaded_colors(vertices, faces, color):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    triangle = vertices[faces]
    normals = np.zeros_like(vertices)
    area = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    light = np.array([0., .4, .92]); light /= np.linalg.norm(light)
    return np.asarray(color) * (.55 + .45 * np.maximum(normals @ light, 0.))[:, None]


def rasterize_pixels(pixels, faces, depth, features, image_size, *, perspective=False):
    """Return feature/depth-visible triangle ROI and its (left, top) image offset.

    Smaller depth wins. Perspective mode uses reciprocal-depth interpolation for
    depth and features. Only the original image bounds clip the projected mesh.
    """
    pixels, faces, depth, features = map(np.asarray, (pixels, faces, depth, features))
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels)
            or pixels.dtype.kind not in 'fiu' or not np.isfinite(pixels).all()
            or depth.shape != (len(pixels),) or depth.dtype.kind not in 'fiu'
            or not np.isfinite(depth).all()):
        raise ValueError('Expected finite projected pixel coordinates and one depth per vertex')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(pixels)):
        raise ValueError('Expected nonempty valid triangle indices')
    if (features.ndim != 2 or features.shape[0] != len(pixels) or features.shape[1] < 1
            or features.dtype.kind not in 'fiu' or not np.isfinite(features).all()):
        raise ValueError('Expected finite per-vertex features')
    if (len(image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                                   or v < 1 for v in image_size)):
        raise ValueError('Image size must contain positive integer height and width')
    if perspective and (depth <= 0).any():
        raise ValueError('Perspective vertices must be in front of the camera')
    height, width = image_size
    pixels, depth = pixels.astype(float), depth.astype(float)
    limit = np.array([width, height])
    lower = np.clip(np.floor(pixels.min(0)), 0, limit).astype(int)
    upper = np.clip(np.ceil(pixels.max(0)) + 1, 0, limit).astype(int)
    shape = np.maximum(upper - lower, 0)[::-1]
    image = np.zeros((*shape, features.shape[1]), np.float32)
    ids = np.full(shape, -1, np.int32)
    zbuffer = np.full(shape, np.inf)
    if not np.all(shape):
        return image, ids, tuple(lower)
    screen = pixels - lower
    for index, triangle in enumerate(faces):
        p = screen[triangle]
        low = np.clip(np.ceil(p.min(0)), 0, shape[::-1] - 1).astype(int)
        high = np.clip(np.floor(p.max(0)), 0, shape[::-1] - 1).astype(int)
        if (high < low).any():
            continue
        determinant = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0])
                       + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if determinant == 0:
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        first = ((p[1, 1] - p[2, 1]) * (x - p[2, 0])
                 + (p[2, 0] - p[1, 0]) * (y - p[2, 1])) / determinant
        second = ((p[2, 1] - p[0, 1]) * (x - p[2, 0])
                  + (p[0, 0] - p[2, 0]) * (y - p[2, 1])) / determinant
        weights = np.stack((first, second, 1. - first - second), axis=-1)
        inside = (weights >= -1e-10).all(-1)
        if perspective:
            reciprocal = weights / depth[triangle]
            denominator = reciprocal.sum(-1)
            safe = np.where(inside, denominator, 1.)
            z = 1. / safe
            weights = reciprocal / safe[..., None]
        else:
            z = weights @ depth[triangle]
        region = np.s_[low[1]:high[1] + 1, low[0]:high[0] + 1]
        visible = inside & (z < zbuffer[region])
        zbuffer[region][visible] = z[visible]
        ids[region][visible] = index
        image[region][visible] = (weights @ features[triangle])[visible]
    return image, ids, tuple(lower)
