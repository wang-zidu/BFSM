"""CPU orthographic mesh display with per-pixel depth and smooth vertex lighting."""
import numpy as np


def orthographic_image(vertices, faces, color, center, radius, size=512):
    """Render +Z-facing geometry: +X is screen-right and +Y is screen-up.

    Geometry and triangle winding are preserved. Lighting uses area-weighted
    vertex normals in the same world frame as the fixed frontal light.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or vertices.dtype.kind not in 'fiu' or not np.isfinite(vertices).all()):
        raise ValueError('Expected finite real N by 3 vertices')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Expected integer triangles within the vertex array')
    center, color = np.asarray(center), np.asarray(color)
    if center.dtype.kind not in 'fiu' or color.dtype.kind not in 'fiu':
        raise ValueError('Expected real center and RGB material')
    center, color = center.astype(float), color.astype(float)
    if (center.shape != (3,) or color.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(color).all() or (color < 0).any() or (color > 1).any()):
        raise ValueError('Expected a finite center and RGB material in [0, 1]')
    if (not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0
            or isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2):
        raise ValueError('Expected a positive radius and integer image size of at least two')
    coordinates = vertices.astype(float) - center
    triangles = coordinates[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(coordinates)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    light = np.array([0., .4, .92]); light /= np.linalg.norm(light)
    colors = color * (.55 + .45 * np.maximum(normals @ light, 0.))[:, None]
    projected = coordinates.copy()
    projected[:, :2] = (projected[:, :2] / radius + 1.) * ((size - 1) / 2.)
    projected[:, 1] = size - 1 - projected[:, 1]
    if not np.isfinite(projected).all() or not np.isfinite(colors).all():
        raise ValueError('Orthographic projection and lighting must remain finite')
    image = np.ones((size, size, 3), dtype=np.float32)
    depth = np.full((size, size), -np.inf)
    triangle_ids = np.full((size, size), -1, dtype=np.int32)
    for index, indices in enumerate(faces):
        p = projected[indices]
        low = np.clip(np.ceil(p[:, :2].min(0)), 0, size - 1).astype(int)
        high = np.clip(np.floor(p[:, :2].max(0)), 0, size - 1).astype(int)
        if np.any(high < low):
            continue
        denominator = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0])
                       + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if denominator == 0:
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        first = ((p[1, 1] - p[2, 1]) * (x - p[2, 0])
                 + (p[2, 0] - p[1, 0]) * (y - p[2, 1])) / denominator
        second = ((p[2, 1] - p[0, 1]) * (x - p[2, 0])
                  + (p[0, 0] - p[2, 0]) * (y - p[2, 1])) / denominator
        third = 1. - first - second
        weights = np.stack((first, second, third), axis=-1)
        z = weights @ p[:, 2]
        region = np.s_[low[1]:high[1] + 1, low[0]:high[0] + 1]
        selected = (weights >= -1e-10).all(-1) & (z > depth[region])
        depth[region][selected] = z[selected]
        triangle_ids[region][selected] = index
        image[region][selected] = (weights @ colors[indices])[selected]
    return np.clip(image, 0., 1.), depth, triangle_ids
