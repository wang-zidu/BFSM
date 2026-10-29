"""Small OBJ helpers that preserve vertex order and optional per-vertex colour."""
from pathlib import Path
import os
import tempfile

import numpy as np
import trimesh


def read_mesh(path):
    mesh = trimesh.load(path, process=False, maintain_order=True, force='mesh')
    if mesh.vertices.ndim != 2 or not np.isfinite(mesh.vertices).all():
        raise ValueError('Expected a finite three-dimensional mesh')
    return mesh


def write_mesh(path, vertices, faces, colors=None):
    path = Path(path)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in 'fiu'
            or not np.isfinite(vertices).all()):
        raise ValueError('Invalid vertices')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Triangle indices must be a nonempty integer N by 3 array in range')
    faces = faces.astype(np.int64, copy=False)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype.kind not in 'fiu':
            raise ValueError('Vertex colours must be real numeric values')
        colors = colors.astype(float, copy=False)
        if colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] < 3:
            raise ValueError('Vertex colours must have one RGB row per vertex')
        colors = colors[:, :3].copy()
        if not np.isfinite(colors).all():
            raise ValueError('Vertex colours must be finite')
        if colors.max(initial=0) > 1:
            colors /= 255
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for index, vertex in enumerate(vertices):
            values = vertex if colors is None else np.r_[vertex, colors[index]]
            stream.write('v ' + ' '.join(f'{value:.9g}' for value in values) + '\n')
        for triangle in faces:
            stream.write('f ' + ' '.join(str(int(value) + 1) for value in triangle) + '\n')


def replace_obj_vertices(source, destination, vertices):
    """Replace coordinates atomically, preserving OBJ records and allowing in-place use."""
    vertices = np.asarray(vertices)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in 'fiu'
            or not np.isfinite(vertices).all()):
        raise ValueError('Replacement vertices must be a finite N by 3 array')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with Path(source).open() as original, tempfile.NamedTemporaryFile(
                mode='w', dir=destination.parent, prefix='.' + destination.name + '.',
                suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            count = 0
            for line in original:
                tokens = line.split()
                if tokens and tokens[0] == 'v':
                    if count >= len(vertices):
                        raise ValueError('Source has more vertices than the replacement')
                    output.write('v ' + ' '.join(f'{v:.9g}' for v in vertices[count]) +
                                 (' ' + ' '.join(tokens[4:]) if len(tokens) > 4 else '') + '\n')
                    count += 1
                else:
                    output.write(line)
            if count != len(vertices):
                raise ValueError('Source has fewer vertices than the replacement')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
