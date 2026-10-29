"""Compose independently prepared bone pieces with explicit rigid transforms."""
import argparse
from pathlib import Path
import tomllib

import numpy as np

from bfsm_core.mesh import read_mesh, write_mesh


def compose(config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    settings = tomllib.loads(config.read_text())
    pieces = settings.get('pieces')
    if not isinstance(pieces, list) or not pieces:
        raise ValueError('Configuration requires a nonempty pieces array')
    names = [piece.get('name') for piece in pieces]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError('Piece names must be nonempty and unique')
    inputs = [config] + [(config.parent / piece['mesh']).resolve() for piece in pieces]
    bridge = settings.get('bridge')
    if bridge is not None:
        inputs.append((config.parent / bridge['faces']).resolve())
    if any(output == path or (output.exists() and path.exists() and output.samefile(path)) for path in inputs):
        raise ValueError('Output must not overwrite configuration, a bone piece or bridge input')
    vertices, faces, offset = [], [], 0
    for piece, path in zip(pieces, inputs[1:]):
        mesh = read_mesh(path)
        axis = np.asarray(piece['axis'], dtype=float)
        pivot = np.asarray(piece['pivot'], dtype=float)
        translation = np.asarray(piece['translation'], dtype=float)
        angle = float(piece['angle_degrees'])
        if (any(value.shape != (3,) or not np.isfinite(value).all() for value in (axis, pivot, translation))
                or not np.isfinite(angle)):
            raise ValueError('Rigid transforms need finite 3-vectors and an angle')
        with np.errstate(over='ignore'):
            length = np.linalg.norm(axis)
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Rotation axis must have a finite positive length')
        axis = axis / length
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        theta = np.deg2rad(angle)
        rotation = np.cos(theta)*np.eye(3) + (1-np.cos(theta))*np.outer(axis, axis) + np.sin(theta)*skew
        moved = (mesh.vertices-pivot) @ rotation.T + pivot + translation
        vertices.append(moved)
        faces.append(mesh.faces+offset)
        offset += len(moved)
    vertices, faces = np.vstack(vertices), np.vstack(faces)
    if bridge is not None:
        with np.load(inputs[-1], allow_pickle=False) as archive:
            added = archive['added_faces']
        if (added.ndim != 2 or added.shape[1] != 3 or added.dtype.kind not in 'iu' or
                len(added) == 0 or added.min() < 0 or added.max() >= len(vertices)):
            raise ValueError('Bridge faces must be nonempty integer triangle indices in piece-concatenation order')
        triangles = vertices[added]
        area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
        if not np.isfinite(area).all() or np.any(area <= 1e-14):
            raise ValueError('Bridge triangles must be finite and nondegenerate')
        faces = np.vstack((faces, added))
    write_mesh(output, vertices, faces)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(compose(args.config, args.output))


if __name__ == '__main__':
    main()
