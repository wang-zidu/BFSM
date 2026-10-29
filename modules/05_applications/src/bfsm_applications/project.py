"""Project an original-order 35709-vertex BFSM face to the native FLAME facial patch."""
import argparse
from pathlib import Path
import numpy as np
from bfsm_core.mesh import read_mesh, write_mesh
from .model import NativeFace
from .fit import check_output_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--face', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    model = NativeFace()
    check_output_inputs(args.output.parent, [args.output.name], [args.face, model.source_path])
    mesh = read_mesh(args.face)
    vertices = model.project(np.asarray(mesh.vertices))
    colors = None
    if mesh.visual.kind == 'vertex':
        colors = model.mapping @ np.asarray(mesh.visual.vertex_colors[:, :3], float)
    write_mesh(args.output, vertices, model.faces, colors)


if __name__ == '__main__':
    main()
