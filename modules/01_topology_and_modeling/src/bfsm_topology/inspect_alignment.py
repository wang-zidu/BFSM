"""Export unchanged scanner/CT geometry as a MeshLab alignment inspection scene."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bfsm_core.mesh import read_mesh


def write_ply(path, vertices, faces, color):
    """Use double XYZ text so the inspection export preserves input coordinates."""
    with path.open('w') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write(f'element vertex {len(vertices)}\n')
        handle.write('property double x\nproperty double y\nproperty double z\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        handle.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        rgb = ' '.join(map(str, color))
        for point in vertices:
            handle.write(' '.join(format(float(x), '.17g') for x in point) + ' ' + rgb + '\n')
        for face in faces:
            handle.write('3 ' + ' '.join(str(int(i)) for i in face) + '\n')


def export_alignment(ct_face, skull, output, scan_face=None):
    inputs = [('CT facial skin', Path(ct_face), 'ct_skin.ply', (145, 189, 212)),
              ('CT skull', Path(skull), 'ct_skull.ply', (163, 167, 173))]
    if scan_face is not None:
        inputs.append(('Scanner facial skin', Path(scan_face), 'scanner_skin.ply', (55, 125, 180)))
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new alignment inspection directory')
    for _, path, _, _ in inputs:
        if output.resolve() == path.resolve() or output.resolve() in path.resolve().parents:
            raise ValueError('Inspection output must not contain an input file')
    # Validate every input before creating any output. No pose is estimated here.
    loaded = [(label, path, name, color, read_mesh(path)) for label, path, name, color in inputs]
    for _, _, _, _, mesh in loaded:
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Expected finite triangular input meshes')
    output.mkdir(parents=True)
    project = ET.Element('MeshLabProject')
    group = ET.SubElement(project, 'MeshGroup')
    for label, path, name, color, mesh in loaded:
        write_ply(output / name, np.asarray(mesh.vertices), np.asarray(mesh.faces), color)
        layer = ET.SubElement(group, 'MLMesh', label=label, filename=name)
        ET.SubElement(layer, 'MLMatrix44').text = '\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n'
        print(f'{label}: {len(mesh.vertices)} vertices / {len(mesh.faces)} triangles; coordinates unchanged')
    ET.SubElement(project, 'RasterGroup')
    scene = output / 'alignment.mlp'
    scene.write_bytes(b'<!DOCTYPE MeshLabDocument>\n' + ET.tostring(project, encoding='utf-8'))
    print(f'Open {scene} in MeshLab. This is an inspection scene, not an alignment result.')
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct-face', required=True)
    parser.add_argument('--skull', required=True)
    parser.add_argument('--scan-face')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_alignment(args.ct_face, args.skull, args.output, args.scan_face)


if __name__ == '__main__':
    main()
