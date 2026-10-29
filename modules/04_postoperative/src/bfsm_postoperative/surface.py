"""Cotangent ARAP detail transfer from the original postoperative implementation."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from bfsm_core.mesh import read_mesh, replace_obj_vertices
from bfsm_core.paths import check_output_inputs


def cotangent_system(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    def cot(u, v):
        return np.einsum('ij,ij->i', u, v) / (np.linalg.norm(np.cross(u, v), axis=1) + 1e-8)
    weights = np.concatenate((cot(c - a, c - b), cot(a - b, a - c), cot(b - c, b - a)))
    first = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    second = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    row, column = np.r_[first, second], np.r_[second, first]
    weights = np.r_[weights, weights]
    adjacency = sparse.coo_matrix((weights, (row, column)), shape=(len(vertices), len(vertices))).tocsr()
    adjacency.sum_duplicates()
    negative_edges = int(np.count_nonzero(adjacency.data < 0) // 2)
    # Negative edge sums can make local rotation energies indefinite: on the
    # actual scan even original-position anchors caused spontaneous deformation.
    # Clamp only after both incident cotangents have been summed.
    adjacency.data = np.maximum(adjacency.data, 0)
    adjacency.eliminate_zeros()
    # The original Laplacian has negative off-diagonals. Edge weights are their negation.
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    edges = adjacency.tocoo()
    return laplacian, edges.row, edges.col, edges.data, negative_edges


def arap(vertices, faces, anchor_indices, anchor_positions, iterations=10, duplicate_policy='last'):
    original = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    raw_indices = np.asarray(anchor_indices)
    positions = np.asarray(anchor_positions, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or not len(original) or not np.isfinite(original).all():
        raise ValueError('ARAP vertices must be a nonempty finite N by 3 array')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(original)):
        raise ValueError('ARAP triangles must contain integer vertex indices in range')
    if raw_indices.dtype.kind not in 'iu':
        raise ValueError('Anchor indices must be integers, not rounded coordinates')
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Anchor positions must be finite three-dimensional points')
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError('ARAP iteration count must be a nonnegative integer')
    current = original.copy()
    indices = raw_indices.astype(np.int64).ravel()
    if not len(indices) or len(indices) != len(positions):
        raise ValueError('At least one matching anchor index/position is required')
    if indices.min() < 0 or indices.max() >= len(original):
        raise ValueError('Anchor index out of range')
    unique, inverse, count = np.unique(indices, return_inverse=True, return_counts=True)
    # Match the original constraint loop: the last prescription wins when
    # multiple topology vertices select the same scanned-surface vertex.
    _, reversed_first = np.unique(indices[::-1], return_index=True)
    targets = positions[len(indices) - 1 - reversed_first].copy()
    residual = np.linalg.norm(positions - targets[inverse], axis=1)
    conflict = int(np.count_nonzero(residual > 1e-10))
    if conflict and duplicate_policy == 'error':
        raise ValueError('Multiple control points prescribe different positions for the same surface vertex')
    if duplicate_policy not in {'last', 'error'}:
        raise ValueError('Duplicate policy must be last or error')
    laplacian, row, column, weights, negative_edges = cotangent_system(original, faces)
    adjacency = sparse.coo_matrix((np.ones(len(row)), (row, column)), shape=laplacian.shape).tocsr()
    number, labels = connected_components(adjacency, directed=False)
    anchored_labels = np.unique(labels[unique])
    missing = np.setdiff1d(np.arange(number), anchored_labels)
    fixed = np.asarray([np.flatnonzero(labels == label)[0] for label in missing], dtype=np.int64)
    if len(fixed):
        # Each unconstrained component otherwise has an undetermined rigid translation.
        unique = np.r_[unique, fixed]
        targets = np.vstack((targets, original[fixed]))
    system = laplacian.tolil()
    for index in unique:
        system.rows[index] = [int(index)]
        system.data[index] = [1.0]
    factor = splu(system.tocsc())
    rest_edges = original[row] - original[column]
    for iteration in range(iterations):
        deformed_edges = current[row] - current[column]
        covariance = np.zeros((len(original), 3, 3))
        values = weights[:, None, None] * deformed_edges[:, :, None] * rest_edges[:, None, :]
        np.add.at(covariance, row, values)
        u, _, vt = np.linalg.svd(covariance)
        rotations = u @ vt
        reflection = np.linalg.det(rotations) < 0
        u[reflection, :, -1] *= -1
        rotations = u @ vt
        edge_rhs = 0.5 * weights[:, None] * np.einsum(
            'nij,nj->ni', rotations[row] + rotations[column], rest_edges)
        rhs = np.zeros_like(original)
        np.add.at(rhs, row, edge_rhs)
        rhs[unique] = targets
        current = factor.solve(rhs)
        if not np.isfinite(current).all():
            raise ValueError('ARAP solve produced non-finite vertices')
        print(f'ARAP {iteration + 1}/{iterations}', flush=True)
    record = {'iterations': iterations, 'edge_weights': 'negative_laplacian_off_diagonal',
              'negative_cotangent_edges_clamped': negative_edges,
              'duplicate_policy': duplicate_policy, 'duplicate_anchor_count': int(len(indices) - len(count)),
              'conflicting_control_point_count': conflict, 'max_duplicate_target_deviation': float(residual.max(initial=0)),
              'unconstrained_components_pinned_to_original': int(len(fixed))}
    return current, record


def surface_assets(surface_path, output_path):
    """Validate referenced materials before geometry or destination files are changed."""
    source_root, destination = Path(surface_path).parent, Path(output_path).parent
    reserved = {Path(output_path).resolve()}
    copies = {}

    def relative_name(value):
        path = Path(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Material and texture paths must stay relative to the surface')
        return path

    def register(source, target):
        if not source.is_file():
            raise ValueError(f'Referenced surface asset is missing: {source.name}')
        resolved = target.resolve()
        if resolved in reserved:
            raise ValueError('Material or texture output collides with the geometry path')
        if resolved in copies and copies[resolved][0].resolve() != source.resolve():
            raise ValueError('Different surface assets map to the same output file')
        copies[resolved] = (source, target)

    for line in Path(surface_path).read_text().splitlines():
        if line.startswith('mtllib '):
            relative = relative_name(line.split(maxsplit=1)[1])
            source, target = source_root / relative, destination / relative
            register(source, target)
            for entry in source.read_text().splitlines():
                if entry.startswith('map_Kd '):
                    texture = relative_name(entry.split(maxsplit=1)[1])
                    register(source.parent / texture, target.parent / texture)
    return list(copies.values())


def transfer_details(surface_path, before_face_path, predicted_face_path, output_path, duplicate_policy='last'):
    assets = surface_assets(surface_path, output_path)
    destination = Path(output_path)
    protected = (before_face_path, predicted_face_path, *(source for source, _ in assets))
    check_output_inputs(destination.parent, (destination.name,), protected)
    for source, target in assets:
        # Copying a material to itself is the intentional in-place surface case.
        if source.resolve() != target.resolve():
            check_output_inputs(target.parent, (target.name,),
                                (surface_path, before_face_path, predicted_face_path,
                                 *(other for other, _ in assets if other.resolve() != source.resolve())))
    surface, before, predicted = [read_mesh(path) for path in (surface_path, before_face_path, predicted_face_path)]
    if len(before.vertices) != len(predicted.vertices):
        raise ValueError('Before/predicted face vertex order must match')
    if not np.array_equal(before.faces, predicted.faces):
        raise ValueError('Before/predicted face topology must match')
    _, anchors = cKDTree(surface.vertices).query(before.vertices)
    # Original postoperative semantics: selected scan vertices are constrained
    # to the absolute predicted topology positions, not residual-preserving offsets.
    target = predicted.vertices
    vertices, report = arap(surface.vertices, surface.faces, anchors, target,
                            duplicate_policy=duplicate_policy)
    report['anchor_targets'] = 'predicted_topology_absolute_positions'
    replace_obj_vertices(surface_path, output_path, vertices)
    for source, target in assets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', required=True)
    parser.add_argument('--before-face', required=True)
    parser.add_argument('--predicted-face', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duplicate-policy', choices=('last', 'error'), default='last')
    args = parser.parse_args()
    transfer_details(args.surface, args.before_face, args.predicted_face, args.output, args.duplicate_policy)


if __name__ == '__main__':
    main()
