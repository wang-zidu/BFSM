"""Build native FLAME-face BFSM/TMM from explicitly supplied licensed assets.

No original FLAME identity weights, renderer, patient scans or upstream converter
implementation are needed. The fixed template correspondence is a new geometry
representation; its source asset terms still apply to resulting derivatives.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import trimesh
from bfsm_core.paths import check_output_inputs
from .prepare_flame import ArrayUnpickler, StoredState


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_legacy_array(stream):
    version = np.lib.format.read_magic(stream)
    _, _, dtype = np.lib.format._read_array_header(stream, version)
    if not dtype.hasobject:
        raise ValueError('Expected the legacy scalar correspondence dictionary')
    value = ArrayUnpickler(stream, encoding='latin1').load()
    if not isinstance(value, np.ndarray) or value.size != 1:
        raise ValueError('Expected one correspondence dictionary')
    return value.item()


def load_correspondence(path):
    # Sparse and Chumpy pickle constructors are inert StoredState objects;
    # no SciPy/Chumpy model class is imported or executed by the unpickler.
    with zipfile.ZipFile(path) as archive:
        with archive.open('BFM2009_cropped_corr.npy') as stream:
            data = read_legacy_array(stream)
        with archive.open('FLAME_mask_ids.npy') as stream:
            mask = np.load(stream, allow_pickle=False).reshape(-1)
    legacy = data['mtx']
    if not isinstance(legacy, StoredState):
        raise ValueError('Expected inert legacy sparse state')
    state = legacy.state
    matrix = sparse.csc_matrix((state['data'], state['indices'], state['indptr']),
                              shape=state['_shape']).tocsr()
    matrix.eliminate_zeros()
    if matrix.shape != (5023, 106430):
        raise ValueError('Expected the 5023 by (2 * 53215) correspondence version')
    return matrix, np.asarray(data['f_out'], dtype=int), mask


def pack_sparse(target, name, matrix):
    target.update({name+'_data': matrix.data, name+'_indices': matrix.indices,
                   name+'_indptr': matrix.indptr, name+'_shape': np.asarray(matrix.shape)})


def fixed_map(matrix, full_triangles, converter_mask, crop, anatomical, reference, source_triangles):
    if crop.shape != (35709,) or np.unique(crop).size != len(crop) or crop.min() < 0 or crop.max() >= 53215:
        raise ValueError('Expected 35709 unique BFM crop indices in [0, 53215)')
    inverse = np.full(53215, -1, int)
    inverse[crop] = np.arange(len(crop))
    rows, columns, values, ids = [], [], [], []
    for vertex in converter_mask:
        row = matrix.getrow(int(vertex))
        indices, weights = row.indices, row.data
        if np.any(indices >= 53215) or np.any(inverse[indices] < 0) or np.any(weights < -1e-12):
            continue
        ids.append(vertex)
        for index, weight in zip(inverse[indices], weights):
            rows.append(len(ids)-1)
            columns.append(index)
            values.append(weight)
    mapping = sparse.csr_matrix((values, (rows, columns)), shape=(len(ids), 35709))
    ids = np.asarray(ids)
    mask = np.intersect1d(converter_mask, anatomical['face'])
    exclusions = np.unique(np.concatenate([anatomical[name] for name in
        ['neck', 'scalp', 'left_ear', 'right_ear', 'left_eyeball', 'right_eyeball']]))
    mask = np.setdiff1d(mask, exclusions)
    retained = np.isin(ids, mask)
    mapping, ids = mapping[retained].tocsr(), ids[retained]
    seed_count = len(ids)

    lookup = np.full(5023, -1, int)
    lookup[mask] = np.arange(len(mask))
    local_triangles = lookup[full_triangles[np.all(lookup[full_triangles] >= 0, axis=1)]]
    edges = np.concatenate([local_triangles[:, [0, 1]], local_triangles[:, [1, 2]], local_triangles[:, [2, 0]]])
    adjacency = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mask), len(mask)))
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    known = lookup[ids]
    unknown = np.setdiff1d(np.arange(len(mask)), known)
    positions = np.empty((len(mask), 3))
    positions[known] = mapping @ reference
    positions[unknown] = spsolve(laplacian[unknown][:, unknown], -laplacian[unknown][:, known] @ positions[known])
    triangles = reference[source_triangles]
    _, near = cKDTree(triangles.mean(1)).query(positions[unknown], k=128)
    probes = np.repeat(positions[unknown], 128, axis=0)
    closest = trimesh.triangles.closest_point(triangles[near.ravel()], probes).reshape(-1, 128, 3)
    distances = np.linalg.norm(closest - positions[unknown, None], axis=2)
    best = distances.argmin(1)
    chosen = near[np.arange(len(unknown)), best]
    points = closest[np.arange(len(unknown)), best]
    weights = trimesh.triangles.points_to_barycentric(triangles[chosen], points)
    weights = np.maximum(weights, 0)
    weights /= weights.sum(1, keepdims=True)
    filled = sparse.csr_matrix((weights.ravel(), (np.repeat(unknown, 3), source_triangles[chosen].ravel())), shape=(len(mask), 35709))
    permutation = sparse.csr_matrix((np.ones(len(known)), (known, np.arange(len(known)))), shape=(len(mask), len(known)))
    harmonic = spsolve(laplacian[unknown][:, unknown], -laplacian[unknown][:, known].toarray())
    harmonic = np.maximum(harmonic, 0)
    harmonic[harmonic < 1e-10] = 0
    harmonic /= harmonic.sum(1, keepdims=True)
    harmonic_rows = sparse.csr_matrix(harmonic) @ mapping
    insertion = sparse.csr_matrix((np.ones(len(unknown)), (unknown, np.arange(len(unknown)))), shape=(len(mask), len(unknown)))
    filled = .85 * filled + .15 * (insertion @ harmonic_rows)
    mapping = (permutation @ mapping + filled).tocsr()
    mapping.eliminate_zeros()
    _, labels = connected_components(adjacency, directed=False)
    skin = labels == np.argmax(np.bincount(labels))
    ids, mapping = mask[skin], mapping[skin].tocsr()
    lookup[:] = -1
    lookup[ids] = np.arange(len(ids))
    triangles = lookup[full_triangles[np.all(lookup[full_triangles] >= 0, axis=1)]]
    report = {'initial_vertices': seed_count, 'completed_vertices': len(unknown),
              'completion_method': 'harmonic_0.15_plus_surface_projection_0.85',
              'projection_max': float(distances.min(1).max()),
              'projection_median': float(np.median(distances.min(1))),
              'face_vertices': len(ids), 'face_triangles': len(triangles)}
    return mapping, ids, triangles, report


def build(args):
    check_output_inputs(args.output, ['flame_fsmm_200.npz', 'flame_tmm_200.npz'],
                        [args.correspondence, args.crop_indices, args.anatomical_masks, args.fsmm, args.tmm])
    face = load_arrays(args.fsmm)
    tissue = load_arrays(args.tmm)
    if any(np.asarray(face[key]).ndim != 0 or np.asarray(face[key]).dtype.kind not in 'iu'
           for key in ('face_vertex_count', 'skull_vertex_count')):
        raise ValueError('FSMM vertex counts must be integer scalars')
    count = int(face['face_vertex_count'])
    skull_count = int(face['skull_vertex_count'])
    if count != 35709 or skull_count <= 0:
        raise ValueError('Expected the 35709-vertex BFSM face and a positive skull vertex count')
    components = 200
    if (face['shapeMU'].size != 3 * (count + skull_count) or
            face['shapeBase'].shape != (3 * (count + skull_count), components)):
        raise ValueError('FSMM geometry must match its vertex counts and 200 identity components')
    for key, vertex_count in (('face_tri', count), ('skull_tri', skull_count)):
        triangles = face[key]
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or
                triangles.dtype.kind not in 'iu' or triangles.min() < 0 or
                triangles.max() >= vertex_count or not np.array_equal(triangles, tissue[key])):
            raise ValueError(f'FSMM/TMM {key} must share valid integer triangles in vertex range')
    pairs = tissue['face_to_skull_ind_new']
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or
            pairs.dtype.kind not in 'iu' or pairs.min() < 0 or
            pairs[:, 0].max() >= count or pairs[:, 1].max() >= skull_count):
        raise ValueError('TMM pairs must reference the supplied face and skull vertex counts')
    if ('face_to_skull_ind_new' in face and
            not np.array_equal(face['face_to_skull_ind_new'], pairs)):
        raise ValueError('FSMM and TMM pair records must have identical order')
    if (tissue['shapeMU'].size != 3 * len(pairs) or
            tissue['shapeBase'].shape != (3 * len(pairs), components)):
        raise ValueError('TMM geometry must match its pair records and 200 tissue components')
    if any(arrays[key].dtype.kind not in 'fiu' or not np.isfinite(arrays[key]).all() for arrays in (face, tissue)
           for key in ('shapeMU', 'shapeBase')):
        raise ValueError('Model geometry must be finite')
    mean = face['shapeMU'].reshape(-1, 3)
    basis = face['shapeBase'].reshape(-1, 3, components)
    center = face.get('recenter_offset', mean.mean(0, keepdims=True))
    if (np.asarray(center).size != 3 or np.asarray(center).dtype.kind not in 'fiu'
            or not np.isfinite(center).all()):
        raise ValueError('FSMM recenter_offset must contain three finite coordinates')
    center = np.asarray(center).reshape(1, 3)
    matrix, full_triangles, converter_mask = load_correspondence(args.correspondence)
    crop = np.load(args.crop_indices, allow_pickle=False).reshape(-1)
    with args.anatomical_masks.open('rb') as stream:
        anatomical = ArrayUnpickler(stream, encoding='latin1').load()
    mapping, ids, triangles, report = fixed_map(matrix, full_triangles, converter_mask, crop,
        anatomical, mean[:count].astype(float), face['face_tri'].astype(int))
    mapped_mean = mapping @ mean[:count]
    mapped_basis = (mapping @ basis[:count].reshape(count, -1)).reshape(len(ids), 3, components)
    native = {key: face[key] for key in ['skull_tri', 'skull_vertex_count']}
    native.update(shapeMU=np.concatenate([mapped_mean, mean[count:]]).reshape(1, -1).astype(np.float32),
        shapeBase=np.concatenate([mapped_basis, basis[count:]]).reshape(-1, components).astype(np.float32),
        face_tri=triangles, face_vertex_count=np.asarray(len(ids)),
        recenter_offset=center, flame_vertex_ids=ids)
    pack_sparse(native, 'A', mapping)

    counts = np.bincount(pairs[:, 0], minlength=count)
    aggregation = sparse.csr_matrix((1/counts[pairs[:, 0]], (pairs[:, 0], np.arange(len(pairs)))), shape=(count, len(pairs)))
    supported = np.asarray((mapping @ (counts == 0).astype(float)) == 0).reshape(-1)
    combined = (mapping[supported] @ aggregation).tocsr()
    selection = sparse.csr_matrix((np.ones(len(pairs)), (np.arange(len(pairs)), pairs[:, 1])), shape=(len(pairs), skull_count))
    anchors = (combined @ selection).tocsr()
    tissue_mean = combined @ tissue['shapeMU'].reshape(-1, 3)
    tissue_basis = (combined @ tissue['shapeBase'].reshape(len(pairs), -1)).reshape(-1, components)
    native_tissue = dict(shapeMU=tissue_mean.reshape(1, -1).astype(np.float32),
        shapeBase=tissue_basis.astype(np.float32), flame_vertex_ids=ids[supported],
        native_face_indices=np.flatnonzero(supported), vector_sign=np.asarray('skull_minus_face'),
        aggregation=np.asarray('equal_weight_per_old_face_vertex'))
    for name, value in [('C', combined), ('R', aggregation), ('H', anchors)]:
        pack_sparse(native_tissue, name, value)
    report.update(tissue_vertices=int(supported.sum()), A_row_sum_error=float(abs(mapping.sum(1)-1).max()),
                  C_row_sum_error=float(abs(combined.sum(1)-1).max()), H_row_sum_error=float(abs(anchors.sum(1)-1).max()))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, arrays in [('flame_fsmm_200.npz', native), ('flame_tmm_200.npz', native_tissue)]:
        output = args.output / filename
        np.savez_compressed(output, **arrays)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['correspondence', 'crop-indices', 'anatomical-masks', 'fsmm', 'tmm', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    build(p.parse_args())


if __name__ == '__main__':
    main()
