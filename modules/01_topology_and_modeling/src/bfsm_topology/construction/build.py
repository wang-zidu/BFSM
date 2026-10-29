"""Construct geometry-only FSMM and TMM with the original variance-scaled PCA formula."""
import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA


def vertices_from_file(path, triangles, vertex_count, dtype):
    """Read arrays or triangular OBJ without reindexing, merging, or applying materials."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        vertices = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.obj':
        vertices, faces = [], []
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] == 'v':
                    if len(tokens) < 4:
                        raise ValueError(f'{path.name}:{number}: invalid OBJ vertex')
                    vertices.append([float(value) for value in tokens[1:4]])
                elif tokens[0] == 'f':
                    if len(tokens) != 4:
                        raise ValueError(f'{path.name}:{number}: only triangular OBJ faces are supported')
                    face = [int(token.split('/')[0]) for token in tokens[1:]]
                    if any(index == 0 for index in face):
                        raise ValueError('OBJ vertex indices cannot be zero')
                    faces.append([index - 1 if index > 0 else len(vertices) + index for index in face])
        if not np.array_equal(np.asarray(faces, dtype=np.int64), triangles):
            raise ValueError(f'{path.name}: triangle indices/order differ from the fixed template')
    else:
        raise ValueError('Use .obj meshes or .npy vertex arrays; no implicit mesh conversion')
    vertices = np.asarray(vertices, dtype=dtype)
    if vertices.shape != (vertex_count, 3) or not np.isfinite(vertices).all():
        raise ValueError(f'{path.name}: expected finite vertices with shape {(vertex_count, 3)}')
    return vertices


def read_template(path):
    """Only copy topology, pair indices and region masks; never copy existing weights."""
    with np.load(path, allow_pickle=False) as data:
        required = ('face_tri', 'skull_tri', 'face_vertex_count', 'skull_vertex_count',
                    'face_to_skull_ind_new')
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f'Template is missing fields: {sorted(missing)}')
        names = list(required) + [name for name in data.files
                                  if name.startswith('region_') or name in ('face_buf', 'point_buf')]
        arrays = {name: data[name].copy() for name in names}
    for part in ('face', 'skull'):
        count = arrays[f'{part}_vertex_count']
        if count.shape != () or count.dtype.kind not in 'iu' or int(count) < 3:
            raise ValueError(f'{part}_vertex_count must be a positive integer scalar')
        triangles = arrays[f'{part}_tri']
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
                or triangles.dtype.kind not in 'iu' or triangles.min() < 0 or triangles.max() >= int(count)):
            raise ValueError(f'Invalid {part} triangles')
        if np.any((triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2])
                  | (triangles[:, 0] == triangles[:, 2])):
            raise ValueError(f'{part} triangles contain repeated vertex indices')
    pairs = arrays['face_to_skull_ind_new']
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or pairs.dtype.kind not in 'iu'
            or pairs.min() < 0 or pairs[:, 0].max() >= int(arrays['face_vertex_count'])
            or pairs[:, 1].max() >= int(arrays['skull_vertex_count'])):
        raise ValueError('Invalid face/skull pair indices')
    for name, array in arrays.items():
        if name.startswith('region_') and (array.ndim != 1 or not len(array) or array.dtype.kind not in 'iu'
                or array.min() < 0 or array.max() >= int(arrays['face_vertex_count'])
                or len(np.unique(array)) != len(array)):
            raise ValueError(f'Invalid face region indices: {name}')
    face_buf, point_buf = geometry_buffers(arrays['face_tri'], int(arrays['face_vertex_count']),
                                           arrays.get('face_buf'), arrays.get('point_buf'))
    arrays['face_buf'], arrays['point_buf'] = face_buf, point_buf
    return arrays


def geometry_buffers(triangles, vertex_count, face_buf=None, point_buf=None):
    """Validate/generate static normal buffers; padding selects an appended zero normal."""
    triangles = np.asarray(triangles)
    if face_buf is None:
        face_buf = triangles.copy()
    elif face_buf.dtype.kind not in 'iu' or not np.array_equal(face_buf, triangles):
        raise ValueError('face_buf must exactly match face_tri, including triangle order')
    adjacency = [[] for _ in range(vertex_count)]
    for face_index, triangle in enumerate(triangles):
        for vertex in triangle:
            adjacency[int(vertex)].append(face_index)
    degree = max((len(indices) for indices in adjacency), default=0)
    sentinel = len(triangles)
    if point_buf is None:
        width = max(degree, 1)
    else:
        if (point_buf.ndim != 2 or point_buf.shape[0] != vertex_count
                or point_buf.shape[1] < max(degree, 1) or point_buf.dtype.kind not in 'iu'
                or point_buf.min() < 0 or point_buf.max() > sentinel):
            raise ValueError('point_buf must contain zero-based face indices and face-count padding')
        width = point_buf.shape[1]
    expected = np.full((vertex_count, width), sentinel, dtype=np.int64)
    for vertex, indices in enumerate(adjacency):
        expected[vertex, :len(indices)] = indices
    if point_buf is None:
        point_buf = expected
    elif not np.array_equal(np.sort(point_buf, axis=1), np.sort(expected, axis=1)):
        raise ValueError('point_buf must list every incident face once, with only face-count padding')
    # Preserve supplied adjacency order: normal summation order can affect float rounding.
    return face_buf.copy(), point_buf.copy()


def validate_contract(contract):
    expected = {
        'same_topology_and_vertex_order': True,
        'paired_common_coordinate_frame': True,
        'all_samples_prealigned': True,
        'alignment': 'external_preprocessing_no_alignment_in_this_cli',
        'axis_convention': 'x_lateral_y_superior_z_anterior',
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Contract requires {key}={value!r}; align/reindex upstream first')
    if contract.get('units') not in ('mm', 'cm', 'm', 'model_unit'):
        raise ValueError('Contract units must be mm, cm, m or model_unit')
    return {**expected, 'units': contract['units']}


def scaled_pca(samples, components):
    """Exactly the source formula: components_.T * sqrt(explained_variance_)."""
    if samples.ndim != 2 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError('PCA requires at least two finite samples')
    if components < 1 or components > min(len(samples) - 1, samples.shape[1]):
        raise ValueError(f'{components} components require at least {components + 1} nondegenerate samples')
    if np.all(samples == samples[0]):
        raise ValueError(f'Requested {components} components but centered data numerical rank is 0; '
                         'no zero-padding or duplicated basis columns are allowed')
    # Full SVD avoids an implicit randomized solver and reveals actual numerical rank.
    pca = PCA(n_components=None, svd_solver='full')
    pca.fit(samples)
    if any(not np.isfinite(value).all() for value in (
            pca.mean_, pca.components_, pca.singular_values_,
            pca.explained_variance_, pca.explained_variance_ratio_)):
        raise ValueError('PCA statistics overflowed; check coordinate scale or use float64 inputs')
    tolerance = np.finfo(samples.dtype).eps * max(samples.shape) * pca.singular_values_[0]
    rank = int(np.count_nonzero(pca.singular_values_ > tolerance))
    if components > rank:
        raise ValueError(f'Requested {components} components but centered data numerical rank is {rank}; '
                         'no zero-padding or duplicated basis columns are allowed')
    base = pca.components_[:components].T * np.sqrt(pca.explained_variance_[:components])[None, :]
    if not np.isfinite(base).all():
        raise ValueError('PCA basis is non-finite; check coordinate scale or use float64 inputs')
    arrays = {'shapeMU': pca.mean_[None, :], 'shapeBase': base}
    summary = {
        'sample_count': len(samples), 'components': components, 'effective_rank': rank,
        'rank_tolerance': float(tolerance), 'dtype': str(samples.dtype),
        'retained_variance_fraction': float(pca.explained_variance_ratio_[:components].sum()),
        'explained_variance': pca.explained_variance_[:components].tolist(),
        'solver': 'full', 'variance_denominator': len(samples) - 1,
    }
    return arrays, summary


def build(manifest_path, template_path, output, components=200, dtype='float32'):
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    manifest = (tomllib.loads(manifest_path.read_text()) if manifest_path.suffix.lower() == '.toml'
                else json.loads(manifest_path.read_text()))
    contract = validate_contract(manifest.get('contract', {}))
    template = read_template(template_path)
    rows = manifest.get('samples', [])
    if not isinstance(rows, list) or len(rows) < components + 1:
        raise ValueError(f'At least {components + 1} input pairs are needed for {components} components')
    paths = []
    for row in rows:
        if not isinstance(row, dict) or 'face' not in row or 'skull' not in row:
            raise ValueError('Each sample requires face and skull file paths')
        paths.append(tuple((manifest_path.parent / row[key]).resolve() for key in ('face', 'skull')))
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated input pair paths are not independent samples')
    if any((output / name).exists() or (output / name).is_symlink()
           for name in ('fsmm.npz', 'tmm.npz')):
        raise FileExistsError('Output already contains model artifacts; use a new directory')
    face_count, skull_count = (int(template[f'{part}_vertex_count']) for part in ('face', 'skull'))
    pairs = template['face_to_skull_ind_new']
    joint_samples, tissue_samples = [], []
    for index, (face_path, skull_path) in enumerate(paths):
        face = vertices_from_file(face_path, template['face_tri'], face_count, dtype)
        skull = vertices_from_file(skull_path, template['skull_tri'], skull_count, dtype)
        joint_samples.append(np.concatenate((face, skull)).reshape(-1))
        tissue_samples.append((skull[pairs[:, 1]] - face[pairs[:, 0]]).reshape(-1))
        print(f'Validated aligned pair {index + 1}/{len(paths)}', flush=True)
    fsmm, fsmm_summary = scaled_pca(np.asarray(joint_samples), components)
    tmm, tmm_summary = scaled_pca(np.asarray(tissue_samples), components)
    # Outputs deliberately contain no row identifiers, source mesh vertices, texture, or per-person coefficients.
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'fsmm.npz', **fsmm, **template)
    np.savez_compressed(output / 'tmm.npz', **tmm, **template)
    report = {
        'format_version': 1, 'geometry_only': True, 'coordinate_contract': contract,
        'face_skull_layout': 'flatten(concatenate(face_vertices, skull_vertices))',
        'tissue_sign': 'skull_minus_face', 'tissue_layout': 'flatten(skull[pairs[:,1]] - face[pairs[:,0]])',
        'basis_formula': 'components_.T * sqrt(explained_variance_)',
        'coefficient_units': 'standard_deviations; zero is the population mean',
        'mirroring': 'none; any mirrored inputs must already have valid canonical correspondence',
        'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
        'fsmm': fsmm_summary, 'tmm': tmm_summary,
        'source_rows_or_coefficients_saved': False,
        'privacy': 'PCA is not anonymization; omitting individual coefficients is not a privacy guarantee',
        'outputs': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ('fsmm.npz', 'tmm.npz')},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Local TOML with contract and paired sample paths (legacy JSON accepted)')
    parser.add_argument('--template', required=True, help='NPZ topology, counts, pair indices and optional face regions')
    parser.add_argument('--output', required=True, help='New local output directory; existing model artifacts are not overwritten')
    parser.add_argument('--components', type=int, default=200)
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    args = parser.parse_args()
    result = build(args.manifest, args.template, args.output, args.components, args.dtype)
    print(json.dumps({'status': 'complete', 'fsmm_components': result['fsmm']['components'],
                      'tmm_components': result['tmm']['components']}, indent=2))


if __name__ == '__main__':
    main()
