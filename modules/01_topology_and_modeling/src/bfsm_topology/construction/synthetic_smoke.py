"""Exercise the real construction CLI using labelled, non-patient synthetic cube pairs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.decomposition import PCA

from .build import geometry_buffers, read_template, scaled_pca, vertices_from_file


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new smoke-test directory')
    output.mkdir(parents=True)
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                          [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    pairs = np.column_stack((np.arange(8), np.arange(8)))
    template_path = output / 'synthetic_template.npz'
    np.savez_compressed(template_path, face_tri=triangles, skull_tri=triangles,
                        face_vertex_count=np.array(8), skull_vertex_count=np.array(8),
                        face_to_skull_ind_new=pairs, region_up=np.array([2, 3, 6, 7]))
    rng = np.random.default_rng(20261027)
    rows = []
    for index in range(6):
        for part, scale in [('face', 1.25), ('skull', .85)]:
            value = scale * vertices + rng.normal(0, .04, vertices.shape)
            path = output / f'synthetic_{index}_{part}.npy'
            np.save(path, value)
        rows.append({part: f'synthetic_{index}_{part}.npy' for part in ('face', 'skull')})
    manifest_path = output / 'synthetic_manifest.toml'
    manifest_path.write_text(
        '[contract]\n'
        'same_topology_and_vertex_order = true\n'
        'paired_common_coordinate_frame = true\n'
        'all_samples_prealigned = true\n'
        'units = "model_unit"\n'
        'alignment = "external_preprocessing_no_alignment_in_this_cli"\n'
        'axis_convention = "x_lateral_y_superior_z_anterior"\n'
        + ''.join(f'\n[[samples]]\nface = "{row["face"]}"\nskull = "{row["skull"]}"\n'
                  for row in rows))
    results = {'fixture': 'six randomly perturbed synthetic cube pairs; no patient data', 'checks': []}
    base_command = [sys.executable, '-m', 'bfsm_topology.construction.build', '--manifest', str(manifest_path),
                    '--template', str(template_path)]
    for dtype in ('float32', 'float64'):
        model_folder = output / dtype
        completed = subprocess.run(base_command + ['--components', '3', '--dtype', dtype,
                                   '--output', str(model_folder)], text=True, capture_output=True)
        (output / f'{dtype}_cli.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'Real construction CLI failed; see {dtype}_cli.log')
        fsmm_rows, tmm_rows = [], []
        for row in rows:
            face, skull = (np.load(output / row[part]).astype(dtype) for part in ('face', 'skull'))
            fsmm_rows.append(np.concatenate((face, skull)).reshape(-1))
            tmm_rows.append((skull[pairs[:, 1]] - face[pairs[:, 0]]).reshape(-1))
        for model, samples in [('fsmm', fsmm_rows), ('tmm', tmm_rows)]:
            # These are the source construction equations, independently evaluated.
            original = PCA(n_components=len(samples))
            original.fit(np.asarray(samples))
            expected_base = original.components_.T * np.sqrt(original.explained_variance_)[None, :]
            with np.load(model_folder / f'{model}.npz', allow_pickle=False) as archive:
                mean_error = float(np.max(np.abs(archive['shapeMU'] - original.mean_[None, :])))
                base_error = float(np.max(np.abs(archive['shapeBase'] - expected_base[:, :3])))
                assert mean_error == 0 and base_error == 0, (dtype, model, mean_error, base_error)
                assert not any(key.startswith('tex') or key in ('coefficients', 'samples') for key in archive.files)
                assert np.array_equal(archive['face_buf'], triangles)
                expected_face, expected_point = geometry_buffers(triangles, 8)
                assert np.array_equal(archive['face_buf'], expected_face)
                assert np.array_equal(archive['point_buf'], expected_point)
            results['checks'].append({'dtype': dtype, 'model': model,
                                      'mean_max_abs_error': mean_error, 'basis_max_abs_error': base_error})
    failed = subprocess.run(base_command + ['--components', '200', '--output', str(output / 'must_not_exist')],
                            text=True, capture_output=True)
    assert failed.returncode != 0 and not (output / 'must_not_exist').exists()
    results['reject_200_from_six_samples'] = True
    try:
        scaled_pca(np.ones((6, 24), dtype=np.float64), 3)
    except ValueError as error:
        assert 'rank is 0' in str(error)
    else:
        raise AssertionError('Constant data must not produce fake basis columns')
    results['reject_rank_zero'] = True
    mesh_path = output / 'synthetic_face.obj'
    with mesh_path.open('w') as stream:
        for vertex in vertices:
            stream.write('v ' + ' '.join(map(str, vertex)) + '\n')
        for tri in triangles:
            stream.write('f ' + ' '.join(str(value + 1) for value in tri) + '\n')
    assert np.array_equal(vertices_from_file(mesh_path, triangles, 8, 'float64'), vertices)
    try:
        vertices_from_file(mesh_path, triangles[::-1], 8, 'float64')
    except ValueError as error:
        assert 'triangle indices/order' in str(error)
    else:
        raise AssertionError('Changed topology must be rejected')
    results['reject_triangle_order_mismatch'] = True
    loaded = read_template(output / 'float32/fsmm.npz')
    assert np.array_equal(loaded['face_buf'], triangles)
    _, expected_point = geometry_buffers(triangles, 8)
    assert np.array_equal(loaded['point_buf'], expected_point)
    results['generated_geometry_buffers_round_trip'] = True
    reversed_adjacency = expected_point[:, ::-1].copy()
    _, preserved = geometry_buffers(triangles, 8, triangles, reversed_adjacency)
    assert np.array_equal(preserved, reversed_adjacency)
    results['supplied_adjacency_order_preserved'] = True
    corrupt = expected_point.copy()
    corrupt[0, 0] = len(triangles)
    try:
        geometry_buffers(triangles, 8, triangles, corrupt)
    except ValueError as error:
        assert 'every incident face' in str(error)
    else:
        raise AssertionError('Missing incident face must be rejected')
    results['reject_incorrect_normal_buffer'] = True
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == '__main__':
    main()
