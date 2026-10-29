"""Face-only weighted linear-shape/similarity fitting; no ear or symmetry loss."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from bfsm_core.mesh import read_mesh, write_mesh
from .model import NativeFace, real_array, triangles as validate_triangles



def check_output_inputs(output, filenames, inputs):
    """Reject exports that would overwrite an observation or model asset."""
    inputs = [Path(path).resolve() for path in inputs if path is not None]
    for filename in filenames:
        destination = (output / filename).resolve()
        for source in inputs:
            same_file = (destination.exists() and source.exists()
                         and destination.samefile(source))
            if destination == source or same_file:
                raise ValueError(f'Output would overwrite an input asset: {destination}')


def similarity(source, target, weights):
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    x, y = source - a, target - b
    u, singular, vt = np.linalg.svd((x * weights[:, None]).T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = np.sum(singular * sign) / np.sum(weights[:, None] * x * x)
    return scale, rotation, b - scale * a @ rotation


def fit_shape(mean, basis, target, weights=None, regularization=.001, iterations=30):
    """Alternate a full similarity and ridge identity fit in canonical units.

    The prior is scaled by the average identity Hessian diagonal. Zero weights
    exclude missing structure entirely; there is no mirroring or full-head term.
    """
    mean, basis, target = map(np.asarray, (mean, basis, target))
    if mean.shape != target.shape or basis.shape[:2] != mean.shape:
        raise ValueError('Corresponding three-dimensional vertices are required')
    weights = np.ones(len(mean)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(mean),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be finite, nonnegative and match the face')
    valid = (weights > 0) & np.isfinite(target).all(1)
    if valid.sum() < 6 or np.linalg.matrix_rank(target[valid] - target[valid].mean(0)) < 2:
        raise ValueError('At least six non-collinear reliable face points are required')
    if regularization <= 0 or iterations < 1:
        raise ValueError('Positive regularization and iteration count are required')
    m, b, y, w = mean[valid], basis[valid], target[valid], weights[valid]
    flat = b.reshape(-1, b.shape[-1])
    wf = np.repeat(w, 3)
    hessian = flat.T @ (wf[:, None] * flat)
    ridge = regularization * np.trace(hessian) / hessian.shape[0]
    factor = cho_factor(hessian + ridge * np.eye(hessian.shape[0]))
    coefficient = np.zeros(b.shape[-1])
    losses = []
    for _ in range(iterations):
        current = m + b @ coefficient
        scale, rotation, translation = similarity(current, y, w)
        canonical_target = ((y - translation) @ rotation.T) / scale
        coefficient = cho_solve(factor, flat.T @ (wf * (canonical_target - m).reshape(-1)))
        fitted = scale * (m + b @ coefficient) @ rotation + translation
        losses.append(float(np.sqrt(np.average(np.sum((fitted-y)**2, axis=1), weights=w))))
    canonical = mean + basis @ coefficient
    transformed = scale * canonical @ rotation + translation
    return transformed, coefficient, {'scale': float(scale), 'rotation': rotation.tolist(),
            'translation': translation.tolist(), 'weighted_rmse': losses[-1],
            'iterations': iterations, 'regularization': regularization,
            'ridge': float(ridge), 'valid_points': int(valid.sum()),
            'excluded_points': int((~valid).sum()), 'history_rmse': losses}


def main():
    p = argparse.ArgumentParser(description='BFSM face to a fitted FLAME face-only output')
    p.add_argument('--face', type=Path, required=True, help='Original-order 35709-point BFSM face OBJ')
    p.add_argument('--flame-neutral', type=Path, required=True,
                   help='Locally prepared licensed FLAME NPZ: v_template, shapedirs, f')
    p.add_argument('--weights', type=Path, help='NPY weights in original 5023 FLAME order or native face order; zero = missing')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regularization', type=float, default=.001)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--include-unobserved', action='store_true',
                   help='Also export prior-completed full face; never treat it as observed anatomy')
    args = p.parse_args()
    model = NativeFace()
    filenames = ['flame_face.obj', 'bfsm_projected_face.obj', 'flame_fit.npz']
    if args.include_unobserved:
        filenames.append('flame_prior_completed.obj')
    check_output_inputs(args.output, filenames,
                        [args.face, args.flame_neutral, args.weights, model.source_path])
    old = np.asarray(read_mesh(args.face).vertices)
    target = model.project(old)
    weights = None if args.weights is None else np.load(args.weights, allow_pickle=False)
    if weights is not None and weights.shape == (5023,):
        weights = weights[model.ids]
    with np.load(args.flame_neutral, allow_pickle=False) as archive:
        mean = real_array(archive['v_template'], 'FLAME template')
        directions = real_array(archive['shapedirs'], 'FLAME identity basis')
        if (mean.shape != (5023, 3) or directions.ndim != 3 or
                directions.shape[:2] != mean.shape or directions.shape[2] < 300):
            raise ValueError('Expected canonical FLAME2020 template and identity shapedirs')
        basis = directions[:, :, :300]
        full_to_sub = np.full(5023, -1, int)
        full_to_sub[model.ids] = np.arange(len(model.ids))
        triangles = validate_triangles(archive['f'], 5023, 'FLAME topology')
        subtri = full_to_sub[triangles[np.all(full_to_sub[triangles] >= 0, axis=1)]]
        if not np.array_equal(subtri, model.faces):
            raise ValueError('FLAME topology does not match the derived model version')
    result, coefficient, metadata = fit_shape(mean[model.ids], basis[model.ids], target,
                            weights, args.regularization, args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    observed = np.ones(len(model.ids), bool) if weights is None else weights > 0
    kept = np.flatnonzero(observed)
    reverse = np.full(len(model.ids), -1, int)
    reverse[kept] = np.arange(len(kept))
    observed_tri = model.faces[np.all(observed[model.faces], axis=1)]
    if len(observed_tri) == 0:
        raise ValueError('Reliable face points contain no complete output triangle')
    write_mesh(args.output/'flame_face.obj', result[kept], reverse[observed_tri])
    if args.include_unobserved:
        write_mesh(args.output/'flame_prior_completed.obj', result, model.faces)
    write_mesh(args.output/'bfsm_projected_face.obj', target, model.faces)
    np.savez_compressed(args.output/'flame_fit.npz', identity=coefficient, flame_vertex_ids=model.ids,
                        observed_output_flame_ids=model.ids[kept],
                        weights=np.ones(len(model.ids)) if weights is None else weights,
                        scale=np.asarray(metadata['scale']), rotation=np.asarray(metadata['rotation']),
                        translation=np.asarray(metadata['translation']))
    metadata.update({'scope':'facial skin only; no ears, symmetry, expression or jaw-pose optimization',
                     'input':'BFSM reconstructed face projected through fixed template map',
                     'missing_region_policy':'zero weight excludes observation; prior-completed output is not evidence of anatomy'})
    print(json.dumps({key: value for key, value in metadata.items() if key != 'history_rmse'}, indent=2))


if __name__ == '__main__':
    main()
