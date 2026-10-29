"""Fit coupled BFSM coefficients to corresponding facial skin."""
import argparse
import json
from pathlib import Path
import numpy as np
from bfsm_core.mesh import read_mesh, write_mesh
from .fit import fit_shape, check_output_inputs
from .model import NativeFace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--flame-face', type=Path, help='5023 FLAME vertices or native facial skin in original order')
    inputs.add_argument('--bfsm-face', type=Path, help='35709 original-order BFSM face, projected through fixed A')
    p.add_argument('--weights', type=Path, help='NPY original-FLAME/native skin reliability; zero excludes missing anatomy')
    p.add_argument('--regularization', type=float, default=1.)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-unobserved', action='store_true')
    args = p.parse_args()
    model = NativeFace()
    filenames = ['face.obj', 'skull.obj', 'coefficients.npz']
    if args.include_unobserved:
        filenames.append('face_prior_completed.obj')
    check_output_inputs(args.output, filenames,
                        [args.flame_face, args.bfsm_face, args.weights, model.source_path])
    target = np.asarray(read_mesh(args.flame_face or args.bfsm_face).vertices)
    if args.bfsm_face:
        target = model.project(target)
    elif len(target) == 5023:
        target = target[model.ids]
    elif target.shape != (model.face_count, 3):
        raise ValueError('FLAME input must match the 5023 template or native skin order')
    weights = None if args.weights is None else np.load(args.weights, allow_pickle=False)
    if weights is not None and weights.shape == (5023,):
        weights = weights[model.ids]
    face, identity, metadata = fit_shape(model.mean[:model.face_count], model.basis[:model.face_count],
                        target, weights, args.regularization, args.iterations)
    _, canonical_skull = model.decode(identity, recenter=False)
    skull = metadata['scale'] * canonical_skull @ np.asarray(metadata['rotation']) + metadata['translation']
    args.output.mkdir(parents=True, exist_ok=True)
    observed = np.ones(model.face_count, bool) if weights is None else weights > 0
    kept = np.flatnonzero(observed)
    reverse = np.full(model.face_count, -1, int)
    reverse[kept] = np.arange(len(kept))
    observed_tri = model.faces[np.all(observed[model.faces], axis=1)]
    if not len(observed_tri):
        raise ValueError('Reliable region contains no complete facial triangle')
    write_mesh(args.output/'face.obj', face[kept], reverse[observed_tri])
    if args.include_unobserved:
        write_mesh(args.output/'face_prior_completed.obj', face, model.faces)
    write_mesh(args.output/'skull.obj', skull, model.arrays['skull_tri'])
    np.savez_compressed(args.output/'coefficients.npz', identity=identity, flame_vertex_ids=model.ids,
                        observed_output_flame_ids=model.ids[kept],
                        scale=np.asarray(metadata['scale']), rotation=np.asarray(metadata['rotation']),
                        translation=np.asarray(metadata['translation']))
    metadata.update({'input':'neutral corresponding facial skin', 'shared_identity_dimensions':len(identity),
                     'scope':'native BFSM facial topology; coupled skull inference; no clinical accuracy claim',
                     'unobserved_regions':'prior completion; coefficients depend only on positive-weight observations'})
    print(json.dumps({k:v for k,v in metadata.items() if k!='history_rmse'}, indent=2))


if __name__ == '__main__':
    main()
