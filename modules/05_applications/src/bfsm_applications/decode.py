"""Decode the shared FLAME-face/skull model and independent tissue examples."""
import argparse
from pathlib import Path
import numpy as np
from bfsm_core.mesh import write_mesh
from bfsm_core.paths import check_output_inputs
from bfsm_core.tissue import TISSUE_SCHEDULE
from .model import NativeFace, NativeTissue
from .tissue import TissueFaceFitter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--identity-std', type=float, default=0.)
    p.add_argument('--tissue-std', type=float, default=.5)
    p.add_argument('--thickness', type=float, default=1.)
    p.add_argument('--device', default='cpu', help='Device for the original 5000-step tissue face fit')
    p.add_argument('--source-fsmm', type=Path, help='Original-topology FSMM NPZ matching the native models')
    p.add_argument('--source-tmm', type=Path, help='Original paired-vector TMM NPZ matching the native models')
    args = p.parse_args()
    for name in ('identity_std', 'tissue_std'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(args.thickness) or args.thickness <= 0:
        raise ValueError('Thickness must be finite and positive')
    face_model, tissue_model = NativeFace(), NativeTissue()
    tissue_model.validate_face(face_model)
    fitter = TissueFaceFitter(face_model, tissue_model, args.source_fsmm, args.source_tmm, args.device)
    check_output_inputs(args.output, ['face.obj', 'skull.obj', 'tissue_face.obj', 'coefficients.npz'],
                        fitter.input_paths)
    rng = np.random.default_rng(args.seed)
    identity = rng.normal(0, args.identity_std, face_model.basis.shape[-1])
    tissue = rng.normal(0, args.tissue_std, tissue_model.basis.shape[-1])
    face, skull = face_model.decode(identity)
    result = fitter.fit(skull, tissue, args.thickness)
    if any(not np.isfinite(value).all() for value in (face, skull, result['face'])):
        raise ValueError('Decoded geometry is non-finite; reduce coefficient scale or thickness')
    args.output.mkdir(parents=True, exist_ok=True)
    write_mesh(args.output / 'face.obj', face, face_model.faces)
    write_mesh(args.output / 'skull.obj', skull, face_model.arrays['skull_tri'])
    write_mesh(args.output / 'tissue_face.obj', result['face'], face_model.faces)
    np.savez_compressed(args.output / 'coefficients.npz', identity=identity, tissue=tissue,
                        face_flame_ids=face_model.ids, tissue_flame_ids=tissue_model.ids, anchors=result['anchors'],
                        displacement=result['displacement'], tissue_endpoints=result['endpoints'],
                        target_points=result['target_points'], source_face_raw=result['source_face_raw'],
                        fit_stage_iterations=np.asarray([stage[1] for stage in TISSUE_SCHEDULE]),
                        tissue_scale=np.asarray(args.thickness),
                        **{f'fit_{key}': value for key, value in result['fit_coefficients'].items()})


if __name__ == '__main__':
    main()
