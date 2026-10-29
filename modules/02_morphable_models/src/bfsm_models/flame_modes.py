"""Display FLAME-topology FSMM modes and original-schedule fitted TMM faces."""
import argparse
from pathlib import Path

import numpy as np

from bfsm_core.mesh import write_mesh
from bfsm_core.paths import check_output_inputs
from bfsm_core.tissue import TISSUE_SCHEDULE
from bfsm_models.modes import mode_states, static_figure, interactive_figure


def generate(fsmm_path, tmm_path, output, sigma=3., *, source_fsmm_path=None,
             source_tmm_path=None, device='cpu'):
    from bfsm_applications.model import NativeFace, NativeTissue
    from bfsm_applications.tissue import TissueFaceFitter

    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    fsmm, tmm = NativeFace(fsmm_path), NativeTissue(tmm_path)
    fitter = TissueFaceFitter(fsmm, tmm, source_fsmm_path, source_tmm_path, device)
    if fsmm.basis.shape[-1] < 3 or tmm.basis.shape[-1] < 2:
        raise ValueError('Need at least three FSMM and two TMM components')
    filenames = []
    for kind, count in (('fsmm', 3), ('tmm', 2)):
        filenames += [f'{kind}/{part}_tri.npy' for part in ('face', 'skull')]
        filenames += [f'{kind}/{kind}_modes.png', f'{kind}/{kind}_modes.pdf',
                      f'{kind}/{kind}_interactive.html', f'{kind}/PLOTLY_LICENSE.txt']
        filenames += [f'{kind}/{kind}/{name}/{file}'
                      for name, _, _ in mode_states(count, sigma)
                      for file in ('face.obj', 'skull.obj', 'state.npz')]
    check_output_inputs(output, filenames, fitter.input_paths)
    _, fixed_skull = fsmm.decode()
    records = []
    for kind, count, basis in (('fsmm', 3, fsmm.basis), ('tmm', 2, fitter.tissue_basis)):
        folder = output / kind
        folder.mkdir(parents=True)
        np.save(folder / 'face_tri.npy', fsmm.faces, allow_pickle=False)
        np.save(folder / 'skull_tri.npy', fsmm.arrays['skull_tri'], allow_pickle=False)
        for name, pc, value in mode_states(count, sigma):
            coefficients = np.zeros(basis.shape[-1], dtype=basis.dtype if kind == 'tmm' else float)
            if pc is not None:
                coefficients[pc] = value
            extra = {}
            if kind == 'fsmm':
                face, skull = fsmm.decode(coefficients)
            else:
                print(f'TMM {name}: {value:+g} sigma; original 5000-step fit before FLAME projection', flush=True)
                result = fitter.fit(fixed_skull, coefficients)
                face, skull = result['face'], result['skull']
                extra = dict(source_face_raw=result['source_face_raw'],
                             target_points=result['target_points'],
                             tissue_endpoints=result['endpoints'],
                             tissue_flame_ids=tmm.ids,
                             fit_stage_iterations=np.asarray([stage[1] for stage in TISSUE_SCHEDULE]),
                             **{f'fit_{key}': val for key, val in result['fit_coefficients'].items()})
                records.append(dict(state=name, history=result['history'],
                                    target_rms_coordinate_error=result['target_rms_coordinate_error']))
            if not np.isfinite(face).all() or not np.isfinite(skull).all():
                raise ValueError('Non-finite generated geometry')
            state = folder / kind / name
            write_mesh(state / 'face.obj', face, fsmm.faces)
            write_mesh(state / 'skull.obj', skull, fsmm.arrays['skull_tri'])
            np.savez_compressed(state / 'state.npz', face=face, skull=skull,
                                coefficients=coefficients, **extra)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fsmm', default='assets/models/flame_fsmm_200.npz')
    parser.add_argument('--tmm', default='assets/models/flame_tmm_200.npz')
    parser.add_argument('--source-fsmm', help='Original-topology FSMM NPZ matching the FLAME models')
    parser.add_argument('--source-tmm', help='Original paired-vector TMM NPZ matching the FLAME models')
    parser.add_argument('--device', default='cpu', help='Device for each original 5000-step TMM face fit')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sigma', type=float, default=3.)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    generate(args.fsmm, args.tmm, args.output, args.sigma,
             source_fsmm_path=args.source_fsmm, source_tmm_path=args.source_tmm, device=args.device)
    if not args.generate_only:
        for kind in ('fsmm', 'tmm'):
            static_figure(Path(args.output) / kind, kind)
            interactive_figure(Path(args.output) / kind, kind)


if __name__ == '__main__':
    main()
