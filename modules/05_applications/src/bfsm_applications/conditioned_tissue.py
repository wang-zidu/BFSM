"""Image-conditioned tissue variations in the original BFSM fitting frame."""
import argparse
from pathlib import Path
import numpy as np

from bfsm_core.mesh import write_mesh
from bfsm_core.paths import check_output_inputs
from .model import NativeFace, NativeTissue, real_array
from .tissue import TissueFaceFitter

VARIANTS = [('s0.75', .75, True), ('s1.25', 1.25, True),
            ('id1', 1., False), ('id2', 1., False)]


def run(reconstruction, output, *, model=None, tissue_model=None,
        source_fsmm=None, source_tmm=None, device='cpu', seed=0):
    """Condition all paired tissue means on one image-derived shared identity.

    The original TMM basis and four 5000-step fits operate in BFSM coordinates.
    Projection and the saved DECA similarity return each face to the input frame.
    """
    reconstruction, output = Path(reconstruction), Path(output)
    native, tissue = NativeFace(model), NativeTissue(tissue_model)
    fitter = TissueFaceFitter(native, tissue, source_fsmm, source_tmm, device)
    filenames = ['before_face.obj', 'skull.obj']
    filenames += [name + suffix for name, _, _ in VARIANTS
                  for suffix in ('.obj', '_parameters.npz')]
    check_output_inputs(output, filenames, [reconstruction, *fitter.input_paths])
    with np.load(reconstruction, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    identity = real_array(data['bfsm_identity'], 'DECA shared BFSM identity')
    native_face, native_skull = native.decode(identity, recenter=False)
    for key, expected in [('canonical_bfsm_face', native_face),
                          ('canonical_bfsm_skull', native_skull)]:
        if not np.array_equal(data[key], expected):
            raise ValueError('DECA reconstruction and selected native BFSM model differ')
    scale = real_array(data['scale'], 'DECA scale')
    rotation = real_array(data['rotation'], 'DECA rotation')
    translation = real_array(data['translation'], 'DECA translation')
    if (scale.ndim != 0 or scale <= 0 or rotation.shape != (3, 3) or
            translation.shape != (3,) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError('DECA similarity must have a positive scale and proper rotation')
    nf = fitter.source_model.face_count
    mean = fitter.fsmm['shapeMU'].reshape(-1)
    basis = fitter.fsmm['shapeBase']
    # Match the original paired-surface tissue demo's float32 geometry input.
    full = (mean + basis @ identity).reshape(-1, 3).astype(np.float32)
    full_face, raw_skull = full[:nf], full[nf:]
    paired_mean = raw_skull[fitter.pairs[:, 1]] - full_face[fitter.pairs[:, 0]]
    centered_skull = raw_skull.astype(np.float64) - fitter.center
    world_skull = scale * native_skull @ rotation + translation
    if not np.array_equal(real_array(data['skull'], 'DECA skull'), world_skull):
        raise ValueError('DECA skull does not match its saved similarity and shared identity')
    rng = np.random.RandomState(seed)
    output.mkdir(parents=True, exist_ok=True)
    write_mesh(output / 'before_face.obj', scale * native_face @ rotation + translation, native.faces)
    write_mesh(output / 'skull.obj', world_skull, native.arrays['skull_tri'])
    for name, thickness, zero in VARIANTS:
        coefficients = (np.zeros(fitter.tissue_basis.shape[1]) if zero else
                        rng.normal(0, .5, fitter.tissue_basis.shape[1]))
        result = fitter.fit(centered_skull, coefficients, thickness,
                            displacement_mean=paired_mean)
        face = scale * (result['face'] + fitter.center) @ rotation + translation
        write_mesh(output / (name + '.obj'), face, native.faces)
        np.savez_compressed(output / (name + '_parameters.npz'),
            tissue_coefficients=coefficients, tissue_scale=np.asarray(thickness),
            baseline_displacement=paired_mean, target_points=result['target_points'],
            source_face=result['source_face_raw'], native_face=result['face'],
            source_skull=raw_skull, source_identity=identity,
            scale=scale, rotation=rotation, translation=translation,
            **{f'fit_{key}': value for key, value in result['fit_coefficients'].items()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction', type=Path, required=True,
                        help='reconstruction.npz from the DECA image frontend')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Matching native FSMM')
    parser.add_argument('--tissue-model', type=Path, help='Matching native TMM')
    parser.add_argument('--source-fsmm', type=Path, help='Original BFSM geometry model')
    parser.add_argument('--source-tmm', type=Path, help='Original paired-vector TMM')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args.reconstruction, args.output, model=args.model, tissue_model=args.tissue_model,
        source_fsmm=args.source_fsmm, source_tmm=args.source_tmm,
        device=args.device, seed=args.seed)


if __name__ == '__main__':
    main()
