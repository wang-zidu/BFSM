"""Reconstruct one image, then run the original image-conditioned tissue variations."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bfsm_core.paths import ROOT, check_output_inputs
from bfsm_core.parametric import resolve_model_path
from bfsm_core.mesh import read_mesh
from .demo_reconstruction import main as reconstruct, parse_args as reconstruction_args, _validate_output
from .util.io import write_obj_with_colors

VARIANTS = ('s0.75', 's1.25', 'id1', 'id2')


def run(image, output, *, assets=None, fsmm=None, tmm=None, device='cuda:0', seed=0,
        detector='retinaface', crop=True, render=True):
    """Image → shared-coefficient face/skull → four original 5000-step fits.

    Input vertex colors are held fixed, equivalent to the original zero texture
    coefficient change. No FSMM texture PCA is loaded or evaluated.
    """
    image, output = Path(image), Path(output).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    fsmm = resolve_model_path(fsmm or ROOT / 'assets/models/fsmm_200.npz')
    tmm = resolve_model_path(tmm or ROOT / 'assets/models/tmm_200.npz')
    assets = Path(assets or ROOT / 'assets/external/image')
    args = reconstruction_args(['--inputpath', str(image), '--savepath', str(output / 'reconstruction'),
        '--assets', str(assets), '--fsmm', str(fsmm), '--device', device,
        '--seed', str(seed), '--detector', detector, '--iscrop', str(int(crop))])
    _validate_output(args)
    sources = [image, fsmm, tmm]
    if assets.is_dir():
        sources.extend(path for path in assets.rglob('*') if path.is_file())
    stem = image.stem
    reconstruction_names = ['reconstruction.npz', 'network_input.png', stem + '.png', stem + '.npy',
                            stem + '_fsmm_face.obj', stem + '_skull.obj',
                            stem + '_pcaTex.obj', stem + '_extractTex.obj']
    filenames = [Path('reconstruction') / stem / name for name in reconstruction_names]
    filenames += [Path('tissue') / (name + suffix) for name in VARIANTS
                  for suffix in ('.obj', '_parameters.npz', '_tissue_points.ply')]
    filenames += [Path('tissue/input_face.obj'), Path('tissue/input_skull.obj'),
                  Path('tissue/reconstruction_frame.npz'), Path('tissue/canonical/input_face.obj'),
                  Path('tissue/canonical/input_skull.obj')]
    filenames += [Path('tissue/canonical') / (name + suffix) for name in VARIANTS
                  for suffix in ('.obj', '_parameters.npz', '_tissue_points.ply')]
    if render:
        filenames.extend([Path('tissue/change_tissue.png'), Path('summary.png')])
    for filename in filenames:
        if output not in (output / filename).resolve().parents:
            raise ValueError('An image/tissue export must remain inside the output directory')
    check_output_inputs(output, filenames, sources)
    reconstruct(args)
    result = _fit_reconstruction(output / 'reconstruction' / stem, output / 'tissue',
                                 fsmm=fsmm, tmm=tmm, device=device, seed=seed, render=render)
    if render:
        from .preview import compose
        compose(output / 'reconstruction' / stem / (stem + '.png'),
                output / 'tissue/change_tissue.png', output / 'summary.png')
    return result


def _fit_reconstruction(reconstruction, output, *, fsmm, tmm, device='cuda:0', seed=0, render=True):
    """Fit verified reconstructed pairs in canonical coordinates, then restore image SRT."""
    import shutil
    import torch
    import trimesh
    from bfsm_core.model import FaceSkullModel
    from bfsm_models.tissue import main as change_tissue, render_grid
    from bfsm_models._tissue import extract_vertex_colors_from_obj
    reconstruction, output = Path(reconstruction), Path(output).resolve()
    stem = reconstruction.name
    face, skull = [reconstruction / (stem + suffix) for suffix in ('_fsmm_face.obj', '_skull.obj')]
    arrays_path = reconstruction / 'reconstruction.npz'
    sources = [face, skull, arrays_path, fsmm, tmm]
    names = ['input_face.obj', 'input_skull.obj', 'reconstruction_frame.npz']
    names += [name + suffix for name in VARIANTS
              for suffix in ('.obj', '_parameters.npz', '_tissue_points.ply')]
    names += ['canonical/input_face.obj', 'canonical/input_skull.obj']
    names += ['canonical/' + name + suffix for name in VARIANTS
              for suffix in ('.obj', '_parameters.npz', '_tissue_points.ply')]
    if render:
        names.append('change_tissue.png')
    for name in names:
        if output not in (output / name).resolve().parents:
            raise ValueError('A tissue export must remain inside its output directory')
    check_output_inputs(output, names, sources)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    model = FaceSkullModel(fsmm, device='cpu')
    coefficients = {}
    for key, shape in [('id', (1, model.components)), ('angle', (1, 3)),
                       ('scale', (1, 1)), ('trans', (1, 3))]:
        value = np.asarray(arrays['coeff_' + key])
        if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Reconstruction coefficients must be finite and match the source FSMM')
        coefficients[key] = torch.as_tensor(value, dtype=torch.float32)
        if not torch.isfinite(coefficients[key]).all():
            raise ValueError('Reconstruction coefficients must remain finite in float32')
    scale = float(coefficients['scale'].item())
    if scale <= 0:
        raise ValueError('Reconstruction similarity scale must be positive')
    with torch.no_grad():
        decoded = model.decode(coefficients)
        rotation = model.rotation(coefficients['angle']).numpy()[0].astype(np.float64)
        identity = dict(coefficients, angle=torch.zeros((1, 3)), scale=torch.ones((1, 1)),
                        trans=torch.zeros((1, 3)))
        canonical_decoded = model.decode(identity)
    translation = coefficients['trans'].numpy()[0].astype(np.float64)
    inverse = np.linalg.inv(rotation)
    meshes = [read_mesh(face), read_mesh(skull)]
    canonical = []
    for mesh, predicted, expected, triangles in zip(meshes, decoded, canonical_decoded,
                                                   (model.arrays['face_tri'], model.arrays['skull_tri'])):
        if (not np.array_equal(mesh.faces, triangles) or
                not np.allclose(mesh.vertices, predicted.numpy()[0], atol=2e-6, rtol=0)):
            raise ValueError('Reconstructed face/skull do not match their FSMM coefficients')
        vertices = (mesh.vertices - translation) @ inverse / scale
        if not np.allclose(vertices, expected.numpy()[0], atol=2e-6, rtol=0):
            raise ValueError('Reconstruction inverse similarity does not match canonical FSMM geometry')
        canonical.append(vertices)
    colors = extract_vertex_colors_from_obj(face)
    if colors.shape != canonical[0].shape or not np.isfinite(colors).all():
        raise ValueError('Reconstructed face colors must match the tissue face vertices')
    canonical_dir = output / 'canonical'
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_face, canonical_skull = canonical_dir / 'input_face.obj', canonical_dir / 'input_skull.obj'
    write_obj_with_colors(str(canonical_face), canonical[0], meshes[0].faces, colors)
    write_obj_with_colors(str(canonical_skull), canonical[1], meshes[1].faces,
                          np.full_like(canonical[1], .75))
    change_tissue(SimpleNamespace(face_obj_path=str(canonical_face), skull_obj_path=str(canonical_skull),
        output_dir=str(canonical_dir), fsmm=str(fsmm), tmm=str(tmm), face_tri=None,
        device=device, seed=seed, texture_variations=False, variant=None,
        no_render=True, legacy_cache=False))
    shutil.copy2(face, output / 'input_face.obj')
    shutil.copy2(skull, output / 'input_skull.obj')
    np.savez_compressed(output / 'reconstruction_frame.npz', scale=np.asarray(scale),
                        rotation=rotation, translation=translation)
    for name in VARIANTS:
        mesh = read_mesh(canonical_dir / (name + '.obj'))
        image_vertices = scale * mesh.vertices @ rotation + translation
        write_obj_with_colors(str(output / (name + '.obj')), image_vertices, mesh.faces, colors)
        with np.load(canonical_dir / (name + '_parameters.npz'), allow_pickle=False) as archive:
            fitted = {key: archive[key] for key in archive.files}
        image_target = scale * fitted['target_points'] @ rotation + translation
        np.savez_compressed(output / (name + '_parameters.npz'), **fitted,
            image_scale=np.asarray(scale), image_rotation=rotation, image_translation=translation,
            image_face_vertices=image_vertices, image_target_points=image_target)
        trimesh.PointCloud(image_target).export(output / (name + '_tissue_points.ply'))
    if render:
        render_grid([output / 'input_face.obj'] + [output / (name + '.obj') for name in VARIANTS],
                    output / 'change_tissue.png', device, neutral=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='One input photograph or artwork')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--fsmm', type=Path)
    parser.add_argument('--tmm', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detector', choices=['retinaface', 'mtcnn'], default='retinaface')
    parser.add_argument('--no-crop', action='store_true', help='Input must already be 224 by 224 pixels')
    parser.add_argument('--no-render', action='store_true', help='Skip only the tissue comparison grid')
    args = parser.parse_args(argv)
    run(args.image, args.output, assets=args.assets, fsmm=args.fsmm, tmm=args.tmm,
        device=args.device, seed=args.seed,
        detector=args.detector, crop=not args.no_crop, render=not args.no_render)


if __name__ == '__main__':
    main()
