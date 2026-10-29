"""DECA expression/pose reconstruction, coupled BFSM fitting and image overlays."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp

from bfsm_core.mesh import write_mesh
from .fit import fit_shape, check_output_inputs
from .model import NativeFace


def crop_image(image, landmarks, resolution=224, scale=1.25):
    """DECA's 68-landmark TestData crop, RGB [0,1], CHW."""
    landmarks = np.asarray(landmarks)
    if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
        raise ValueError('Expected finite 68x2 image-pixel landmarks')
    left, top = landmarks.min(0)
    right, bottom = landmarks.max(0)
    old_size = (right-left+bottom-top)/2*1.1
    size = int(old_size*scale)
    if size < 2:
        raise ValueError('Landmark bounding box is degenerate')
    center = np.array([(right+left)/2, (bottom+top)/2])
    source = np.array([center+[-size/2, -size/2], center+[-size/2, size/2],
                       center+[size/2, -size/2]])
    destination = np.array([[0, 0], [0, resolution-1], [resolution-1, 0]])
    transform = estimate_transform('similarity', source, destination)
    cropped = warp(image/255., transform.inverse, output_shape=(resolution, resolution))
    return cropped.transpose(2, 0, 1).astype(np.float32), transform.params


def encode(image_tensor, deca_root, checkpoint, device):
    """Run the original external ResnetEncoder with strictly matching E_flame weights."""
    deca_root = Path(deca_root).resolve()
    if not (deca_root/'decalib/models/encoders.py').is_file():
        raise ValueError('deca-root must contain the licensed original decalib source')
    sys.path.insert(0, str(deca_root))
    try:
        encoder_class = importlib.import_module('decalib.models.encoders').ResnetEncoder
        encoder = encoder_class(outsize=236).to(device).eval()
    finally:
        sys.path.pop(0)
    # The checkpoint comes from the original DECA distribution.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    encoder.load_state_dict(state['E_flame'], strict=True)
    with torch.inference_mode():
        parameters = encoder(torch.as_tensor(image_tensor[None], device=device)).cpu().numpy()[0]
    if parameters.shape != (236,) or not np.isfinite(parameters).all():
        raise ValueError('Unexpected DECA encoder output')
    return dict(zip(['shape', 'texture', 'expression', 'pose', 'camera', 'light'],
                    np.split(parameters, [100, 150, 200, 206, 209])))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--deca-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Default: deca-root/data/deca_model.tar')
    parser.add_argument('--flame-neutral', type=Path, required=True,
                        help='Prepared matching FLAME NPZ with identity, expression and skinning arrays')
    parser.add_argument('--model', type=Path, help='Derived FLAME-face BFSM NPZ')
    parser.add_argument('--landmarks', type=Path, help='Optional 68x2 NPY/TXT in original image pixels')
    parser.add_argument('--weights', type=Path, help='Reliable-region NPY weights: native face or FLAME5023; zero excludes missing anatomy')
    parser.add_argument('--include-unobserved', action='store_true',
                        help='Also export explicitly named prior-completed full faces; never observed anatomy')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--regularization', type=float, default=1.,
                        help='Identity ridge regularization (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = NativeFace(args.model) if args.model else NativeFace()
    checkpoint = args.checkpoint or args.deca_root/'data/deca_model.tar'
    filenames = ['deca_face.obj', 'bfsm_face.obj', 'bfsm_skull.obj', 'reconstruction.png',
                 'network_input.png', 'reconstruction.npz']
    if args.include_unobserved:
        filenames += ['deca_prior_completed_face.obj', 'bfsm_prior_completed_face.obj']
    check_output_inputs(args.output, filenames, [args.image, args.landmarks, args.weights,
                        args.flame_neutral, checkpoint, model.source_path])
    image = np.asarray(imread(args.image))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise ValueError('Expected an 8-bit RGB/grayscale image')
    image = image[:, :, :3]
    if args.landmarks:
        landmarks = np.load(args.landmarks, allow_pickle=False) if args.landmarks.suffix == '.npy' else np.loadtxt(args.landmarks)
        landmark_source = 'explicit image-pixel 68-point landmarks'
    else:
        import face_alignment
        detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                               flip_input=False, device=args.device)
        detected = detector.get_landmarks_from_image(image)
        if detected is None or len(detected) != 1:
            raise ValueError('Exactly one detected face is required; supply explicit landmarks otherwise')
        landmarks = detected[0]
        landmark_source = 'FAN TWO_D, flip_input=False; original DECA detector convention'
    cropped, transform = crop_image(image, landmarks)
    parameters = encode(cropped, args.deca_root, checkpoint, args.device)
    with np.load(args.flame_neutral, allow_pickle=False) as archive:
        flame = {key: archive[key] for key in archive.files}
        template, basis, full_faces = flame['v_template'], flame['shapedirs'], flame['f']
    if template.shape != (5023, 3) or basis.shape[:2] != template.shape or basis.shape[2] < 100:
        raise ValueError('Expected the DECA-compatible canonical FLAME model')
    inverse = np.full(5023, -1, dtype=int)
    inverse[model.ids] = np.arange(len(model.ids))
    faces = inverse[full_faces[np.all(inverse[full_faces] >= 0, axis=1)]]
    if not np.array_equal(faces, model.faces):
        raise ValueError('FLAME topology differs from the derived BFSM face model')
    from .deca_geometry import decode_flame, project_pixels
    posed = decode_flame(flame, parameters['shape'], parameters['expression'], parameters['pose'])
    project_pixels(posed, parameters['camera'])
    weights = None if args.weights is None else np.load(args.weights, allow_pickle=False)
    if weights is not None and weights.shape == (5023,):
        weights = weights[model.ids]
    target = posed[model.ids]
    fitted, identity, fit = fit_shape(model.mean[:model.face_count],
                     model.basis[:model.face_count], target, weights,
                     args.regularization, args.iterations)
    canonical_face, canonical_skull = model.decode(identity, recenter=False)
    scale, rotation, translation = fit['scale'], np.asarray(fit['rotation']), np.asarray(fit['translation'])
    skull = scale*canonical_skull@rotation+translation
    observed = np.ones(model.face_count, dtype=bool) if weights is None else weights > 0
    kept = np.flatnonzero(observed)
    inverse_observed = np.full(model.face_count, -1, dtype=int)
    inverse_observed[kept] = np.arange(len(kept))
    observed_triangles = model.faces[np.all(observed[model.faces], axis=1)]
    if len(observed_triangles) == 0:
        raise ValueError('Reliable face points contain no complete output triangle')
    observed_triangles = inverse_observed[observed_triangles]
    args.output.mkdir(parents=True, exist_ok=True)
    write_mesh(args.output/'deca_face.obj', target[kept], observed_triangles)
    write_mesh(args.output/'bfsm_face.obj', fitted[kept], observed_triangles)
    if args.include_unobserved:
        write_mesh(args.output/'deca_prior_completed_face.obj', target, model.faces)
        write_mesh(args.output/'bfsm_prior_completed_face.obj', fitted, model.faces)
    write_mesh(args.output/'bfsm_skull.obj', skull, model.arrays['skull_tri'])
    imsave(args.output/'network_input.png', np.rint(cropped.transpose(1, 2, 0)*255).astype(np.uint8))
    np.savez_compressed(args.output/'reconstruction.npz', **{'deca_'+key: value for key, value in parameters.items()},
                        bfsm_identity=identity, scale=scale, rotation=rotation, translation=translation,
                        posed_flame_vertices=posed,
                        face_pixels=project_pixels(fitted, parameters['camera']),
                        skull_pixels=project_pixels(skull, parameters['camera']),
                        face=fitted, skull=skull, canonical_bfsm_face=canonical_face,
                        canonical_bfsm_skull=canonical_skull, crop_transform=transform,
                        landmarks=landmarks, flame_vertex_ids=model.ids,
                        observed_output_flame_ids=model.ids[kept],
                        reliable_weights=np.ones(model.face_count) if weights is None else weights)
    from .deca_preview import render_preview
    render_preview(image, transform, parameters['camera'], target[kept], fitted[kept],
                   observed_triangles, skull, model.arrays['skull_tri'],
                   args.output/'reconstruction.png')
    print(json.dumps({'weighted_rmse': fit['weighted_rmse'], 'valid_points': fit['valid_points'],
                      'expression_and_pose_applied': True}, indent=2))


if __name__ == '__main__':
    main()
