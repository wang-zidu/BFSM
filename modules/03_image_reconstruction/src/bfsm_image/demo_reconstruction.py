import argparse
from pathlib import Path
from bfsm_core.paths import ROOT
import cv2
import os
import numpy as np
import torch
from PIL import Image

from .face_box import face_box
from .model.recon import face_model
from .util.preprocess import get_data_path
from .util.io import visualize
def _validate_output(args):
    output = Path(args.savepath).resolve()
    inputs = ([args.inputpath] if Path(args.inputpath).is_file()
              else get_data_path(args.inputpath))
    stems = [Path(path).stem for path in inputs]
    if len(stems) != len(set(stems)):
        raise ValueError('Input images must have distinct filename stems to avoid overwriting results')
    if any(stem in {'.', '..', ''} for stem in stems):
        raise ValueError('Input images must have safe, nonempty filename stems')
    # Match the decoder's legacy model-path resolution as well as the explicit path.
    from bfsm_core.parametric import resolve_model_path
    sources = [Path(path).resolve() for path in [*inputs, args.fsmm,
                                               resolve_model_path(args.fsmm)]]
    assets = Path(args.assets)
    if assets.is_dir():
        sources.extend(path.resolve() for path in assets.rglob('*') if path.is_file())
    if any(path == output or output in path.parents for path in sources):
        raise ValueError("Choose an output directory that does not contain input images or model assets")
    for stem in stems:
        destination = (output / stem).resolve()
        if output not in destination.parents:
            raise ValueError('An image result directory must remain inside the output directory')
        names = ['reconstruction.npz', 'network_input.png', stem + '.png', stem + '.npy',
                 stem + '_fsmm_face.obj', stem + '_skull.obj']
        if args.extractTex:
            names.append(stem + '_extractTex.obj')
        if args.useTex:
            names.append(stem + '_pcaTex.obj')
        for name in names:
            target = (destination / name).resolve()
            if output not in target.parents:
                raise ValueError('An image export must remain inside the output directory')
            for source in sources:
                if target == source or (target.exists() and source.exists()
                                        and target.samefile(source)):
                    raise ValueError(f'Output would overwrite an input asset: {target}')


def main(args):
    _validate_output(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    im_path = [args.inputpath] if Path(args.inputpath).is_file() else get_data_path(args.inputpath)

    for i in range(len(im_path)):
        print(i, im_path[i])
        im = Image.open(im_path[i]).convert('RGB')
        trans_params, im_tensor = facebox_detector(im)

        recon_model.input_img = im_tensor.to(args.device)
        results = recon_model.forward()

        name = Path(im_path[i]).stem
        destination = Path(args.savepath) / name
        destination.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
        arrays.update({"coeff_" + k: v for k, v in results["fsmm_coefficients"].items()})
        np.savez_compressed(destination / "reconstruction.npz", **arrays)
        Image.fromarray((im_tensor[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(destination / "network_input.png")
        my_visualize = visualize(results, args)
        my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), str(destination), name)
        from .util.io import write_obj_with_colors
        fitted = results['fsmm_face'][0].copy()
        fitted[:, 2] = 10 - fitted[:, 2]
        write_obj_with_colors(str(destination / (name + '_fsmm_face.obj')), fitted, results['tri'], results['face_texture'][0])



def parse_bool(value):
    normalized = value.lower()
    if normalized in ('true', '1'):
        return True
    if normalized in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError('Expected true, false, 1 or 0')


def parse_args(argv=None):
    root = ROOT
    bundled_image = root / 'data' / 'input' / 'image_portrait' / '2.jpg'
    parser = argparse.ArgumentParser(description='Reconstruct a face and inferred skull from one image or an image directory.')

    parser.add_argument('--assets', default=str(root / 'assets' / 'external' / 'image'), help='directory of externally obtained network and topology assets')
    parser.add_argument('--seed', default=0, type=int, help='random seed used for reconstruction')
    parser.add_argument('--fsmm', default=str(root / 'assets' / 'models' / 'fsmm_200.npz'), help='geometry-only FSMM file')
    parser.add_argument('--legacy-cache', action='store_true', help='audit only: export the pre-final-update skull, matching the original bug')
    parser.add_argument('-i', '--inputpath', default=str(bundled_image), type=str,
                        help='input image or image folder')
    parser.add_argument('-s', '--savepath', default=str(root / 'data' / 'output' / 'runs' / 'image'), type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cuda or cpu' )

    # process test images
    parser.add_argument('--iscrop', default=True, type=parse_bool,
                        help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='face detector for cropping image, support for mtcnn and retinaface')

    # save
    parser.add_argument('--ldm68', default=True, type=parse_bool,
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=False, type=parse_bool,
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=False, type=parse_bool,
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=False, type=parse_bool,
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=False, type=parse_bool,
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=True, type=parse_bool,
                        help='save and show segmentation in 2d with visible mask' )
    parser.add_argument('--skull-seg', default=True, type=parse_bool,
                        help='save and show projected upper/lower bone and tooth regions')
    parser.add_argument('--useTex', default=True, type=parse_bool,
                        help='export the network face with BFM colors and lighting')
    parser.add_argument('--extractTex', default=True, type=parse_bool,
                        help='export the network face with image-sampled vertex colors')

    # backbone
    parser.add_argument('--backbone', default='deep3dcloseeye', choices=['deep3dcloseeye'], type=str,
                        help='backbone for reconstruction')

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
