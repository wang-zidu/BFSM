"""Prepare a photograph-derived face/skull pair in canonical BFSM coordinates."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from bfsm_core.mesh import write_mesh
from bfsm_core.model import FaceSkullModel
from bfsm_core.paths import ROOT, model_file, check_output_inputs


@dataclass
class ImageFrame:
    """Canonical BFSM to the image frontend's exported OBJ frame; not CT units."""
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def to_world(self, vertices):
        return self.scale * np.asarray(vertices) @ self.rotation + self.translation

    def to_canonical(self, vertices):
        return (np.asarray(vertices) - self.translation) @ self.rotation.T / self.scale


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _context(image, model, seed):
    """Bind cached results to input, weights, source and the original CLI settings."""
    import bfsm_image
    import bfsm_core
    frontend = Path(bfsm_image.__file__).resolve().parent
    assets = ROOT / 'assets/external/image'
    digest = hashlib.sha256()
    settings = ('photograph;retinaface;iscrop=1;deep3dcloseeye;'
                'ldm68=1;useTex=1;extractTex=1;legacy_cache=0;'
                'steps=500,500,500,1500,2000;seed=' + str(seed))
    digest.update(settings.encode())
    for name, path in [('image', image), ('fsmm', model)]:
        digest.update((name + ':' + _sha256(path)).encode())
    for path in sorted(frontend.rglob('*')):
        if path.suffix in ('.py', '.pyx'):
            digest.update((path.relative_to(frontend).as_posix() + ':' + _sha256(path)).encode())
    for name in ('face_model.npy', 'similarity_Lm3D_all.mat', 'large_base_net.pth',
                 'epoch_20_deep3d_closeeye.pth', 'retinaface_resnet50_2020-07-20_old_torch.pth'):
        digest.update((name + ':' + _sha256(assets / name)).encode())
    shared = Path(bfsm_core.__file__).resolve().parent
    sources = [shared / 'parametric.py', shared / 'model.py']
    sources.extend(path for path in (shared / 'rendering').rglob('*')
                   if path.suffix in {'.py', '.pyx', '.cpp', '.h'}
                   and path.name != 'mesh_core_cython.cpp'
                   and not any(part in {'build', '__pycache__'} for part in path.parts))
    for path in sorted(sources):
        digest.update(('bfsm_core/' + path.relative_to(shared).as_posix() + ':' + _sha256(path)).encode())
    return digest.hexdigest()


def prepare_image(image, output, *, device='cuda:0', reconstruction_dir=None, seed=0, model=None):
    """Run the original image frontend or explicitly verify a matching cached run.

    Outputs are the canonical corresponding FSMM face/skull pair and the image
    frame transform. Both meshes are inferred geometry, not observed scans.
    """
    image = Path(image).resolve()
    model_path = model_file(model).resolve()
    output = Path(output).resolve()
    if any(path == output or output in path.parents for path in (image, model_path)):
        raise ValueError("Choose an output directory that does not contain the input image or model")
    protected = [image, model_path]
    assets = ROOT / 'assets/external/image'
    if assets.is_dir():
        protected.extend(path for path in assets.rglob('*') if path.is_file())
    if reconstruction_dir is not None:
        protected.extend(Path(reconstruction_dir) / name
                         for name in ('reconstruction.npz', 'cache_context.npz'))
    exports = ['before_face.obj', 'before_skull.obj', 'image_frame.npz']
    if reconstruction_dir is None:
        exports.append(str(Path('image_reconstruction') / image.stem / 'cache_context.npz'))
    check_output_inputs(output, exports, protected)
    output.mkdir(parents=True, exist_ok=True)
    signature = _context(image, model_path, seed)
    if reconstruction_dir is None:
        destination = output / 'image_reconstruction'
        command = [sys.executable, '-m', 'bfsm_image.demo_reconstruction',
                   '--inputpath', str(image), '--savepath', str(destination),
                   '--assets', str(ROOT / 'assets/external/image'), '--fsmm', str(model_path),
                   '--device', str(device), '--seed', str(seed),
                   '--iscrop', '1', '--detector', 'retinaface', '--ldm68', '1',
                   '--useTex', '1', '--extractTex', '1']
        subprocess.run(command, check=True, cwd=ROOT)
        cache = destination / image.stem
        result = cache / 'reconstruction.npz'
        np.savez_compressed(cache / 'cache_context.npz', context=np.asarray(signature),
                            result_sha256=np.asarray(_sha256(result)))
    else:
        cache = Path(reconstruction_dir).resolve()
        result = cache / 'reconstruction.npz'
        receipt = cache / 'cache_context.npz'
        if not receipt.is_file():
            raise ValueError('Explicit image cache requires cache_context.npz from a verified frontend run')
        with np.load(receipt, allow_pickle=False) as data:
            if (data['context'].item() != signature or
                    data['result_sha256'].item() != _sha256(result)):
                raise ValueError('Image cache does not match the input, model, frontend or settings')

    with np.load(result, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    coefficients = {key: torch.as_tensor(arrays['coeff_' + key], dtype=torch.float32)
                    for key in ('id', 'angle', 'scale', 'trans')}
    geometry = FaceSkullModel(model_path, device='cpu')
    if coefficients['id'].shape != (1, geometry.components):
        raise ValueError('Image fit coefficient count does not match the selected FSMM')
    for value in coefficients.values():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite image fit coefficients')
    scale = float(coefficients['scale'].item())
    if scale <= 0:
        raise ValueError('Image fit scale must be positive')
    rotation = geometry.rotation(coefficients['angle']).numpy()[0].astype(float)
    frame = ImageFrame(scale, rotation, coefficients['trans'].numpy()[0].astype(float))

    def from_camera(key, count):
        values = np.asarray(arrays[key], dtype=float)
        if values.shape != (1, count, 3) or not np.isfinite(values).all():
            raise ValueError('Invalid reconstruction array: ' + key)
        values = values[0].copy()
        values[:, 2] = 10 - values[:, 2]
        return values

    image_face = from_camera('fsmm_face', geometry.face_count)
    image_skull = from_camera('v3d_skull', geometry.skull_count)
    face, skull = frame.to_canonical(image_face), frame.to_canonical(image_skull)
    with torch.no_grad():
        expected_face, expected_skull = geometry.decode(coefficients)
        neutral = geometry.initial_coefficients()
        neutral['id'] = coefficients['id']
        canonical_face, canonical_skull = geometry.decode(neutral)
    for actual, expected in ((image_face, expected_face.numpy()[0]),
                             (image_skull, expected_skull.numpy()[0]),
                             (face, canonical_face.numpy()[0]),
                             (skull, canonical_skull.numpy()[0])):
        if not np.allclose(actual, expected, atol=5e-6, rtol=1e-6):
            raise ValueError('Image reconstruction coefficients and mesh do not agree with the selected FSMM')
    triangles = np.asarray(arrays['tri'])
    skull_triangles = np.asarray(arrays['skull_tri'])
    if (not np.array_equal(triangles, geometry.arrays['face_tri']) or
            not np.array_equal(skull_triangles, geometry.arrays['skull_tri'])):
        raise ValueError('Image reconstruction topology does not match the selected FSMM')
    face_colors = np.asarray(arrays['face_texture'])[0]
    paths = {key: output / name for key, name in
             [('face', 'before_face.obj'), ('skull', 'before_skull.obj')]}
    write_mesh(paths['face'], face, triangles, face_colors)
    write_mesh(paths['skull'], skull, skull_triangles)
    np.savez_compressed(output / 'image_frame.npz', scale=np.asarray(scale), rotation=rotation,
                        translation=frame.translation, input_sha256=np.asarray(_sha256(image)),
                        model_sha256=np.asarray(_sha256(model_path)),
                        canonical_face=face, canonical_skull=skull,
                        source_surface_kind=np.asarray('same-topology FSMM reconstruction from image; not a scanned surface'))
    return {**paths, 'transform': frame}
