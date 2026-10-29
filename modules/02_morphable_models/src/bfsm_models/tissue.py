"""Original tissue-vector perturbation and five-stage face fitting demo."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from bfsm_core.palette import mesh_colors
from bfsm_core.paths import check_output_inputs

from ._tissue import process_face_skull, change_texture, change_tissue
from bfsm_core.parametric import resolve_model_path


def load_model(path):
    path = resolve_model_path(path)
    loaded = np.load(path, allow_pickle=path.suffix == '.npy')
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    else:
        arrays = loaded.item()
    if any(np.asarray(arrays[key]).dtype.kind not in 'fiu' or not np.isfinite(arrays[key]).all()
           for key in ('shapeMU', 'shapeBase')):
        raise ValueError('Model geometry must contain finite real numbers')
    return arrays


def render_grid(paths, output, device, neutral=False):
    import cv2
    import trimesh
    fov = 2 * np.arctan(112 / 1015) * 180 / np.pi
    if torch.device(device).type == 'cpu':
        from bfsm_core.rendering.cpu_renderer import MeshRenderer_cpu
        renderer = MeshRenderer_cpu(rasterize_fov=fov, znear=5., zfar=15., rasterize_size=512)
    else:
        from bfsm_core.rendering.nv_diffrast import MeshRenderer
        renderer = MeshRenderer(rasterize_fov=fov, znear=5., zfar=15., rasterize_size=512, use_opengl=False)
    canvas = np.full((int(np.ceil(len(paths) / 3)) * 512, min(3, len(paths)) * 512, 3), 255, np.uint8)
    for i, path in enumerate(paths):
        mesh = trimesh.load(path, process=False, maintain_order=True)
        vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float32).unsqueeze(0)
        vertices[..., -1] = 10. - vertices[..., -1]
        triangles = torch.tensor(mesh.faces, device=device, dtype=torch.int64)
        colors = torch.tensor(mesh_colors(mesh, 'face', neutral=neutral),
                              device=device, dtype=torch.float32).unsqueeze(0)
        # Preview only: use the shared frontal light without changing geometry.
        normals = torch.tensor(np.array(mesh.vertex_normals), device=device, dtype=torch.float32)
        light = torch.tensor([0., .4, .92], device=device, dtype=torch.float32)
        light = light / torch.linalg.vector_norm(light)
        illumination = .55 + .45 * (normals @ light).clamp(0., 1.)
        colors = colors * illumination[None, :, None]
        with torch.no_grad():
            mask, _, image, _ = renderer(vertices.contiguous(), triangles.contiguous(), feat=colors.contiguous())
            image = image * mask + torch.ones_like(image) * (1 - mask)
        rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        cv2.putText(rgb, path.stem, (8, 26), cv2.FONT_HERSHEY_DUPLEX, .9, (0, 0, 0), 2)
        row, col = divmod(i, 3)
        canvas[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512] = rgb
    cv2.imwrite(str(output), canvas)


def main(args):
    out = Path(args.output_dir)
    # Protect every written product, including hardlinks to an input asset.
    selected_outputs = args.variant or ['s0.75', 's1.25', 'id1', 'id2']
    filenames = [name + suffix for name in selected_outputs
                 for suffix in ('.obj', '_parameters.npz', '_tissue_points.ply')]
    for source, name in ((args.face_obj_path, 'input_face.obj'), (args.skull_obj_path, 'input_skull.obj')):
        if Path(source).resolve() != (out / name).resolve():
            filenames.append(name)
    if args.texture_variations:
        filenames += ['tex1.obj', 'tex2.obj']
    if not args.no_render:
        filenames.append('change_tissue.png')
    check_output_inputs(out, filenames, [args.face_obj_path, args.skull_obj_path,
                        resolve_model_path(args.fsmm), resolve_model_path(args.tmm), args.face_tri])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fsmm, tmm = load_model(args.fsmm), load_model(args.tmm)
    face_tri = fsmm['face_tri'] if 'face_tri' in fsmm else np.load(args.face_tri)
    pairs = tmm['face_to_skull_ind_new']
    data = process_face_skull(pairs, args.face_obj_path, args.skull_obj_path,
                              require_colors=args.texture_variations)
    for source, name in ((args.face_obj_path, 'input_face.obj'), (args.skull_obj_path, 'input_skull.obj')):
        if Path(source).resolve() != (out / name).resolve():
            shutil.copy2(source, out / name)
    paths = [out / 'input_face.obj']
    if args.texture_variations:
        if 'texBase' not in fsmm:
            raise ValueError('Texture variations require a separately obtained FSMM texture model.')
        for name in ('tex1', 'tex2'):
            paths.append(Path(change_texture(fsmm, face_tri, str(out), data['face_v'],
                                             output_name=name, shapeMu=data['d_i'], texMu=data['t_base'])))
    variants = [('s0.75', .75, True), ('s1.25', 1.25, True), ('id1', 1., False), ('id2', 1., False)]
    selected = set(args.variant or [item[0] for item in variants])
    for name, scale, zero in variants:
        if name not in selected:
            continue
        paths.append(Path(change_tissue(
            FaceSkullMorphableModel=fsmm, TissueMorphableModel=tmm, face_tri=face_tri,
            face_to_skull_ind_new=pairs, skull_path=args.skull_obj_path, output_dir=str(out),
            alpha_shape=np.zeros(tmm['shapeBase'].shape[1]) if zero else None,
            alpha_color=np.zeros(fsmm['texBase'].shape[1]) if args.texture_variations else None,
            alpha_scale=scale, output_name=name, use_texture=args.texture_variations,
            shapeMu=data['d_i'], texMu=data['t_base'], model_path=args.fsmm,
            device=str(device), legacy_cache=args.legacy_cache)))
    if not args.no_render:
        render_grid(paths, out / 'change_tissue.png', str(device), neutral=not args.texture_variations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--face_obj_path', required=True)
    parser.add_argument('-s', '--skull_obj_path', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('--fsmm', required=True, help='FSMM geometry NPZ or trusted legacy NPY')
    parser.add_argument('--tmm', required=True, help='TMM geometry NPZ or trusted legacy NPY')
    parser.add_argument('--face-tri', help='Required only for legacy FSMM files without face_tri')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--texture-variations', action='store_true', help='Requires separate private texture weights')
    parser.add_argument('--variant', action='append', choices=['s0.75', 's1.25', 'id1', 'id2'])
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--legacy-cache', action='store_true', help=argparse.SUPPRESS)
    main(parser.parse_args())
