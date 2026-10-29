"""Visualize fixed upper/lower skull and tooth regions on BFSM geometry."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from bfsm_core.mesh import read_mesh, write_mesh
from bfsm_core.paths import model_file, check_output_inputs

REGIONS = ('upper_bone', 'upper_teeth', 'lower_bone', 'lower_teeth')
COLORS = ('#e5dd8c', '#79b1d6', '#94bb92', '#ac8ac4')


def topology_hash(faces):
    return hashlib.sha256(np.asarray(faces, dtype='<i8').tobytes()).hexdigest()


def load_regions(path, faces, vertex_count):
    with np.load(path, allow_pickle=False) as archive:
        masks = {name: archive[name] for name in REGIONS}
        if (int(archive['vertex_count']) != vertex_count or
                archive['topology_sha256'].item() != topology_hash(faces)):
            raise ValueError('Region asset requires the original BFSM skull vertex order and topology')
        if any(mask.dtype != np.bool_ or mask.shape != (vertex_count,) for mask in masks.values()):
            raise ValueError('Region masks must be one boolean per skull vertex')
        if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
            raise ValueError('The four skull regions must form a nonoverlapping complete partition')
    return masks


def export_regions(output, *, model=None, skull=None, regions=None, render=False, device='cpu', size=256):
    output = Path(output)
    if skull is None:
        with np.load(model_file(model), allow_pickle=False) as archive:
            mean = archive['shapeMU']
            center = archive.get('recenter_offset')
            if (mean.dtype.kind not in 'fiu' or not np.isfinite(mean).all() or
                    (center is not None and (center.dtype.kind not in 'fiu' or not np.isfinite(center).all()))):
                raise ValueError('Model geometry and center must contain finite real numbers')
            mean = mean.astype(np.float32).reshape(-1, 3)
            mean -= archive.get('recenter_offset', mean.mean(axis=0, keepdims=True))
            count = int(archive['face_vertex_count'])
            vertices, faces = mean[count:], archive['skull_tri']
    else:
        mesh = read_mesh(skull)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    region_file = Path(regions) if regions is not None else Path(__file__).with_name('skull_regions.npz')
    masks = load_regions(region_file, faces, len(vertices))
    inputs = [region_file, Path(skull) if skull is not None else model_file(model)]
    if any(output.resolve() == path.resolve() or output.resolve() in path.resolve().parents for path in inputs):
        raise ValueError('Region output must not contain or overwrite its input files')
    filenames = [name + '.obj' for name in REGIONS] + ['region_colored_skull.obj']
    if render:
        filenames += ['skull_regions.png', 'views/comparison.png']
        filenames += ['views/skull_regions_' + view + '.png'
                      for view in ('front', 'left_oblique', 'left_profile')]
    check_output_inputs(output, filenames, inputs)
    colors = np.zeros((len(vertices), 3))
    output.mkdir(parents=True, exist_ok=True)
    for name, color in zip(REGIONS, COLORS):
        mask = masks[name]
        colors[mask] = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255.
        indices = np.flatnonzero(mask)
        reverse = np.full(len(vertices), -1, dtype=int)
        reverse[indices] = np.arange(len(indices))
        selected = faces[np.all(mask[faces], axis=1)]
        write_mesh(output / (name + '.obj'), vertices[indices], reverse[selected], colors[indices])
        print(f'{name}: {len(indices)} vertices, {len(selected)} complete-region triangles')
    colored = output / 'region_colored_skull.obj'
    write_mesh(colored, vertices, faces, colors)
    if render:
        from PIL import Image, ImageDraw, ImageFont
        from bfsm_core.visualize import render_comparison
        picture = render_comparison([('skull_regions', colored)], output / 'views', device=device, size=size)
        with Image.open(picture) as original:
            canvas = Image.new('RGB', (original.width, original.height + 55), 'white')
            canvas.paste(original, (0, 55))
        draw = ImageDraw.Draw(canvas)
        legend_font = ImageFont.truetype('DejaVuSans.ttf', 18)
        for i, (name, color) in enumerate(zip(REGIONS, COLORS)):
            x = 8 + i * canvas.width // 4
            draw.rectangle((x, 9, x + 12, 21), fill=color)
            draw.text((x + 17, 9), name.replace('_', ' '), fill='black', font=legend_font)
        draw.text((8, 33), 'Fixed regions; not osteotomy or individual tooth labels', fill='black', font=legend_font)
        canvas.save(output / 'skull_regions.png')
    return colored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='FSMM NPZ; default assets/models/fsmm_200.npz')
    parser.add_argument('--skull', help='Optional registered skull in model vertex order instead of the mean')
    parser.add_argument('--regions', help='Optional compatible region-label NPZ')
    parser.add_argument('--output', required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    export_regions(args.output, model=args.model, skull=args.skull, regions=args.regions,
                   render=args.render, device=args.device, size=args.size)


if __name__ == '__main__':
    main()
