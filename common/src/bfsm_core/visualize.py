"""Fixed-camera mesh views for inspection of face/skull demonstration results."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .mesh import read_mesh
from .palette import mesh_colors
from .paths import check_output_inputs


def render_comparison(mesh_paths, output, device='cuda:0', size=512, neutral=False):
    mesh_paths = list(mesh_paths)
    # Labels are displayed verbatim; only their filename components are sanitized.
    filename_names = [''.join(char if char.isalnum() or char in '._-' else '_'
                              for char in name).strip('.') for name, _ in mesh_paths]
    if any(not name for name in filename_names) or len(set(filename_names)) != len(filename_names):
        raise ValueError('Mesh labels must produce distinct nonempty image filenames')
    filenames = [name + '_' + view + '.png' for name in filename_names
                 for view in ('front', 'left_oblique', 'left_profile')] + ['comparison.png']
    check_output_inputs(output, filenames, [path for _, path in mesh_paths])
    from pytorch3d.renderer import (FoVOrthographicCameras, MeshRenderer, MeshRasterizer,
        RasterizationSettings, SoftPhongShader, PointLights, TexturesVertex, look_at_view_transform,
        Materials, BlendParams, TexturesUV)
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_obj

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = [(name, read_mesh(path)) for name, path in mesh_paths]
    all_vertices = np.vstack([mesh.vertices for _, mesh in loaded])
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    centre = (lower + upper) * .5
    radius = float(np.max(upper - lower) * .57)
    lighting = PointLights(device=device, location=((0., 1.6, 3.68),),
                           ambient_color=((.55, .55, .55),), diffuse_color=((.45, .45, .45),),
                           specular_color=((0., 0., 0.),))
    settings = RasterizationSettings(image_size=size, blur_radius=0, faces_per_pixel=1)
    panels = []
    label_font = ImageFont.truetype('DejaVuSans.ttf', 18)
    views = [('front', 0), ('left_oblique', 55), ('left_profile', 90)]
    for (name, path), (_, mesh), filename_name in zip(mesh_paths, loaded, filename_names):
        vertices = torch.as_tensor(np.asarray(mesh.vertices - centre), dtype=torch.float32, device=device)
        faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long, device=device)
        colors = mesh_colors(mesh, name, neutral=neutral)
        textures = TexturesVertex(verts_features=[torch.tensor(colors, dtype=torch.float32, device=device)])
        if not neutral and mesh.visual.kind == 'texture':
            # OBJ can have a separate UV index for every face corner. Preserve those
            # seams instead of reducing the photographed scan to one colour per vertex.
            source_vertices, source_faces, auxiliary = load_obj(str(path), device=device)
            if auxiliary.texture_images:
                if len(auxiliary.texture_images) != 1:
                    raise ValueError('Comparison renderer expects one texture image per OBJ')
                if (not torch.equal(source_faces.verts_idx, faces) or
                        not np.allclose(source_vertices.cpu().numpy(), mesh.vertices, atol=1e-7)):
                    raise ValueError('Textured OBJ topology changed during loading')
                if (source_faces.textures_idx < 0).any():
                    raise ValueError('Textured comparison mesh has missing UV indices')
                textures = TexturesUV(
                    maps=[next(iter(auxiliary.texture_images.values())).to(device)],
                    faces_uvs=[source_faces.textures_idx.to(device)],
                    verts_uvs=[auxiliary.verts_uvs.to(device)])
        meshes = Meshes(verts=[vertices], faces=[faces], textures=textures)
        row = []
        for view_name, azimuth in views:
            rotation, translation = look_at_view_transform(dist=4., elev=0., azim=azimuth, device=device)
            cameras = FoVOrthographicCameras(device=device, R=rotation, T=translation,
                znear=.01, zfar=100., min_x=-radius, max_x=radius, min_y=-radius, max_y=radius)
            renderer = MeshRenderer(MeshRasterizer(cameras=cameras, raster_settings=settings),
                SoftPhongShader(device=device, cameras=cameras, lights=lighting,
                    materials=Materials(device=device, shininess=12), blend_params=BlendParams(background_color=(1, 1, 1))))
            with torch.no_grad():
                pixels = renderer(meshes)[0, ..., :3].clamp(0, 1).cpu().numpy()
            picture = Image.fromarray((pixels * 255).astype(np.uint8))
            canvas = Image.new('RGB', (size, size + 26), 'white')
            canvas.paste(picture, (0, 26))
            display_name = name.replace('original_', 'Input ').replace('registered_', 'Fit ').replace('_', ' ')
            display_view = {'front': '0°', 'left_oblique': '55°', 'left_profile': '90°'}[view_name]
            label = display_name + ' / ' + display_view
            ImageDraw.Draw(canvas).text((8, 1), label, font=label_font, fill='black')
            canvas.save(output / f'{filename_name}_{view_name}.png')
            row.append(canvas)
        panels.append(row)
    grid = Image.new('RGB', (size * len(views), (size + 26) * len(panels)), 'white')
    for row_index, row in enumerate(panels):
        for column, picture in enumerate(row):
            grid.paste(picture, (column * size, row_index * (size + 26)))
    grid.save(output / 'comparison.png')
    return output / 'comparison.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', action='append', required=True, help='label=path, repeat for each row')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--neutral', action='store_true', help='Inspect geometry without source colours')
    args = parser.parse_args()
    render_comparison([entry.split('=', 1) for entry in args.mesh], args.output, args.device, args.size, args.neutral)


if __name__ == '__main__':
    main()
