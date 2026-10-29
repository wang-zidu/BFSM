"""Image-aligned DECA/BFSM overlays using the saved crop and weak camera."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from bfsm_core.rendering.image_space import rasterize_pixels, shaded_colors


def overlay(image, transform, camera, vertices, faces, color=(.78, .78, .78)):
    image, transform, camera = np.asarray(image), np.asarray(transform), np.asarray(camera)
    from .deca_geometry import project_pixels
    project_pixels(vertices, camera)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or transform.shape != (3, 3) or not np.isfinite(transform).all()
            or abs(np.linalg.det(transform)) < 1e-12):
        raise ValueError('Overlay requires an RGB image and invertible original-to-crop transform')
    if not np.allclose(transform[2], [0., 0., 1.], atol=1e-12):
        raise ValueError('Expected an affine DECA crop transform')
    crop_pixels = project_pixels(vertices, camera)
    homogeneous = np.column_stack((crop_pixels, np.ones(len(crop_pixels)))) @ np.linalg.inv(transform).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    colors = shaded_colors(vertices, faces, color)
    rendered, ids, (left, top) = rasterize_pixels(pixels, faces, -np.asarray(vertices)[:, 2],
                                                colors, image.shape[:2])
    output = image.copy()
    height, width = ids.shape
    region = np.s_[top:top + height, left:left + width]
    selected = ids >= 0
    output[region][selected] = np.rint(np.clip(rendered[selected], 0., 1.) * 255).astype(np.uint8)
    return output



def render_preview(image, transform, camera, deca, fitted, faces, skull, skull_faces,
                   output, *, display_crop=None):
    panels = [image, overlay(image, transform, camera, deca, faces),
              overlay(image, transform, camera, fitted, faces),
              overlay(image, transform, camera, skull, skull_faces, (.75, .75, .75))]
    canvas = Image.new('RGB', (1104, 366), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 22)
    titles = ('Input', 'DECA face overlay', 'Fitted BFSM face', 'Inferred skull')
    for i, (panel, title) in enumerate(zip(panels, titles)):
        picture = Image.fromarray(panel)
        if display_crop is not None:
            picture = picture.crop(display_crop)
        tile = ImageOps.contain(picture, (276, 330), Image.Resampling.LANCZOS)
        canvas.paste(tile, (i * 276 + (276 - tile.width)//2, 36 + (330 - tile.height)//2))
        draw.text((i*276+138, 18), title, font=font, fill='black', anchor='mm')
    output = Path(output)
    canvas.save(output)
    return output
