"""Compose the seven reconstruction panels and five fitted tissue states."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bfsm_core.paths import check_output_inputs

TITLES = ('Input', 'PCA texture overlay', 'Gray face', 'Inferred skull',
          '68 landmarks', 'Face segmentation', 'Skull segmentation', 'TMM input',
          'TMM scale 0.75', 'TMM scale 1.25', 'TMM variation 1', 'TMM variation 2')


def compose(reconstruction, tissue, output, *, display_crop=None):
    """Read default 4x2/3x2 previews and make a labelled, aspect-preserving 4x3 grid.

    display_crop is an optional fractional box applied only to reconstruction
    panels for documentation. It never changes reconstruction inputs or meshes.
    """
    reconstruction, tissue, output = map(Path, (reconstruction, tissue, output))
    check_output_inputs(output.parent, [output.name], [reconstruction, tissue])
    with Image.open(reconstruction) as source:
        image = source.convert('RGB')
    with Image.open(tissue) as source:
        variations = source.convert('RGB')
    if image.width % 4 or image.height % 2:
        raise ValueError('Expected the default four-column reconstruction preview')
    if variations.size != (1536, 1024):
        raise ValueError('Expected the five-state 512-pixel tissue preview')
    width, height = image.width // 4, image.height // 2
    if display_crop is not None:
        if (len(display_crop) != 4 or not all(0 <= v <= 1 for v in display_crop)
                or display_crop[0] >= display_crop[2] or display_crop[1] >= display_crop[3]):
            raise ValueError('Display crop must be a nonempty fractional box')
    panels = []
    for i in range(7):
        x, y = i % 4 * width, i // 4 * height
        panel = image.crop((x, y, x + width, y + height))
        if display_crop is not None:
            panel = panel.crop(tuple(round(v * (width if j % 2 == 0 else height))
                                     for j, v in enumerate(display_crop)))
        panels.append(panel)
    for i in range(5):
        x, y = i % 3 * 512, i // 3 * 512
        panel = variations.crop((x, y, x + 512, y + 512))
        # Replace the original filename caption with the common readable header.
        ImageDraw.Draw(panel).rectangle((0, 0, 511, 35), fill='white')
        panels.append(panel)
    size, header = 512, 56
    canvas = Image.new('RGB', (4 * size, 3 * (size + header)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 28)
    for i, (panel, title) in enumerate(zip(panels, TITLES)):
        x, y = i % 4 * size, i // 4 * (size + header)
        draw.text((x + size // 2, y + header // 2), title, fill='black', font=font, anchor='mm')
        tile = ImageOps.contain(panel, (size, size), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x + (size - tile.width) // 2, y + header + (size - tile.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
