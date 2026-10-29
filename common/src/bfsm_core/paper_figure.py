"""Arrange actual fixed-camera comparison renders as before/prediction view pairs."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _grid(path):
    picture = Image.open(path).convert('RGB')
    size = picture.width // 3
    if picture.width != 3 * size or picture.height != 4 * (size + 26):
        raise ValueError('Expected the four-row, three-view grid from bfsm_core.visualize')
    return picture, size


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def compose(geometry, output, texture=None, title='Synthetic skull editing and face prediction'):
    geometry, output = Path(geometry), Path(output)
    inputs = [geometry] + ([] if texture is None else [Path(texture)])
    if any(output.resolve() == path.resolve() or
           (output.exists() and path.exists() and output.samefile(path)) for path in inputs):
        raise ValueError('Output must differ from the input comparison grids')
    blue, size = _grid(geometry)
    rows = []
    if texture is not None:
        colored, texture_size = _grid(texture)
        if texture_size != size:
            raise ValueError('Texture and geometry grids must share the same resolution')
        rows.append(('RGB', colored, 0, 1))
    rows.extend([('Face', blue, 0, 1), ('Skull', blue, 2, 3)])
    margin, label, gap, group_gap, top, row_gap = 16, 92, 34, 22, 92, 10
    group = 2 * size + gap
    canvas = Image.new('RGB', (2 * margin + label + 3 * group + 2 * group_gap,
                               top + len(rows) * (size + row_gap) + 34), 'white')
    draw = ImageDraw.Draw(canvas)
    ink, blue_ink = '#242424', '#17365d'
    draw.text((margin, -4), title, font=_font(32), fill=blue_ink)
    for view, caption in enumerate(('0°', '55°', '90°')):
        x = margin + label + view * (group + group_gap)
        draw.text((x + 8, 28), caption, font=_font(34), fill=blue_ink)
        draw.text((x + 5, 60), 'Before', font=_font(26), fill=ink)
        draw.text((x + size + gap + 5, 60), 'Prediction / Plan', font=_font(26), fill=ink)
        for row, (name, image, before, after) in enumerate(rows):
            y = top + row * (size + row_gap)
            if view == 0:
                draw.multiline_text((margin, y + size // 2 - 18), name,
                                    font=_font(26), fill=blue_ink, spacing=5)
            for column, source_row in enumerate((before, after)):
                tile = image.crop((view * size, source_row * (size + 26) + 26,
                                   (view + 1) * size, (source_row + 1) * (size + 26)))
                canvas.paste(tile, (x + column * (size + gap), y))
            centre = y + size // 2
            draw.line((x + size + 5, centre, x + size + gap - 7, centre), fill=ink, width=2)
            tip = x + size + gap - 7
            draw.polygon([(tip, centre), (tip - 6, centre - 4), (tip - 6, centre + 4)], fill=ink)
        if view < 2:
            divider = x + group + group_gap // 2
            draw.line((divider, 92, divider, canvas.height - 32), fill='#d8d8d8')
    draw.text((margin, canvas.height - 30),
              'Blue: face prediction. Gold: edited skull.',
              font=_font(22), fill='#555555')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-grid', required=True)
    parser.add_argument('--texture-grid')
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='Synthetic skull editing and face prediction')
    args = parser.parse_args()
    compose(args.geometry_grid, args.output, args.texture_grid, args.title)


if __name__ == '__main__':
    main()
