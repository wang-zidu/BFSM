"""Display coupled FSMM modes and fixed-skull TMM modes."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path

import numpy as np
from bfsm_core.palette import material_hex, material_rgb
from bfsm_core.paths import check_output_inputs


from bfsm_core.tissue import TISSUE_SCHEDULE


def read_model(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(arrays[key].dtype.kind not in 'fiu' or not np.isfinite(arrays[key]).all()
           for key in ('shapeMU', 'shapeBase')):
        raise ValueError('Model geometry must contain finite real numbers')
    return arrays


def mode_states(count, sigma):
    return [('mean', None, 0.)] + [
        (f'pc{pc + 1}_{"minus" if sign < 0 else "plus"}', pc, sign * sigma)
        for pc in range(count) for sign in (-1, 1)
    ]



def _check_figure_outputs(output, filenames):
    """Keep figure exports separate from the completed geometry they visualize."""
    output = Path(output)
    inputs = [output / f'{part}_tri.npy' for part in ('face', 'skull')]
    for model in ('fsmm', 'tmm'):
        inputs.extend((output / model).glob('*/state.npz'))
    check_output_inputs(output, filenames, inputs)


def generate(fsmm_path, tmm_path, output, device='cpu', sigma=3.):
    """Generate exact FSMM decodes and original-schedule TMM fitted faces."""
    from bfsm_core.mesh import write_mesh
    from bfsm_core.model import FaceSkullModel, fit_geometry

    output = Path(output)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    fsmm, tmm = read_model(fsmm_path), read_model(tmm_path)
    nf, ns = int(fsmm['face_vertex_count']), int(fsmm['skull_vertex_count'])
    mean = fsmm['shapeMU'].reshape(-1, 3)
    if mean.shape != (nf + ns, 3) or fsmm['shapeBase'].shape[1] < 3:
        raise ValueError('FSMM must have valid counts and at least three components')
    pairs = tmm['face_to_skull_ind_new']
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or pairs.min() < 0 or pairs[:, 0].max() >= nf or pairs[:, 1].max() >= ns):
        raise ValueError('TMM pairs do not match FSMM vertex counts')
    if tmm['shapeBase'].shape[0] != len(pairs) * 3 or tmm['shapeBase'].shape[1] < 2:
        raise ValueError('TMM must have a pair displacement basis with at least two components')
    for key in ('face_tri', 'skull_tri'):
        if not np.array_equal(tmm[key], fsmm[key]):
            raise ValueError(f'TMM and FSMM differ in {key}')
    for arrays in (fsmm, tmm):
        if not all(np.isfinite(arrays[key]).all() for key in ('shapeMU', 'shapeBase')):
            raise ValueError('Non-finite model geometry')
    if (output / 'fsmm').exists() or (output / 'tmm').exists():
        raise FileExistsError('Use a new output directory; mode results are never silently overwritten')
    filenames = ['face_tri.npy', 'skull_tri.npy', 'PLOTLY_LICENSE.txt']
    for kind, count in (('fsmm', 3), ('tmm', 2)):
        filenames += [f'{kind}_modes.png', f'{kind}_modes.pdf', f'{kind}_interactive.html']
        filenames += [f'{kind}/{name}/{filename}' for name, _, _ in mode_states(count, sigma)
                      for filename in ('face.obj', 'skull.obj', 'state.npz')]
    check_output_inputs(output, filenames, [fsmm_path, tmm_path])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, pc, value in mode_states(3, sigma):
        coefficients = np.zeros(fsmm['shapeBase'].shape[1], dtype=fsmm['shapeBase'].dtype)
        if pc is not None:
            coefficients[pc] = value
        vertices = (fsmm['shapeMU'].reshape(-1) + fsmm['shapeBase'] @ coefficients).reshape(-1, 3)
        folder = output / 'fsmm' / name
        write_mesh(folder / 'face.obj', vertices[:nf], fsmm['face_tri'])
        write_mesh(folder / 'skull.obj', vertices[nf:], fsmm['skull_tri'])
        np.savez_compressed(folder / 'state.npz', face=vertices[:nf], skull=vertices[nf:],
                            coefficients=coefficients)
        records.append({'model': 'fsmm', 'state': name, 'pc': pc, 'sigma': value})

    # Keep the *same* population-mean skull in all TMM panels, independent of b.
    fixed_skull = mean[nf:].copy()
    model = FaceSkullModel(fsmm_path, device=device)
    for name, pc, value in mode_states(2, sigma):
        coefficients = np.zeros(tmm['shapeBase'].shape[1], dtype=tmm['shapeBase'].dtype)
        if pc is not None:
            coefficients[pc] = value
        displacement = (tmm['shapeMU'].reshape(-1) + tmm['shapeBase'] @ coefficients).reshape(-1, 3)
        target = fixed_skull[pairs[:, 1]] - displacement
        print(f'TMM {name}: {value:+g} sigma; fitting original five-stage geometry loss', flush=True)
        face, _, fitted, history = fit_geometry(model, target, pairs[:, 0], TISSUE_SCHEDULE,
                                                log_every=500)
        if not np.isfinite(face).all():
            raise ValueError(f'Non-finite fitted face for {name}')
        folder = output / 'tmm' / name
        write_mesh(folder / 'face.obj', face, fsmm['face_tri'])
        write_mesh(folder / 'skull.obj', fixed_skull, fsmm['skull_tri'])
        np.savez_compressed(folder / 'state.npz', face=face, skull=fixed_skull,
                            coefficients=coefficients, target_points=target,
                            **{f'fit_{key}': value for key, value in fitted.items()})
        residual = face[pairs[:, 0]] - target
        records.append({'model': 'tmm', 'state': name, 'pc': pc, 'sigma': value,
                        'target_rms_coordinate_error': float(np.sqrt(np.mean(residual ** 2))),
                        'history': history})
    for key in ('face_tri', 'skull_tri'):
        np.save(output / f'{key}.npy', fsmm[key], allow_pickle=False)
    metadata = {
        'meaning': 'population PCA modes; generated model shapes, not individual source samples',
        'fsmm': 'one identity coefficient controls both face and skull',
        'tmm': 'independent tissue coefficients; fixed FSMM mean skull; original staged face fit',
        'tissue_sign': 'skull_minus_face', 'tissue_scale': 1., 'sigma': sigma,
        'tissue_schedule': TISSUE_SCHEDULE, 'device': device,
        'source_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in [('fsmm', fsmm_path), ('tmm', tmm_path)]},
        'states': records,
    }
    return metadata


def load_states(output, model):
    result = {}
    for path in sorted((Path(output) / model).glob('*/state.npz')):
        with np.load(path, allow_pickle=False) as state:
            result[path.parent.name] = {'face': state['face'], 'skull': state['skull'],
                                      'coefficients': state['coefficients']}
    return result


def static_figure(output, model):
    """Fixed-camera orthographic display with smooth lighting and per-pixel depth."""
    output = Path(output)
    _check_figure_outputs(output, [f'{model}_modes.png', f'{model}_modes.pdf'])
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    states = load_states(output, model)
    count = 3 if model == 'fsmm' else 2
    all_vertices = np.vstack([state[part] for state in states.values() for part in ('face', 'skull')])
    centre = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2
    radius = max(np.ptp(all_vertices[:, :2], axis=0)) * .56
    faces = {part: np.load(output / f'{part}_tri.npy', allow_pickle=False) for part in ('face', 'skull')}
    fig, axes = plt.subplots(count, 6, figsize=(18, count * 3.3), squeeze=False)
    colors = {part: material_rgb(part) for part in ('face', 'skull')}
    from bfsm_core.rendering.orthographic import orthographic_image
    for pc in range(count):
        for column, state_name in enumerate((f'pc{pc + 1}_minus', 'mean', f'pc{pc + 1}_plus')):
            for part_index, part in enumerate(('face', 'skull')):
                ax = axes[pc, column * 2 + part_index]
                pixels, _, _ = orthographic_image(states[state_name][part], faces[part],
                                                   colors[part], centre, radius)
                ax.imshow(pixels, extent=(-radius, radius, -radius, radius), origin='upper')
                ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')
                ax.axis('off')
                value = float(states[state_name]['coefficients'][pc])
                ax.set_title(f'PC{pc + 1} {value:+g}σ / {part}', fontsize=22, pad=0 if model == 'tmm' else 6)
    title = ('FSMM: shared coefficients change face and skull' if model == 'fsmm' else
             'TMM: independent tissue modes; the displayed skull stays fixed')
    fig.suptitle(title, fontsize=24 if model == 'tmm' else 28, y=.995 if model == 'tmm' else .98)
    fig.text(.5, .015, 'Population model variations · neutral materials · common camera and scale.',
             ha='center', fontsize=22)
    fig.subplots_adjust(left=.015, right=.985, top=.90, bottom=.06, hspace=.16, wspace=.015)
    fig.savefig(output / f'{model}_modes.png', dpi=130, facecolor='white')
    fig.savefig(output / f'{model}_modes.pdf', dpi=150, facecolor='white')
    plt.close(fig)


def interactive_figure(output, model):
    """Self-contained Plotly HTML: all JavaScript and mesh data are embedded locally."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output = Path(output)
    _check_figure_outputs(output, [f'{model}_interactive.html', 'PLOTLY_LICENSE.txt'])
    states = load_states(output, model)
    faces = {part: np.load(output / f'{part}_tri.npy', allow_pickle=False) for part in ('face', 'skull')}
    skull_title = 'Skull (shared FSMM coefficients)' if model == 'fsmm' else 'Skull (fixed mean)'
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=('Face', skull_title))
    buttons = []
    names = list(states)
    bounds = np.vstack([state[part] for state in states.values() for part in ('face', 'skull')])
    centre = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    radius = np.ptp(bounds, axis=0).max() * .56
    for index, (name, state) in enumerate(states.items()):
        for column, part in enumerate(('face', 'skull'), 1):
            v, tri = state[part], faces[part]
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                color=material_hex(part),
                flatshading=False, visible=(name == 'mean'), name=part,
                lighting=dict(ambient=.55, diffuse=.45, specular=0),
                lightposition=dict(x=0, y=400, z=920)), row=1, col=column)
        visible = [False] * (2 * len(states))
        visible[index * 2:index * 2 + 2] = [True, True]
        buttons.append(dict(label=name, method='update', args=[{'visible': visible}]))
    scene = dict(aspectmode='cube', camera=dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0)),
                 **{f'{axis}axis': dict(visible=False, range=[float(c - radius), float(c + radius)])
                    for axis, c in zip('xyz', centre)})
    fig.update_annotations(font=dict(size=24))
    fig.update_layout(title=dict(text=f'{model.upper()} population modes', font=dict(size=30), x=.5),
        font=dict(size=22), scene=scene, scene2=scene, height=800,
        margin=dict(l=20, r=20, t=170, b=25),
        updatemenus=[dict(buttons=buttons, type='dropdown', direction='down',
                         font=dict(size=22), x=.02, y=1.20, pad=dict(l=12, r=12, t=8, b=8))],
        showlegend=False)
    html_path = output / f'{model}_interactive.html'
    fig.write_html(html_path, include_plotlyjs=True, include_mathjax=False, full_html=True,
                   div_id=f'{model}-modes', config={'displaylogo': False, 'responsive': True})
    # Explicitly prevent network requests even when the HTML is opened with network access.
    policy = ("default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; "
              "style-src 'unsafe-inline'; img-src data: blob:; font-src data:; "
              "worker-src blob:; connect-src 'none'")
    html_path.write_text(html_path.read_text().replace(
        '<head>', f'<head><meta http-equiv="Content-Security-Policy" content="{policy}">', 1))
    distribution = importlib.metadata.distribution('plotly')
    license_files = [file for file in distribution.files or []
                     if str(file).endswith('.dist-info/licenses/LICENSE.txt')]
    if license_files:
        license_text = distribution.locate_file(license_files[0]).read_text()
    else:
        license_text = (Path(__file__).parent / 'PLOTLY_LICENSE.txt').read_text()
    (output / 'PLOTLY_LICENSE.txt').write_text(license_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fsmm', default='assets/models/fsmm_200.npz')
    parser.add_argument('--tmm', default='assets/models/tmm_200.npz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sigma', type=float, default=3.)
    parser.add_argument('--render-existing', action='store_true', help='Render an already completed modes directory')
    parser.add_argument('--generate-only', action='store_true', help='Generate geometry; defer PNG/PDF/HTML')
    args = parser.parse_args()
    if not args.render_existing:
        generate(args.fsmm, args.tmm, args.output, args.device, args.sigma)
    if not args.generate_only:
        for model in ('fsmm', 'tmm'):
            static_figure(args.output, model)
            interactive_figure(args.output, model)


if __name__ == '__main__':
    main()
