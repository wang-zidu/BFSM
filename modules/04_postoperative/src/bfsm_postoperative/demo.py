"""Predict postoperative geometry from registered scans or photo-reconstructed FSMM pairs."""
import argparse
import hashlib
from numbers import Real
import random
import shutil
import tomllib
from pathlib import Path

import numpy as np
import torch

from .edit import mandibular_edit
from bfsm_core.mesh import read_mesh, replace_obj_vertices
from bfsm_core import model as geometry_model
from bfsm_core.model import FaceSkullModel
from bfsm_core.paths import ROOT, model_file
from .postoperative import predict, transport_tissue
from .surface import surface_assets, transfer_details
from bfsm_core.visualize import render_comparison


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prediction_signature(face, skull, plan, model):
    """Identify the exact numerical inputs and implementation for an explicit cache."""
    files = [face, skull, plan, model, Path(__file__).with_name('postoperative.py'),
             Path(geometry_model.__file__)]
    return np.asarray([_digest(path) for path in files])


def _prediction(face, skull, plan, folder, model_path, model, device, cache=None):
    signature = prediction_signature(face, skull, plan, model_path)
    if cache is None:
        vertices, record = predict(face, skull, plan, folder, model_path, device)
    else:
        cache = Path(cache)
        with np.load(cache / 'prediction_inputs.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['sha256'], signature):
                raise ValueError('Prediction cache inputs or implementation differ; rerun prediction')
        before, old_skull, edited = [read_mesh(path) for path in (face, skull, plan)]
        mapping = model.arrays['face_to_skull_ind_new']
        target, _ = transport_tissue(old_skull, edited, mapping[:, 1], before.vertices[mapping[:, 0]])
        saved_target = np.load(cache / 'transported_tissue_points.npy', allow_pickle=False)
        if saved_target.shape != target.shape or not np.allclose(saved_target, target, atol=1e-7, rtol=1e-6):
            raise ValueError('Prediction cache tissue targets do not match this plan')
        cached_mesh = read_mesh(cache / 'predicted_face.obj')
        with np.load(cache / 'coefficients.npz', allow_pickle=False) as saved:
            coefficients = {key: torch.as_tensor(saved[key], dtype=torch.float32, device=model.device)
                            for key in ('id', 'angle', 'scale', 'trans')}
        with torch.no_grad():
            decoded, _ = model.decode(coefficients)
        vertices = np.asarray(cached_mesh.vertices)
        if (not np.array_equal(cached_mesh.faces, model.arrays['face_tri']) or
                not np.allclose(vertices, decoded.cpu().numpy()[0], atol=1e-7, rtol=1e-6)):
            raise ValueError('Prediction cache geometry does not decode from its saved coefficients')
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('predicted_face.obj', 'coefficients.npz', 'transported_tissue_points.npy'):
            if (cache / name).resolve() != (folder / name).resolve():
                shutil.copy2(cache / name, folder / name)
        record = {'explicit_verified_cache': str(cache)}
    np.savez(folder / 'prediction_inputs.npz', sha256=signature)
    return vertices, record


def _world_surface(source, destination, transform):
    copies = surface_assets(source, destination)
    vertices = read_mesh(source).vertices
    replace_obj_vertices(source, destination, vertices if transform is None else transform.to_world(vertices))
    for original, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.resolve() != target.resolve():
            shutil.copy2(original, target)


def _configuration(path=None):
    if path is None:
        path = ROOT / 'modules/04_postoperative/configs/scan_demo.toml'
    path = Path(path).resolve()
    configuration = tomllib.loads(path.read_text())
    for options in configuration.get('edits', {}).values():
        if options.get('kind') == 'external_plan':
            options['plan'] = str((path.parent / options['plan']).resolve())
    reference = configuration.get('reference', {})
    if 'geometry' in reference:
        reference['geometry'] = str((path.parent / reference['geometry']).resolve())
    _validate_configuration(configuration)
    return configuration


def _validate_configuration(configuration):
    edits = configuration.get('edits')
    if not isinstance(edits, dict) or not edits:
        raise ValueError('Configuration must contain at least one named edit')
    for name, options in edits.items():
        if (not isinstance(name, str) or name in {'.', '..', 'control'} or not name.strip() or
                any(ord(character) < 32 or ord(character) == 127 for character in name) or
                '/' in name or chr(92) in name or Path(name).is_absolute()):
            raise ValueError('Edit names must be simple directory names other than control')
        if not isinstance(options, dict):
            raise ValueError('Each edit must contain its parameters')
        kind = options.get('kind', name)
        if kind == 'external_plan':
            if not isinstance(options.get('plan'), str) or not options['plan']:
                raise ValueError('External plans need a canonical OBJ path')
        elif kind not in {'advance', 'retract', 'narrow'} or 'strength' not in options:
            raise ValueError('Choose an external_plan or a configured advance/retract/narrow edit')
        else:
            strength = options['strength']
            if (not isinstance(strength, Real) or isinstance(strength, (bool, np.bool_))
                    or not np.isfinite(strength) or strength < 0):
                raise ValueError('Edit strength must be a finite nonnegative number')



def _output_directory(output, sources):
    output = Path(output).expanduser()
    if output.is_symlink():
        raise ValueError('Output must not be a symbolic link')
    output = output.resolve()
    for source in sources:
        if source is None:
            continue
        source = Path(source).resolve()
        if output == source or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError('Output must be separate from input assets and explicit caches')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty postoperative output directory')
    return output


def _select_plan(configuration, plan):
    if plan is None:
        return configuration
    return {'edits': {'external_plan': {'kind': 'external_plan', 'plan': str(Path(plan).resolve())}}}


def _reference(configuration, **paths):
    reference = configuration.get('reference', {})
    for name, path in paths.items():
        expected = reference.get(name + '_sha256')
        if expected is not None and _digest(path) != expected:
            raise ValueError(f'Default Plan is bound to a different {name}; provide your own --plan in this canonical frame')
        if name in {'face', 'skull'} and 'geometry' in reference:
            mesh = read_mesh(path)
            with np.load(reference['geometry'], allow_pickle=False) as saved:
                expected_vertices, expected_faces = saved[name + '_vertices'], saved[name + '_faces']
            if (mesh.vertices.shape != expected_vertices.shape or
                    not np.array_equal(mesh.faces, expected_faces) or
                    not np.allclose(mesh.vertices, expected_vertices,
                                    atol=reference.get('atol', 1e-4), rtol=reference.get('rtol', 1e-5))):
                raise ValueError(f'Default Plan requires its reference canonical {name}; provide your own --plan')
            print(f'Plan reference {name} maximum coordinate difference: '
                  f'{np.max(np.abs(mesh.vertices - expected_vertices)):.9g}')


def _validated_plan_cache(face, plan, model, seed, registration_dir):
    from bfsm_topology.plan import signature as plan_signature
    source = Path(registration_dir).resolve()
    if (_digest(source / 'skull.obj') != _digest(plan) or
            _digest(source / 'topology_face.obj') != _digest(face)):
        raise ValueError('Explicit Plan registration does not match this external Plan and canonical face')
    with np.load(source / 'registration_receipt.npz', allow_pickle=False) as receipt:
        if (not np.array_equal(receipt['sha256'], plan_signature(face, plan, deterministic=True, seed=seed)) or
                str(receipt['result_sha256']) != _digest(source / 'topology_Plan.obj') or
                int(receipt['outer_iterations']) != 150 or int(receipt['total_inner_iterations']) != 7550):
            raise ValueError('Explicit Plan cache lacks a matching full original NICP completion receipt')
    registered = read_mesh(source / 'topology_Plan.obj')
    if (len(registered.vertices) != model.skull_count or
            not np.array_equal(registered.faces, model.arrays['skull_tri'])):
        raise ValueError('Explicit Plan registration does not preserve the selected BFSM skull topology')
    return source


def _preflight_external_plans(configuration, output, *, face=None, model=None,
                              seed=0, registration_dir=None):
    external = [(name, options['plan']) for name, options in configuration['edits'].items()
                if options.get('kind', name) == 'external_plan']
    if not external:
        return
    from bfsm_topology.plan import validate_paths, validate_geometry, validate_plan_mesh
    for name, plan in external:
        validate_paths(face, plan, Path(output) / 'plan_registration' / name)
        if face is None:
            validate_plan_mesh(plan)
        else:
            validate_geometry(face, plan)
            if registration_dir is not None:
                _validated_plan_cache(face, plan, model, seed, registration_dir)


def _registered_plan(face, plan, output, model, device, seed, registration_dir=None):
    from bfsm_topology.plan import run as register_plan, validate_paths
    if registration_dir is None:
        return register_plan(face, plan, output, device=device, deterministic=True, seed=seed)
    validate_paths(face, plan, output)
    source = _validated_plan_cache(face, plan, model, seed, registration_dir)
    shutil.copytree(source, output)
    return output / 'topology_Plan.obj'


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _run_cases(face_path, skull_path, source_surface, output, configuration,
               model=None, device='cuda:0', render=True, prediction_dir=None,
               transform=None, source_kind='scan', reference_skull=None, seed=0, plan_registration_dir=None):
    """Shared edit, tissue-transport and fitting loop, with ARAP only for scans."""
    _validate_configuration(configuration)
    if any(options.get('kind') == 'external_plan' for options in configuration['edits'].values()):
        from bfsm_topology.preprocess_pipeline.reproducibility import configure
        configure(device, seed)
    _seed(seed)
    output = Path(output)
    face_path, skull_path, source_surface = map(Path, (face_path, skull_path, source_surface))
    face, skull = read_mesh(face_path), read_mesh(skull_path)
    model_path = model_file(model)
    model_data = FaceSkullModel(model_path, device='cpu')
    if (len(face.vertices) != model_data.face_count or len(skull.vertices) != model_data.skull_count or
            not np.array_equal(face.faces, model_data.arrays['face_tri']) or
            not np.array_equal(skull.faces, model_data.arrays['skull_tri'])):
        raise ValueError('Prepared face/skull topology differs from the selected BFSM model')
    _preflight_external_plans(configuration, output, face=face_path, model=model_data,
                              seed=seed, registration_dir=plan_registration_dir)
    upper = model_data.arrays['region_up']
    lower = np.setdiff1d(np.arange(len(face.vertices)), upper)
    is_scan = source_kind == 'scan'
    face_label = 'face_scan' if is_scan else 'fsmm_face'
    skull_label = 'skull' if is_scan else 'inferred_skull'
    reference_skull = skull_path if reference_skull is None else Path(reference_skull)
    _world_surface(source_surface, output / 'world/before' / f'{face_label}.obj', transform)
    _world_surface(reference_skull, output / 'world/before' / f'{skull_label}.obj', transform)
    report = {'input_kind': 'original_textured_face_scan_and_ct_skull' if is_scan else
              'photograph_and_fsmm_inferred_face_skull_without_ct',
              'surface_vertex_count': len(read_mesh(source_surface).vertices), 'seed': seed, 'cases': {}}
    control = None
    for name in ('control', *configuration['edits']):
        folder = output / 'topology' / name
        if name == 'control':
            plan = skull_path
        else:
            options = configuration['edits'][name]
            kind = options.get('kind', name)
            if kind == 'external_plan':
                plan = _registered_plan(face_path, Path(options['plan']),
                    output / 'plan_registration' / name, model_data, device, seed, plan_registration_dir)
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plan, folder / 'edited_skull.obj')
                _world_surface(Path(options['plan']), output / 'world' / name / 'edited_external_plan.obj', transform)
            else:
                mandibular_edit(face_path, skull_path, folder, kind, options['strength'])
                plan = folder / 'edited_skull.obj'
        cache = None if prediction_dir is None else Path(prediction_dir) / 'topology' / name
        predicted, record = _prediction(face_path, skull_path, plan, folder, model_path,
                                         model_data, device, cache)
        if name == 'control':
            control = predicted
        # Measured scans need detail transfer. Photo FSMM inputs already have
        # the model topology: export the original prediction without extra ARAP.
        surface = output / 'surface' / name / 'predicted_face.obj'
        if is_scan:
            surface_record = transfer_details(source_surface, face_path, folder / 'predicted_face.obj', surface)
        else:
            surface.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / 'predicted_face.obj', surface)
            surface_record = {'method': 'direct_same_topology', 'additional_deformation': False}
        world = output / 'world' / name
        _world_surface(surface, world / 'predicted_face.obj', transform)
        # This is model topology, not a claimed full-resolution bone deformation.
        _world_surface(plan, world / 'edited_topology_skull.obj', transform)
        delta, effect = predicted - face.vertices, predicted - control
        report['cases'][name] = {'prediction': record,
            **({'arap': surface_record} if is_scan else {'surface': surface_record}),
            'upper_face_rms_to_input': float(np.sqrt(np.mean(delta[upper] ** 2))),
            'lower_face_rms_to_input': float(np.sqrt(np.mean(delta[lower] ** 2))),
            'maximum_face_displacement_to_input': float(np.linalg.norm(delta, axis=1).max()),
            'maximum_edit_effect_to_fitted_control': float(np.linalg.norm(effect, axis=1).max())}
        if render:
            meshes = [(f'before_{face_label}', source_surface), (f'predicted_{face_label}', surface),
                      (f'before_{skull_label}', reference_skull), ('edited_topology_skull', plan)]
            render_comparison(meshes, folder / 'views', device=device)
            render_comparison(meshes, folder / 'geometry_views', device=device, neutral=True)
    return report


def run_scan(example, output, *, model=None, device='cuda:0', render=True,
             registration_dir=None, prediction_dir=None, config=None, seed=0, deterministic=False,
             plan=None, plan_registration_dir=None):
    """Original scan → full registration → edited skull → predicted textured scan."""
    from bfsm_topology.demo import run as register
    _seed(seed)
    example = Path(example)
    raw = example / 'raw'
    configuration = _select_plan(_configuration(config), plan)
    external = [options['plan'] for options in configuration['edits'].values() if options.get('kind') == 'external_plan']
    output = _output_directory(output, [raw, model_file(model), config, registration_dir,
                                       prediction_dir, plan_registration_dir, *external])
    _preflight_external_plans(configuration, output)
    registration = output / 'registration'
    transform = register(raw, registration, device=device, seed=seed, deterministic=deterministic,
                         registration_dir=registration_dir)
    posed = registration / 'posed'
    return _run_cases(posed / 'topology_face.obj', posed / 'topology_skull.obj',
                      posed / 'face_scan.obj', output, configuration, model, device,
                      render, prediction_dir, transform, 'scan', posed / 'skull.obj', seed, plan_registration_dir)


def run_image(image, output, *, model=None, device='cuda:0', render=True,
              reconstruction_dir=None, prediction_dir=None, config=None, seed=0,
              plan=None, plan_registration_dir=None):
    """Photo → same-topology FSMM face/skull → original fit → direct predicted face."""
    from .image_input import prepare_image
    _seed(seed)
    plan_config = config
    if plan is None:
        plan_config = config if config is not None else (ROOT / 'modules/04_postoperative/configs/image_demo.toml')
        configuration = _configuration(plan_config)
    else:
        configuration = _select_plan({}, plan)
    _reference(configuration, image=image, model=model_file(model))
    external = [options['plan'] for options in configuration['edits'].values() if options.get('kind') == 'external_plan']
    output = _output_directory(output, [image, model_file(model), plan_config, reconstruction_dir,
                                       prediction_dir, plan_registration_dir, *external])
    _preflight_external_plans(configuration, output)
    prepared = prepare_image(image, Path(output) / 'reconstruction', device=device,
                             reconstruction_dir=reconstruction_dir, seed=seed, model=model)
    _reference(configuration, face=prepared['face'], skull=prepared['skull'])
    return _run_cases(prepared['face'], prepared['skull'], prepared['face'], output,
                      configuration, model, device, render, prediction_dir,
                      prepared['transform'], 'image', seed=seed, plan_registration_dir=plan_registration_dir)


def main():
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='route', required=True)
    scan = commands.add_parser('scan', help='Original textured scan and CT skull, with full scan ARAP')
    scan.add_argument('--example', default=str(root / 'data/input/sample_001'))
    scan.add_argument('--deterministic', action='store_true', help='Use the reproducible registration renderer and solver settings')
    scan.add_argument('--registration-dir', help='Explicit original-scan registration cache; validated before use')
    image = commands.add_parser('image', help='Any photograph; inferred FSMM pair and direct same-topology prediction, with no CT or ARAP')
    image.add_argument('--image', default=str(root / 'data/input/image_clinical/1.png'))
    image.add_argument('--reconstruction-dir', help='Explicit verified image reconstruction cache')
    for command in (scan, image):
        command.add_argument('--output', required=True)
        command.add_argument('--model')
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--seed', type=int, default=0)
        command.add_argument('--config', help='Case-specific edit configuration; prediction settings are shared')
        command.add_argument('--plan', help='Your external edited skull OBJ in the SAME canonical frame; overrides configured edit')
        command.add_argument('--plan-registration-dir', help='Explicit original Plan NICP output matching the external Plan and canonical face')
        command.add_argument('--prediction-dir', help='Explicit prior output with matching verified predictions')
        command.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    options = dict(model=args.model, device=args.device, render=not args.no_render,
                   prediction_dir=args.prediction_dir, config=args.config, seed=args.seed,
                   plan=args.plan, plan_registration_dir=args.plan_registration_dir)
    if args.route == 'scan':
        run_scan(args.example, args.output, registration_dir=args.registration_dir,
                 deterministic=args.deterministic, **options)
    else:
        run_image(args.image, args.output, reconstruction_dir=args.reconstruction_dir, **options)


if __name__ == '__main__':
    main()
