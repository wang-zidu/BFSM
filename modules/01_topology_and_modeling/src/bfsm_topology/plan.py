"""Register an external skull plan to BFSM topology in a common canonical frame.

The input plan can have different connectivity, including externally edited bone
segments. This registers the fixed template to its surface; the output does not
preserve new cut faces or independently disconnected surgical segments.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np



def signature(face, plan, *, nicp_config=None, skull_template_dir=None, deterministic=False, seed=0):
    """Bind an explicit completed registration cache to inputs and numeric code."""
    from .preprocess_pipeline import topology
    from .preprocess_pipeline.paths import BFM_LDM68_PATH, DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH, DEFAULT_NICP_CONFIG
    files = [Path(face), Path(plan), Path(nicp_config or DEFAULT_NICP_CONFIG),
             DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH, BFM_LDM68_PATH,
             Path(topology.__file__)]
    files.extend(sorted((Path(topology.__file__).parent / 'nicp_core').rglob('*.py')))
    if skull_template_dir is not None:
        files.extend(Path(skull_template_dir) / name for name in
                     ('skull_template.obj', 'up_skull_template.npy', 'middle_skull_template.npy', 'down_skull_template.npy'))
    hashes = []
    for path in files:
        with path.open('rb') as stream:
            hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest())
    return np.asarray(hashes + [str(bool(deterministic)), str(seed)])


def validate_paths(face, plan, output):
    """Check input/output paths without creating files; face may be prepared later."""
    destination = Path(output).expanduser()
    if destination.is_symlink():
        raise ValueError('Output must be a new or empty directory, not a symbolic link')
    destination = destination.resolve()
    for path in (face, plan):
        if path is None:
            continue
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != '.obj':
            raise ValueError('Face and Plan inputs must be OBJ meshes')
        if destination == source.parent or source.parent in destination.parents:
            raise ValueError(f'Plan registration output {destination} must be outside '
                             f'the input directory {source.parent}; use separate input and output directories')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError('Refusing to overwrite an existing Plan registration output')
    return destination


def validate_plan_mesh(plan):
    from bfsm_core.mesh import read_mesh
    target = read_mesh(plan)
    if (target.vertices.ndim != 2 or target.vertices.shape[1] != 3 or
            not len(target.vertices) or target.faces.ndim != 2 or
            target.faces.shape[1] != 3 or not len(target.faces) or
            target.faces.min() < 0 or target.faces.max() >= len(target.vertices)):
        raise ValueError('Plan must be a nonempty finite triangle mesh in the same canonical frame as the face')
    return target


def validate_geometry(face, plan, skull_template_dir=None):
    """Check the prepared face/template contract and Plan before optimization."""
    from bfsm_core.mesh import read_mesh
    from .preprocess_pipeline.paths import DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH
    registered_face = read_mesh(face)
    validate_plan_mesh(plan)
    template_face = read_mesh(DEFAULT_FACE_TEMPLATE_PATH)
    if (len(registered_face.vertices) != 35709 or
            not np.array_equal(registered_face.faces, template_face.faces)):
        raise ValueError('Face must be a canonical registered BFSM face in its original vertex/triangle order')
    skull_template = read_mesh(Path(skull_template_dir) / 'skull_template.obj'
                               if skull_template_dir else DEFAULT_SKULL_TEMPLATE_PATH)
    fixed_skull = read_mesh(DEFAULT_SKULL_TEMPLATE_PATH)
    if (len(skull_template.vertices) != len(fixed_skull.vertices) or
            not np.array_equal(skull_template.faces, fixed_skull.faces)):
        raise ValueError('Skull template must preserve the installed BFSM skull topology and vertex order')
    return skull_template


def run(face, plan, output, *, device='cuda:0', nicp_config=None,
        skull_template_dir=None, deterministic=False, seed=0):
    """Reuse the original Plan ray controls and skull NICP without editing inputs."""
    destination = validate_paths(face, plan, output)
    face, plan = Path(face).expanduser().resolve(), Path(plan).expanduser().resolve()
    skull_template = validate_geometry(face, plan, skull_template_dir)
    from bfsm_core.mesh import read_mesh
    from .preprocess_pipeline.topology import make_skull_control_points, run_skull_nicp
    if deterministic:
        from .preprocess_pipeline.reproducibility import configure
        configure(device, seed)
    source_signature = signature(face, plan, nicp_config=nicp_config, skull_template_dir=skull_template_dir,
                                 deterministic=deterministic, seed=seed)
    # A fresh staging directory prevents replacing source surfaces or old results.
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in ((face, 'topology_face.obj'), (plan, 'skull.obj')):
        with source.open('rb') as original, (destination / name).open('xb') as copied:
            shutil.copyfileobj(original, copied)
    # Same functions and settings as the original skull/Plan registration steps.
    make_skull_control_points(destination, template=False)
    run_skull_nicp(destination, template_dir=skull_template_dir,
                   config_path=nicp_config, device=device)
    result = destination / 'topology_skull.obj'
    fitted = read_mesh(result)
    if (len(fitted.vertices) != len(skull_template.vertices) or
            not np.array_equal(fitted.faces, skull_template.faces)):
        raise ValueError('Plan registration did not preserve the fixed BFSM skull topology')
    output_plan = destination / 'topology_Plan.obj'
    if output_plan.exists():
        raise FileExistsError(output_plan)
    result.rename(output_plan)
    from .preprocess_pipeline.topology import _load_config
    config = _load_config(nicp_config)
    with output_plan.open('rb') as stream:
        result_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    np.savez_compressed(destination / 'registration_receipt.npz', sha256=source_signature,
                        result_sha256=np.asarray(result_sha), outer_iterations=config['outer_iter'],
                        total_inner_iterations=100 + (config['outer_iter'] - 1) * config['inner_iter'])
    return output_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face', required=True, help='Registered 35709-vertex face in canonical coordinates')
    parser.add_argument('--plan', required=True, help='External edited skull OBJ in the SAME canonical coordinates')
    parser.add_argument('--output', required=True, help='New or empty directory outside both input directories')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--nicp-config', help='Original NICP configuration; default keeps all original iterations')
    parser.add_argument('--skull-template-dir', help='Matching skull_template.obj and three template control arrays')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='Seed for --deterministic')
    args = parser.parse_args()
    result = run(args.face, args.plan, args.output, device=args.device,
                 nicp_config=args.nicp_config, skull_template_dir=args.skull_template_dir,
                 deterministic=args.deterministic, seed=args.seed)
    print(result)


if __name__ == '__main__':
    main()
