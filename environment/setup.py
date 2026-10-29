"""Install an explicit dependency profile into its own Python 3.12 environment."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib


PROFILES = ('cpu', 'torch', 'render', 'image', 'registration', 'all')
EXTENSIONS = {
    'pytorch3d': ('https://github.com/facebookresearch/pytorch3d.git',
                  '33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'),
    'nvdiffrast': ('https://github.com/NVlabs/nvdiffrast.git',
                   '253ac4fcea7de5f396371124af597e6cc957bfae'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=PROFILES)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--cache-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / '.cache/setup')
    parser.add_argument('--torch-index', default='https://download.pytorch.org/whl/cu130')
    parser.add_argument('--cuda-arch', default='12.0', help='Verified Blackwell target; select yours explicitly')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--modules', nargs='+', metavar='ID',
                        help='Install selected local modules and their local dependencies; default: all')
    parser.add_argument('--plan', action='store_true', help='Print the exact commands without writing or installing')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('This installer requires Python 3.12')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for folder in sorted((root / 'modules').iterdir()):
        metadata = folder / 'pyproject.toml'
        if metadata.is_file():
            spec = tomllib.loads(metadata.read_text())
            packages[folder.name.split('_', 1)[0]] = (folder, spec.get('tool', {}).get('bfsm', {}).get('requires-modules', []))
    chosen = set(args.modules or packages)
    unknown = chosen - packages.keys()
    if unknown:
        parser.error('Unknown module ID(s): ' + ', '.join(sorted(unknown)))
    pending = list(chosen)
    while pending:
        for dependency in packages[pending.pop()][1]:
            if dependency not in packages:
                parser.error('Missing local dependency: ' + dependency)
            if dependency not in chosen:
                chosen.add(dependency)
                pending.append(dependency)
    prefix, cache = args.prefix.resolve(), args.cache_dir.resolve()
    if prefix in {Path('/'), Path.home()} or prefix == Path(sys.base_prefix).resolve():
        parser.error('Choose a dedicated environment prefix')
    target_python = prefix / 'bin/python'
    if target_python.is_file() and not args.plan:
        probe = subprocess.check_output([str(target_python), '-c',
            "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor)); "
            "print(sys.prefix); print(sys.prefix != sys.base_prefix)"], text=True).splitlines()
        if (len(probe) != 3 or probe[0] != '3.12' or probe[2] != 'True'
                or Path(probe[1]).resolve() != prefix):
            parser.error('Existing prefix must be its own Python 3.12 virtual environment')
    gpu = args.profile in {'render', 'image', 'registration', 'all'}
    if gpu and not args.plan and not shutil.which('nvcc') and not os.environ.get('CUDA_HOME'):
        parser.error('GPU rendering profiles require an installed CUDA toolkit; see README')
    env = os.environ.copy()
    for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_ROOT'):
        env.pop(key, None)
    env['PIP_CONFIG_FILE'] = os.devnull
    env.update({'PIP_CACHE_DIR': str(cache / 'pip'), 'TMPDIR': str(cache / 'tmp'),
                'XDG_CACHE_HOME': str(cache / 'xdg'), 'TORCH_HOME': str(cache / 'torch'),
                'TORCH_EXTENSIONS_DIR': str(cache / 'extensions'),
                'MPLCONFIGDIR': str(cache / 'matplotlib'),
                'PYTHONDONTWRITEBYTECODE': '1', 'MAX_JOBS': str(args.jobs),
                'TORCH_CUDA_ARCH_LIST': args.cuda_arch,
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'NVCC_FLAGS': '--static-global-template-stub=false --device-entity-has-hidden-visibility=false'})

    def run(command):
        print(shlex.join(map(str, command)), flush=True)
        if not args.plan:
            subprocess.run(list(map(str, command)), check=True, env=env)

    if not args.plan:
        for folder in ('pip', 'tmp', 'xdg', 'torch', 'extensions', 'matplotlib'):
            (cache / folder).mkdir(parents=True, exist_ok=True)
    if not (prefix / 'bin/python').is_file():
        if prefix.exists() and any(prefix.iterdir()):
            parser.error('Prefix exists but is not a usable environment')
        run([sys.executable, '-m', 'venv', '--without-pip', prefix])
    python = prefix / 'bin/python'
    run([python, Path(__file__).parent / 'bootstrap_pip.py', '--cache-dir', cache / 'bootstrap'])
    run([python, '-m', 'pip', 'install', 'pip==25.3', 'setuptools==84.2.1', 'wheel==0.48.0'])
    if args.profile != 'cpu':
        run([python, '-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1',
             '--index-url', args.torch_index])
    run([python, '-m', 'pip', 'install', '-r', Path(__file__).parent / 'profiles' / (args.profile + '.txt')])
    selected = ['pytorch3d'] if gpu else []
    if args.profile in {'image', 'all'}:
        selected.append('nvdiffrast')
    for name in selected:
        url, commit = EXTENSIONS[name]
        destination = cache / 'sources' / name
        if not destination.exists():
            if not args.plan:
                destination.parent.mkdir(parents=True, exist_ok=True)
            run(['git', 'clone', url, destination])
        run(['git', '-C', destination, 'checkout', '--detach', commit])
        if name == 'pytorch3d':
            env['FORCE_CUDA'] = '1'
        run([python, '-m', 'pip', 'install', '--no-build-isolation', destination])
    local = [root] + [packages[key][0] for key in sorted(chosen)]
    command = [python, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation']
    for folder in local:
        command.extend(['-e', folder])
    run(command)
    if args.profile in {'image', 'all'}:
        renderer = root / 'common/src/bfsm_core/rendering/cython_renderer'
        run([python, '-c', 'import os,runpy; os.chdir(' + repr(str(renderer)) + '); '
             + "runpy.run_path('setup.py', run_name='__main__')", 'build_ext', '--inplace'])
    run([python, '-m', 'pip', 'check'])
    print('Profile commands finished. Run the documented module checks after installing the required assets.')


if __name__ == '__main__':
    main()
