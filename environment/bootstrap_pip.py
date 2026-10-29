"""Bootstrap a dedicated venv without ensurepip, using a hash-verified PyPI wheel."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error('Run inside the dedicated venv, never the system interpreter')
    if importlib.util.find_spec('pip') is not None:
        return
    version = '25.3'
    with urllib.request.urlopen(f'https://pypi.org/pypi/pip/{version}/json', timeout=60) as response:
        metadata = json.load(response)
    item = next(row for row in metadata['urls'] if row['filename'] == f'pip-{version}-py3-none-any.whl')
    if not item['url'].startswith('https://files.pythonhosted.org/'):
        raise ValueError('Unexpected wheel origin')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    wheel = args.cache_dir / item['filename']
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != item['digests']['sha256']:
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != item['digests']['sha256']:
            raise ValueError('pip wheel hash mismatch')
        wheel.write_bytes(payload)
    code = ("import sys,runpy; wheel=sys.argv[1]; sys.path.insert(0,wheel); "
            "sys.argv=['pip','install','--no-index',wheel]; runpy.run_module('pip',run_name='__main__')")
    subprocess.run([sys.executable, '-c', code, str(wheel.resolve())], check=True)


if __name__ == '__main__':
    main()
