"""Resolve model assets, including legacy models/ paths."""
from pathlib import Path
import os

ROOT = Path(os.environ.get('BFSM_ROOT', Path(__file__).resolve().parents[3])).expanduser().resolve()
MODELS = ROOT / 'assets' / 'models'


def model_file(path=None, default='fsmm_200.npz'):
    result = Path(path) if path is not None else MODELS / default
    if not result.is_file() and result.resolve().parent == (ROOT / 'models').resolve():
        result = MODELS / result.name
    return result


def check_output_inputs(output, filenames, inputs):
    """Refuse a fixed export path that aliases an input asset."""
    output = Path(output)
    inputs = [Path(path).resolve() for path in inputs if path is not None]
    for name in filenames:
        destination = (output / name).resolve()
        for source in inputs:
            same_file = (destination.exists() and source.exists()
                         and destination.samefile(source))
            if destination == source or same_file:
                raise ValueError(f'Output would overwrite an input asset: {destination}')
