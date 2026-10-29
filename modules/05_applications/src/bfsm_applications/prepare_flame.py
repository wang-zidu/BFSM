"""Prepare licensed FLAME2020 identity, expression and joint-skinning arrays.

The prepared model remains subject to its provider terms.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from bfsm_core.paths import check_output_inputs


class StoredState:
    def __setstate__(self, state):
        self.state = state


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(np.core.multiarray, name)
        if module.startswith(('scipy.sparse', 'chumpy')):
            return StoredState
        if module in {'builtins', '__builtin__'} and name in {'set', 'slice'}:
            return {'set': set, 'slice': slice}[name]
        raise ValueError(f'Unsupported model pickle object: {module}.{name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output_file = args.output if str(args.output).endswith('.npz') else Path(str(args.output) + '.npz')
    check_output_inputs(output_file.parent, [output_file.name], [args.model])
    with args.model.open('rb') as stream:
        data = ArrayUnpickler(stream, encoding='latin1').load()
    directions = data['shapedirs']
    directions = directions.state['x'] if isinstance(directions, StoredState) else directions
    template, faces = np.asarray(data['v_template']), np.asarray(data['f'])
    if template.shape != (5023, 3) or directions.shape[:2] != template.shape or directions.shape[-1] < 300:
        raise ValueError('Expected FLAME2020 5023-point template with >=300 identity bases')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joint = data['J_regressor']
    if isinstance(joint, StoredState):
        state = joint.state
        shape = state['_shape']
        sparse = csc_matrix if len(state['indptr']) == shape[1] + 1 else csr_matrix
        joint = sparse((state['data'], state['indices'], state['indptr']), shape=shape).toarray()
    parents = np.asarray(data['kintree_table'][0], dtype=np.int64).copy()
    parents[0] = -1
    if directions.shape[2] < 350:
        raise ValueError('DECA requires 300 identity and at least 50 expression bases')
    arrays = dict(v_template=template, shapedirs=directions[:, :, :300], f=faces,
                  expression_dirs=directions[:, :, 300:350], posedirs=np.asarray(data['posedirs']),
                  joint_regressor=np.asarray(joint), parents=parents,
                  skinning_weights=np.asarray(data['weights']))
    from .deca_geometry import decode_flame
    decode_flame(arrays, np.zeros(100), np.zeros(50), np.zeros(6))
    np.savez_compressed(args.output, **arrays)


if __name__ == '__main__':
    main()
