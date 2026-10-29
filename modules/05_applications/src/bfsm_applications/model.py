"""Fixed topology projection and an independently parameterized tissue model."""
from pathlib import Path
import os
from bfsm_core.paths import MODELS, model_file
import numpy as np
from scipy import sparse




def real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite real numbers')
    return array


def indices(value, name, limit=None):
    array = np.asarray(value)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in 'iu' or
            array.min() < 0 or len(np.unique(array)) != len(array) or
            (limit is not None and array.max() >= limit)):
        raise ValueError(f'{name} must be unique integer indices in range')
    return array


def triangles(value, count, name):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1] != 3 or not len(array) or
            array.dtype.kind not in 'iu' or array.min() < 0 or array.max() >= count):
        raise ValueError(f'{name} must contain integer triangles in vertex range')
    return array


def geometry(archive, count):
    mean = real_array(archive['shapeMU'], 'Model mean')
    basis = real_array(archive['shapeBase'], 'Model basis')
    if (mean.shape != (1, 3 * count) or basis.ndim != 2 or
            basis.shape[0] != 3 * count or basis.shape[1] < 1):
        raise ValueError('Model mean/basis dimensions must match its vertices and components')
    return mean.reshape(count, 3), basis.reshape(count, 3, basis.shape[1])


def matrix(archive, name):
    shape = np.asarray(archive[name + '_shape'])
    data = real_array(archive[name + '_data'], name + ' weights')
    column, pointer = np.asarray(archive[name + '_indices']), np.asarray(archive[name + '_indptr'])
    if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0)):
        raise ValueError(f'{name} requires two positive integer dimensions')
    rows, columns = map(int, shape)
    if (data.ndim != 1 or column.ndim != 1 or column.dtype.kind not in 'iu' or
            pointer.shape != (rows + 1,) or pointer.dtype.kind not in 'iu' or
            len(column) != len(data) or pointer[0] != 0 or pointer[-1] != len(data) or
            np.any(pointer[1:] < pointer[:-1]) or np.any(column < 0) or np.any(column >= columns)):
        raise ValueError(f'{name} contains invalid CSR pointers or column indices')
    result = sparse.csr_matrix((data, column, pointer), shape=(rows, columns))
    result.check_format(full_check=True)
    return result


class NativeFace:
    def __init__(self, path=None):
        self.source_path = model_file(path if path is not None else
            Path(os.environ.get('VT_FLAME_MODEL_DIR', str(MODELS))) / 'flame_fsmm_200.npz')
        with np.load(self.source_path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        a = self.arrays
        counts = [np.asarray(a[key]) for key in ('face_vertex_count', 'skull_vertex_count')]
        if any(value.ndim != 0 or value.dtype.kind not in 'iu' or value <= 0 for value in counts):
            raise ValueError('Model vertex counts must be positive integer scalars')
        self.face_count, self.skull_count = map(int, counts)
        self.ids = indices(a['flame_vertex_ids'], 'FLAME vertex IDs', 5023)
        self.mapping = matrix(a, 'A')
        if len(self.ids) != self.face_count or self.mapping.shape != (self.face_count, 35709):
            raise ValueError('Native face IDs and correspondence must match its vertex count')
        self.faces = triangles(a['face_tri'], self.face_count, 'Face topology')
        triangles(a['skull_tri'], self.skull_count, 'Skull topology')
        self.center = real_array(a['recenter_offset'], 'Model center')
        if self.center.shape not in ((3,), (1, 3)):
            raise ValueError('Model center must contain one three-dimensional point')
        self.center = self.center.reshape(1, 3)
        self.mean, self.basis = geometry(a, self.face_count + self.skull_count)

    def decode(self, identity=None, recenter=True):
        identity = np.zeros(self.basis.shape[-1]) if identity is None else real_array(identity, 'Identity coefficients')
        if identity.shape != (self.basis.shape[-1],) or not np.isfinite(identity).all():
            raise ValueError('Identity must match the model component count')
        vertices = self.mean + self.basis @ identity
        if recenter:
            vertices = vertices - self.center
        real_array(vertices, 'Decoded geometry')
        return vertices[:self.face_count], vertices[self.face_count:]

    def project(self, old_face):
        old_face = real_array(old_face, 'BFSM face')
        if old_face.shape != (self.mapping.shape[1], 3):
            raise ValueError('Expected BFSM face vertices in original order')
        return real_array(self.mapping @ old_face, 'Projected face')


class NativeTissue:
    """d = weighted skull anchor minus face; b is independent of FSMM identity."""
    def __init__(self, path=None):
        self.source_path = model_file(path if path is not None else
            Path(os.environ.get('VT_FLAME_MODEL_DIR', str(MODELS))) / 'flame_tmm_200.npz')
        with np.load(self.source_path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        a = self.arrays
        self.face_indices = indices(a['native_face_indices'], 'Native tissue face indices', 5023)
        self.ids = indices(a['flame_vertex_ids'], 'Tissue FLAME vertex IDs', 5023)
        self.anchor = matrix(a, 'H')
        count = len(self.face_indices)
        if len(self.ids) != count or self.anchor.shape[0] != count:
            raise ValueError('Tissue support and anchors must have the same row count')
        self.mean, self.basis = geometry(a, count)

    def validate_face(self, face):
        if (self.face_indices.max() >= face.face_count or
                not np.array_equal(face.ids[self.face_indices], self.ids) or
                self.anchor.shape[1] != face.skull_count):
            raise ValueError('TMM support and skull anchors do not match the native face model')

    def decode(self, skull, coefficients=None, thickness=1.):
        skull = real_array(skull, 'BFSM skull')
        if skull.shape != (self.anchor.shape[1], 3):
            raise ValueError('Expected original-order BFSM skull')
        coefficients = np.zeros(self.basis.shape[-1]) if coefficients is None else real_array(coefficients, 'Tissue coefficients')
        if coefficients.shape != (self.basis.shape[-1],) or not np.isfinite(coefficients).all():
            raise ValueError('Tissue coefficients must match model dimensions')
        if np.asarray(thickness).ndim != 0 or not np.isrealobj(thickness) or not np.isfinite(thickness) or thickness <= 0:
            raise ValueError('Thickness must be finite and positive')
        displacement = (self.mean + self.basis @ coefficients) * thickness
        anchor = self.anchor @ skull
        real_array(displacement, 'Decoded tissue vectors')
        real_array(anchor, 'Skull anchors')
        return real_array(anchor - displacement, 'Decoded tissue face'), anchor, displacement
