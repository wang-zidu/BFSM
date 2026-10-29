"""Fit original-topology tissue targets before projecting the complete face."""
import numpy as np
from scipy import sparse

from bfsm_core.model import FaceSkullModel, fit_geometry
from bfsm_core.paths import model_file
from bfsm_core.tissue import TISSUE_SCHEDULE
from .model import real_array, matrix


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _equal(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise ValueError(f'Native model does not match source BFSM {name}; rebuild the native models')


def _sparse_equal(actual, expected, name):
    if actual.shape != expected.shape:
        raise ValueError(f'Native {name} dimensions do not match the source models')
    difference = actual - expected
    if difference.nnz and np.any(difference.data != 0):
        raise ValueError(f'Native {name} does not match the source model pairs')


class TissueFaceFitter:
    """Use all original paired observations and the full original 5000-step fit.

    Inputs and returned face/skull are in the native centered model frame.
    Fitting occurs in the original model frame, before a single frame shift and
    projection through A. Native tissue endpoints remain diagnostics only.
    """
    def __init__(self, native_face, native_tissue, source_fsmm_path=None,
                 source_tmm_path=None, device='cpu'):
        self.native_face, self.native_tissue = native_face, native_tissue
        native_tissue.validate_face(native_face)
        self.source_fsmm_path = model_file(source_fsmm_path)
        self.source_tmm_path = model_file(source_tmm_path, 'tmm_200.npz')
        self.source_model = FaceSkullModel(self.source_fsmm_path, device=device)
        self.fsmm, self.tmm = _load(self.source_fsmm_path), _load(self.source_tmm_path)
        f, t = self.fsmm, self.tmm
        nf, ns = self.source_model.face_count, self.source_model.skull_count
        if (nf != native_face.mapping.shape[1] or ns != native_face.skull_count or
                self.source_model.components != native_face.basis.shape[-1]):
            raise ValueError('Source BFSM and native model vertex/component counts differ')
        mean = real_array(f['shapeMU'], 'Source FSMM mean').reshape(nf + ns, 3)
        basis = real_array(f['shapeBase'], 'Source FSMM basis').reshape(nf + ns, 3, -1)
        self.center = np.asarray(f.get('recenter_offset', mean.mean(axis=0, keepdims=True))).reshape(1, 3)
        _equal(native_face.center, self.center, 'joint recenter offset')
        _equal(native_face.mean[:native_face.face_count],
               (native_face.mapping @ mean[:nf]).astype(np.float32), 'mapped face mean')
        _equal(native_face.basis[:native_face.face_count],
               (native_face.mapping @ basis[:nf].reshape(nf, -1)).reshape(
                   native_face.face_count, 3, -1).astype(np.float32), 'mapped face basis')
        _equal(native_face.mean[native_face.face_count:], mean[nf:].astype(np.float32), 'skull mean')
        _equal(native_face.basis[native_face.face_count:], basis[nf:].astype(np.float32), 'skull basis')
        self.mean_skull = mean[nf:].copy()
        if (native_face.mapping.data < 0).any() or not np.allclose(
                np.asarray(native_face.mapping.sum(1)).ravel(), 1., atol=1e-10, rtol=0):
            raise ValueError('Native A must have nonnegative, unit-sum rows')
        for key in ('face_tri', 'skull_tri'):
            _equal(f[key], t[key], 'FSMM/TMM ' + key)
        _equal(native_face.arrays['skull_tri'], f['skull_tri'], 'skull topology')
        pairs = np.asarray(t['face_to_skull_ind_new'])
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or
                pairs.dtype.kind not in 'iu' or pairs.min() < 0 or
                pairs[:, 0].max() >= nf or pairs[:, 1].max() >= ns):
            raise ValueError('Source TMM pairs must be integer face/skull indices in range')
        if 'face_to_skull_ind_new' in f:
            _equal(f['face_to_skull_ind_new'], pairs, 'ordered pair records')
        pairs = pairs.astype(np.int64, copy=False)
        self.pairs = pairs
        self.tissue_mean = real_array(t['shapeMU'], 'Source TMM mean')
        self.tissue_basis = real_array(t['shapeBase'], 'Source TMM basis')
        if (self.tissue_mean.shape != (1, 3 * len(pairs)) or
                self.tissue_basis.shape != (3 * len(pairs), native_tissue.basis.shape[-1])):
            raise ValueError('Source TMM dimensions must match all pair records and native components')
        counts = np.bincount(pairs[:, 0], minlength=nf)
        aggregation = sparse.csr_matrix((1 / counts[pairs[:, 0]],
            (pairs[:, 0], np.arange(len(pairs)))), shape=(nf, len(pairs)))
        support = np.asarray((native_face.mapping @ (counts == 0).astype(float)) == 0).ravel()
        _equal(native_tissue.face_indices, np.flatnonzero(support), 'tissue support')
        combined = (native_face.mapping[support] @ aggregation).tocsr()
        selection = sparse.csr_matrix((np.ones(len(pairs)),
            (np.arange(len(pairs)), pairs[:, 1])), shape=(len(pairs), ns))
        for name, expected in [('R', aggregation), ('C', combined), ('H', (combined @ selection).tocsr())]:
            _sparse_equal(matrix(native_tissue.arrays, name), expected, name)
        _equal(native_tissue.mean, (combined @ self.tissue_mean.reshape(-1, 3)).astype(np.float32),
               'mapped tissue mean')
        _equal(native_tissue.basis, (combined @ self.tissue_basis.reshape(len(pairs), -1)).reshape(
            len(native_tissue.ids), 3, -1).astype(np.float32), 'mapped tissue basis')

    @property
    def input_paths(self):
        return [self.native_face.source_path, self.native_tissue.source_path,
                self.source_fsmm_path, self.source_tmm_path]

    def fit(self, skull_centered, coefficients, thickness=1., displacement_mean=None):
        skull = real_array(skull_centered, 'Fixed skull')
        coefficients = real_array(coefficients, 'Tissue coefficients')
        if skull.shape != (self.native_face.skull_count, 3):
            raise ValueError('Fixed skull must match the original-order native skull')
        if coefficients.shape != (self.tissue_basis.shape[-1],):
            raise ValueError('Tissue coefficients must match the source TMM components')
        scale = real_array(thickness, 'Thickness')
        if scale.ndim != 0 or scale <= 0:
            raise ValueError('Thickness must be a positive finite scalar')
        raw_skull = skull + self.center
        # Restore the exact raw mean for the original population-mode oracle.
        centered_mean = (self.native_face.mean[self.native_face.face_count:].astype(skull.dtype)
                         - self.center.astype(skull.dtype))
        if displacement_mean is None and np.array_equal(skull, centered_mean):
            raw_skull = self.mean_skull.copy()
        baseline = self.tissue_mean
        if displacement_mean is not None:
            baseline = real_array(displacement_mean, 'Conditioned tissue displacement')
            if baseline.shape not in ((1, 3 * len(self.pairs)), (len(self.pairs), 3)):
                raise ValueError('Conditioned displacement must contain every original pair in order')
        if coefficients.dtype == self.tissue_basis.dtype:
            vector = baseline.reshape(-1) + self.tissue_basis @ coefficients
        else:
            vector = baseline.reshape(-1) + np.einsum(
                'ij,aj->ai', self.tissue_basis, coefficients[None])[0]
        displacement = (vector * thickness).reshape(-1, 3)
        target = real_array(raw_skull[self.pairs[:, 1]] - displacement, 'Original tissue targets')
        raw_face, _, fitted, history = fit_geometry(self.source_model, target,
            self.pairs[:, 0], TISSUE_SCHEDULE, log_every=500)
        source_face = raw_face - self.center
        face = self.native_face.project(source_face)
        if displacement_mean is None:
            endpoints, anchors, native_displacement = self.native_tissue.decode(skull, coefficients, thickness)
        else:
            anchors = self.native_tissue.anchor @ skull
            native_displacement = matrix(self.native_tissue.arrays, 'C') @ displacement
            endpoints = anchors - native_displacement
        residual = raw_face[self.pairs[:, 0]] - target
        return {'face': face, 'source_face': source_face, 'source_face_raw': raw_face,
                'skull': skull.copy(), 'target_points': target,
                'coefficients': coefficients, 'fit_coefficients': fitted, 'history': history,
                'target_rms_coordinate_error': float(np.sqrt(np.mean(residual ** 2))),
                'endpoints': endpoints, 'anchors': anchors, 'displacement': native_displacement}
