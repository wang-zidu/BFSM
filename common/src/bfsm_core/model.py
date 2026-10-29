"""Geometry-only decoding of the original shared face/skull identity model."""
from pathlib import Path
from .paths import model_file

import numpy as np
import torch


def validate_model_arrays(arrays):
    """Reject malformed geometry/index payloads before constructing decoder tensors."""
    counts = []
    for key, default in (('face_vertex_count', 35709), ('skull_vertex_count', 23208)):
        count = np.asarray(arrays.get(key, default))
        if count.ndim != 0 or count.dtype.kind not in 'iu' or int(count) < 1:
            raise ValueError('Model vertex counts must be positive integer scalars')
        counts.append(int(count))
    face_count, skull_count = counts
    rows = 3 * (face_count + skull_count)
    for key in ('shapeMU', 'shapeBase'):
        value = np.asarray(arrays[key])
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Model mean and basis must contain finite numeric values')
        with np.errstate(over='ignore', invalid='ignore'):
            finite32 = np.isfinite(value.astype(np.float32)).all()
        if not finite32:
            raise ValueError('Model geometry must remain finite in float32')
        if key == 'shapeMU' and value.size != rows:
            raise ValueError('Joint model mean does not match face/skull vertex counts')
        if key == 'shapeBase' and (value.ndim != 2 or value.shape[0] != rows or value.shape[1] < 1):
            raise ValueError('Model basis must have one row per coordinate and at least one component')
    if 'recenter_offset' in arrays:
        center = np.asarray(arrays['recenter_offset'])
        if center.size != 3 or center.dtype.kind not in 'fiu' or not np.isfinite(center).all():
            raise ValueError('Model recenter offset must contain three finite coordinates')
        with np.errstate(over='ignore', invalid='ignore'):
            if not np.isfinite(center.astype(np.float32)).all():
                raise ValueError('Model recenter offset must remain finite in float32')
    for key, count in (('face_tri', face_count), ('face_buf', face_count), ('skull_tri', skull_count)):
        if key not in arrays:
            continue
        triangles = np.asarray(arrays[key])
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
                or triangles.dtype.kind not in 'iu' or triangles.min() < 0 or triangles.max() >= count):
            raise ValueError('Model triangles must be nonempty integer arrays in vertex range')
    if 'point_buf' in arrays:
        indices = np.asarray(arrays['point_buf'])
        faces = arrays.get('face_buf', arrays.get('face_tri'))
        if (faces is None or indices.ndim != 2 or indices.shape[0] != face_count
                or indices.shape[1] < 1 or indices.dtype.kind not in 'iu'
                or indices.min() < 0 or indices.max() > len(faces)):
            raise ValueError('Model point buffer must contain valid face indices or its padding sentinel')
    if 'face_to_skull_ind_new' in arrays:
        pairs = np.asarray(arrays['face_to_skull_ind_new'])
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or pairs.dtype.kind not in 'iu' or pairs.min() < 0
                or pairs[:, 0].max() >= face_count or pairs[:, 1].max() >= skull_count):
            raise ValueError('Model pairs must be integer face/skull vertex indices in range')
    return face_count, skull_count


class FaceSkullModel:
    def __init__(self, path=None, device='cpu', recenter=True):
        path = model_file(path)
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        self.device = torch.device(device)
        self.face_count, self.skull_count = validate_model_arrays(self.arrays)
        mean = self.arrays['shapeMU'].astype(np.float32).reshape(-1, 3).copy()
        if len(mean) != self.face_count + self.skull_count:
            raise ValueError('Joint model mean does not match face/skull vertex counts')
        # A topology-derived version retains the ORIGINAL joint centre, not a centre
        # recomputed after changing the number/density of face vertices.
        centre = self.arrays.get('recenter_offset', mean.mean(axis=0, keepdims=True))
        self.center = np.asarray(centre, dtype=np.float32).reshape(1, 3) if recenter else np.zeros((1, 3), np.float32)
        mean -= self.center
        if not np.isfinite(self.center).all() or not np.isfinite(mean).all():
            raise ValueError('Recentered model geometry must remain finite in float32')
        self.mean = torch.as_tensor(mean.reshape(1, -1), device=self.device)
        self.base = torch.as_tensor(self.arrays['shapeBase'].astype(np.float32), device=self.device)
        self.components = self.base.shape[1]
        split = self.face_count * 3
        self.face_base, self.skull_base = self.base[:split], self.base[split:]
        self.face_mean, self.skull_mean = self.mean[:, :split], self.mean[:, split:]

    def rotation(self, angles):
        batch = angles.shape[0]
        one = torch.ones((batch, 1), device=self.device)
        zero = torch.zeros((batch, 1), device=self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:]
        rx = torch.cat([one, zero, zero, zero, torch.cos(x), -torch.sin(x),
                        zero, torch.sin(x), torch.cos(x)], 1).reshape(batch, 3, 3)
        ry = torch.cat([torch.cos(y), zero, torch.sin(y), zero, one, zero,
                        -torch.sin(y), zero, torch.cos(y)], 1).reshape(batch, 3, 3)
        rz = torch.cat([torch.cos(z), -torch.sin(z), zero, torch.sin(z), torch.cos(z),
                        zero, zero, zero, one], 1).reshape(batch, 3, 3)
        return (rz @ ry @ rx).permute(0, 2, 1)

    def decode(self, coefficients):
        identity = coefficients['id']
        face = torch.einsum('ij,aj->ai', self.face_base, identity) + self.face_mean
        skull = torch.einsum('ij,aj->ai', self.skull_base, identity) + self.skull_mean
        scale = coefficients['scale']
        face = (scale * face).reshape(-1, self.face_count, 3)
        skull = (scale * skull).reshape(-1, self.skull_count, 3)
        rotation = self.rotation(coefficients['angle'])
        translation = coefficients['trans'].unsqueeze(1)
        return face @ rotation + translation, skull @ rotation + translation

    def initial_coefficients(self):
        return {
            'id': torch.zeros((1, self.components), device=self.device, requires_grad=True),
            'angle': torch.zeros((1, 3), device=self.device, requires_grad=True),
            'scale': torch.ones((1, 1), device=self.device, requires_grad=True),
            'trans': torch.zeros((1, 3), device=self.device, requires_grad=True),
        }


def fit_geometry(model, target, face_indices, schedule, before=None, regions=(),
                 region_weights=(), legacy_export=False, log_every=100):
    """Fit geometry with staged Adam updates and optional region constraints."""
    target = torch.as_tensor(target, dtype=torch.float32, device=model.device)
    if target.ndim != 2 or target.shape[1] != 3 or not len(target) or not torch.isfinite(target).all():
        raise ValueError('Target must contain finite three-dimensional points')

    def index_vector(values):
        value = torch.as_tensor(values, device=model.device)
        if (value.ndim != 1 or not len(value) or value.is_floating_point() or value.is_complex()
                or value.dtype == torch.bool):
            raise ValueError('Face and region indices must be nonempty integer vectors in range')
        value = value.to(torch.long)
        if value.min() < 0 or value.max() >= model.face_count:
            raise ValueError('Face and region indices must be nonempty integer vectors in range')
        return value

    indices = index_vector(face_indices)
    if len(indices) != len(target):
        raise ValueError('Each target point must have one corresponding face index')
    target = target.unsqueeze(0)
    before_tensor = None if before is None else torch.as_tensor(before, dtype=torch.float32, device=model.device).unsqueeze(0)
    if before_tensor is not None and (before_tensor.shape != (1, model.face_count, 3)
                                     or not torch.isfinite(before_tensor).all()):
        raise ValueError('Reference face must match the model and contain finite vertices')
    regions = [index_vector(region) for region in regions]
    weights = np.asarray(region_weights, dtype=float)
    if weights.shape != (len(regions),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Region weights must be finite, nonnegative and match the region list')
    if regions and before_tensor is None:
        raise ValueError('Reference face is required for region constraints')
    if not isinstance(log_every, (int, np.integer)) or log_every < 1:
        raise ValueError('Logging interval must be a positive integer')
    schedule = list(schedule)
    for name, steps, rate, optimize_identity in schedule:
        if (not isinstance(steps, (int, np.integer)) or isinstance(steps, bool) or steps < 0
                or not np.isfinite(rate) or rate < 0):
            raise ValueError('Fit stages require nonnegative integer iterations and finite nonnegative rates')
    coefficients = model.initial_coefficients()
    history = []
    updated = False
    for name, steps, rate, optimize_identity in schedule:
        keys = ['angle', 'scale', 'trans'] + (['id'] if optimize_identity else [])
        optimizer = torch.optim.Adam([coefficients[key] for key in keys], lr=rate)
        for step in range(steps):
            face, skull = model.decode(coefficients)
            terms = [torch.mean((face[:, indices] - target) ** 2)]
            if regions and before_tensor is None:
                raise ValueError('Reference face is required for region constraints')
            for region, weight in zip(regions, region_weights):
                terms.append(torch.mean((face[:, region] - before_tensor[:, region]) ** 2) * weight)
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise ValueError('Fit objective became non-finite; check input scale and weights')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updated = True
            if (step + 1) % log_every == 0 or step + 1 == steps:
                record = {'stage': name, 'iteration': step + 1, 'loss': float(loss.detach())}
                history.append(record)
                print(f'{name} {step + 1}/{steps} loss={record["loss"]:.8g}', flush=True)
    # Decode the final updated coefficients unless legacy export is requested.
    if not legacy_export or not updated:
        with torch.no_grad():
            face, skull = model.decode(coefficients)
    if any(not torch.isfinite(value).all() for value in (face, skull, *coefficients.values())):
        raise ValueError('Fit produced non-finite geometry or coefficients; check input scale and step sizes')
    return (face.detach().cpu().numpy()[0], skull.detach().cpu().numpy()[0],
            {key: value.detach().cpu().numpy() for key, value in coefficients.items()}, history)
