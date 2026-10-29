"""FLAME expression, articulated pose and DECA weak-perspective projection."""
import numpy as np
from scipy.spatial.transform import Rotation


def decode_flame(model, shape, expression, pose):
    """Evaluate shape/expression blend shapes, pose correctives and joint skinning.

    DECA supplies root and jaw axis-angle rotations; neck and eye rotations are
    zero. Returned vertices are in the posed FLAME model frame, before camera.
    """
    required = ('v_template', 'shapedirs', 'expression_dirs', 'posedirs',
                'joint_regressor', 'parents', 'skinning_weights')
    if any(key not in model for key in required):
        raise ValueError('Rebuild the FLAME asset with bfsm_applications.prepare_flame; expression and skinning arrays are required')
    data = {}
    for key in required:
        value = np.asarray(model[key])
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'FLAME {key} must contain finite real numbers')
        data[key] = value
    template, identity, expr, corrective, joint_map, parents, weights = (data[key] for key in required)
    if (template.shape != (5023, 3) or identity.ndim != 3 or identity.shape[:2] != template.shape
            or identity.shape[2] < 100 or expr.ndim != 3 or expr.shape[:2] != template.shape
            or expr.shape[2] < 50 or corrective.shape != (5023, 3, 36)
            or joint_map.shape != (5, 5023) or weights.shape != (5023, 5)
            or parents.shape != (5,) or parents.dtype.kind not in 'iu'
            or parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, 5))):
        raise ValueError('Expected DECA-compatible five-joint FLAME geometry')
    if (weights < 0).any() or not np.allclose(weights.sum(1), 1., atol=1e-5):
        raise ValueError('FLAME skinning weights must be nonnegative and sum to one')
    coefficients = []
    for value, length in ((shape, 100), (expression, 50), (pose, 6)):
        value = np.asarray(value)
        if value.shape != (length,) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('DECA requires finite shape100, expression50 and pose6 vectors')
        value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError('DECA coefficients must remain finite in float32')
        coefficients.append(value)
    shape, expression, pose = coefficients
    template, identity, expr, corrective, joint_map, weights = (
        np.asarray(value, dtype=np.float32) for value in
        (template, identity, expr, corrective, joint_map, weights))
    shaped = template + identity[:, :, :100] @ shape + expr[:, :, :50] @ expression
    joints = joint_map @ shaped
    angles = np.zeros((5, 3), np.float32)
    angles[0], angles[2] = pose[:3], pose[3:]
    rotations = Rotation.from_rotvec(angles).as_matrix().astype(np.float32)
    pose_offsets = corrective @ (rotations[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
    posed = shaped + pose_offsets
    transforms = np.zeros((5, 4, 4), np.float32)
    for i in range(5):
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = rotations[i]
        local[:3, 3] = joints[i] if i == 0 else joints[i] - joints[parents[i]]
        transforms[i] = local if i == 0 else transforms[parents[i]] @ local
    transforms[:, :3, 3] -= np.einsum('nij,nj->ni', transforms[:, :3, :3], joints)
    blended = np.einsum('vj,jab->vab', weights, transforms)
    vertices = np.einsum('vij,vj->vi', blended[:, :3, :3], posed) + blended[:, :3, 3]
    if not np.isfinite(vertices).all():
        raise ValueError('Decoded FLAME geometry is not finite')
    return vertices


def project_pixels(vertices, camera, resolution=224):
    """DECA: s*(x+tx,y+ty,z), flip Y for image rows, then map [-1,1]."""
    vertices, camera = np.asarray(vertices), np.asarray(camera)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or vertices.dtype.kind not in 'fiu' or not np.isfinite(vertices).all()
            or camera.shape != (3,) or camera.dtype.kind not in 'fiu'
            or not np.isfinite(camera).all() or camera[0] <= 0
            or isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2):
        raise ValueError('Projection requires finite XYZ vertices and a positive-scale DECA camera')
    vertices, camera = vertices.astype(float), camera.astype(float)
    projected = camera[0] * (vertices[:, :2] + camera[None, 1:])
    projected[:, 1] *= -1
    projected = (projected + 1) * (resolution / 2)
    if not np.isfinite(projected).all():
        raise ValueError('Projected DECA coordinates must remain finite')
    return projected
