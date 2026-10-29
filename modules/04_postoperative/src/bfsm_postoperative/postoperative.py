"""Original tissue-vector transport and region-constrained postoperative fit."""
import argparse
from pathlib import Path

import numpy as np

from bfsm_core.mesh import read_mesh, write_mesh
from bfsm_core.model import FaceSkullModel, fit_geometry
from bfsm_core.paths import model_file, check_output_inputs

SCHEDULE = [('rigid1', 200, 0.001, False), ('rigid2', 200, 0.0005, False),
            ('nonrigid1', 600, 0.001, True), ('nonrigid2', 1000, 0.0005, True)]
REGIONS = ('up', 'eye_left', 'eye_right', 'brow_left', 'brow_right', 'nose')
WEIGHTS = (2.4, 0.4, 0.4, 0.4, 0.4, 1.2)


def normalize(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norm < 1e-12, 1.0, norm)


def transport_tissue(before, planned, skull_indices, face_positions, legacy=False):
    old_points = before.vertices[skull_indices]
    new_points = planned.vertices[skull_indices]
    old_normals = normalize(before.vertex_normals[skull_indices])
    new_normals = normalize(planned.vertex_normals[skull_indices])
    flip = np.einsum('ij,ij->i', old_normals, new_normals) < 0
    new_normals[flip] *= -1
    offset = face_positions - old_points
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    safe = np.where(length < 1e-12, 1.0, length)
    perpendicular = np.sum(offset * old_normals, axis=1, keepdims=True) * old_normals
    tangent = normalize(offset - perpendicular)
    alpha = np.linalg.norm(perpendicular, axis=1, keepdims=True) / safe
    sign = np.sign(np.sum(offset * old_normals, axis=1, keepdims=True))
    sign[sign == 0] = 1
    projected = tangent - np.sum(tangent * new_normals, axis=1, keepdims=True) * new_normals
    degenerate = ((np.linalg.norm(projected, axis=1) < 1e-12) &
                  (np.linalg.norm(tangent, axis=1) > 1e-12))
    if degenerate.any() and not legacy:
        # Only the undefined projected direction changes: minimal normal-to-normal rotation.
        old = old_normals[degenerate]
        new = new_normals[degenerate]
        axis = np.cross(old, new)
        cosine = np.sum(old * new, axis=1, keepdims=True)
        value = tangent[degenerate]
        projected[degenerate] = (value + np.cross(axis, value) +
                                np.cross(axis, np.cross(axis, value)) / (1 + cosine))
    projected = normalize(projected)
    projected[np.isnan(projected)] = 0
    if not legacy:
        alpha = np.clip(alpha, 0, 1)
    transported = (np.sqrt(1 - alpha ** 2) * length * projected +
                   alpha * length * sign * new_normals)
    if not np.isfinite(transported).all():
        raise ValueError('Non-finite tissue transport; inspect the input normals and edit')
    metadata = {'normal_sign_flips': int(flip.sum()),
                'degenerate_tangent_fallbacks': int(degenerate.sum()),
                'legacy_formula': bool(legacy)}
    return new_points + transported, metadata


def predict(face_path, skull_path, plan_path, output, model_path=None, device='cuda:0',
            weights=WEIGHTS, legacy=False):
    model_path = model_file(model_path)
    check_output_inputs(output, ('predicted_face.obj', 'coefficients.npz', 'transported_tissue_points.npy'),
                        (face_path, skull_path, plan_path, model_path))
    face, skull, plan = (read_mesh(path) for path in (face_path, skull_path, plan_path))
    model = FaceSkullModel(model_path, device=device)
    if len(face.vertices) != model.face_count or len(skull.vertices) != model.skull_count:
        raise ValueError('Input vertex counts do not match the selected model')
    if not np.array_equal(face.faces, model.arrays['face_tri']):
        raise ValueError('Face triangle order differs from the model; do not silently reindex')
    if not np.array_equal(skull.faces, plan.faces) or len(skull.vertices) != len(plan.vertices):
        raise ValueError('Plan must preserve skull topology and vertex order')
    if not np.array_equal(skull.faces, model.arrays['skull_tri']):
        raise ValueError('Skull triangle order differs from the model; do not silently reindex')
    mapping = model.arrays['face_to_skull_ind_new']
    target, metadata = transport_tissue(skull, plan, mapping[:, 1], face.vertices[mapping[:, 0]], legacy)
    regions = [model.arrays['region_' + region] for region in REGIONS]
    predicted, _, coefficients, history = fit_geometry(
        model, target, mapping[:, 0], SCHEDULE, face.vertices, regions, weights,
        legacy_export=legacy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(face.visual.vertex_colors)[:, :3]
    write_mesh(output / 'predicted_face.obj', predicted, model.arrays['face_tri'], colors)
    np.savez(output / 'coefficients.npz', **coefficients)
    np.save(output / 'transported_tissue_points.npy', target, allow_pickle=False)
    record = {**metadata, 'components': model.components, 'schedule': SCHEDULE,
              'regions': REGIONS, 'region_weights': list(weights), 'history': history,
              'output_geometry': 'before_last_optimizer_update' if legacy else 'after_last_optimizer_update',
              'input_contract': 'preoperative_face_skull_and_topology_consistent_demonstration_plan',
              'plan_semantics': 'external_or_explicitly_synthetic_plan_not_a_clinical_prediction_claim'}
    return predicted, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face', required=True)
    parser.add_argument('--skull', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--legacy', action='store_true', help='Original export/degeneracy behaviour for comparisons only')
    args = parser.parse_args()
    predict(args.face, args.skull, args.plan, args.output, args.model, args.device, legacy=args.legacy)


if __name__ == '__main__':
    main()
