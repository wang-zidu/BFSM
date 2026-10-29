"""Reproducible, topology-preserving mandibular edits for the selected software demo."""
import argparse
from pathlib import Path

import numpy as np

from bfsm_core.mesh import read_mesh, write_mesh
from bfsm_core.paths import check_output_inputs


def smoothstep(values):
    values = np.clip(values, 0, 1)
    return values * values * (3 - 2 * values)


def preserve_orientation(vertices, faces, displacement, active, axis):
    """Project scalar edits onto triangle-orientation half-spaces.

    Moving vertices along one axis makes each triangle's normal projected on
    its old normal linear in the three scalar displacements. Small corrections
    protect sliver triangles without weakening the whole demonstration edit.
    """
    triangles = vertices[faces]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.cross(first, second)
    rest_area_squared = np.einsum('ij,ij->i', normal, normal)
    direction = np.eye(3)[axis]
    first_derivative = np.einsum('ij,ij->i', normal, np.cross(direction, second))
    second_derivative = np.einsum('ij,ij->i', normal, np.cross(first, direction))
    gradients = np.column_stack((-first_derivative-second_derivative, first_derivative, second_derivative))
    desired = displacement[:, axis].copy()
    values = desired.copy()
    for iteration in range(64):
        projected_area = rest_area_squared + np.sum(gradients * values[faces], axis=1)
        deficit = .1 * rest_area_squared - projected_area
        violated = deficit > np.maximum(1e-12 * rest_area_squared, 1e-30)
        if not violated.any():
            break
        indices = faces[violated]
        gradient = gradients[violated] * active[indices]
        denominator = np.sum(gradient * gradient, axis=1)
        if np.any(denominator == 0):
            raise ValueError('Orientation constraint has no editable vertices')
        corrections = deficit[violated, None] * gradient / denominator[:, None]
        np.add.at(values, indices.ravel(), corrections.ravel())
    else:
        raise ValueError('Edit orientation constraints did not converge; inspect the region or reduce strength')
    displacement[:, axis] = values
    adjustment = np.abs(values - desired)
    return {'minimum_old_normal_projection_fraction': .1,
            'projection_iterations': iteration, 'adjusted_vertices': int(np.count_nonzero(adjustment)),
            'maximum_scalar_adjustment': float(adjustment.max(initial=0))}


def mandibular_edit(face_path, skull_path, output, kind='advance', strength=0.06, region_path=None):
    """Strength is a fraction of face height, not an unverified millimetre value.

    Default axes match the original canonical model: X lateral, Y superior,
    Z anterior. An explicit saved vertex-weight array can replace region selection.
    """
    check_output_inputs(output, ('edited_skull.obj', 'edit_weights.npy'),
                        (face_path, skull_path, region_path))
    face, skull = read_mesh(face_path), read_mesh(skull_path)
    lower, upper = np.quantile(face.vertices[:, 1], [0.01, 0.99])
    height = float(upper - lower)
    if not np.isfinite(height) or height <= 0 or not np.isfinite(strength) or strength < 0:
        raise ValueError('Face height must be finite and positive; strength must be finite and nonnegative')
    centre_x = float(np.median(face.vertices[:, 0]))
    if region_path:
        weights = np.load(region_path, allow_pickle=False)
        if (weights.shape != (len(skull.vertices),) or not np.isfinite(weights).all()
                or np.min(weights) < 0 or np.max(weights) > 1):
            raise ValueError('Region must contain one weight in [0,1] per skull vertex')
    else:
        # Smooth lower-jaw region with the posterior cranium excluded. Selection is
        # documented and saved, and must be visually checked for this case.
        top, full = lower + 0.40 * height, lower + 0.16 * height
        vertical = smoothstep((top - skull.vertices[:, 1]) / (top - full))
        posterior, anterior = np.quantile(skull.vertices[:, 2], [0.02, 0.98])
        front = smoothstep((skull.vertices[:, 2] - (posterior + .22 * (anterior - posterior))) /
                           (.28 * (anterior - posterior)))
        weights = vertical * front
    if not np.count_nonzero(weights > 0):
        raise ValueError('The mandibular region is empty; check canonical axes')
    displacement = np.zeros_like(skull.vertices)
    if kind in {'advance', 'retract'}:
        displacement[:, 2] = weights * height * strength * (1 if kind == 'advance' else -1)
    elif kind == 'narrow':
        lateral = skull.vertices[:, 0] - centre_x
        half_width = max(float(np.quantile(np.abs(lateral), .98)), 1e-12)
        lateral_weight = smoothstep(np.abs(lateral) / half_width)
        displacement[:, 0] = -np.sign(lateral) * weights * lateral_weight * height * strength
    else:
        raise ValueError('Edit must be advance, retract or narrow')
    orientation = preserve_orientation(skull.vertices, skull.faces, displacement, weights > 0,
                                       0 if kind == 'narrow' else 2)
    edited = skull.vertices + displacement
    triangles = edited[skull.faces]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    old_triangles = skull.vertices[skull.faces]
    old_area = np.linalg.norm(np.cross(old_triangles[:, 1] - old_triangles[:, 0], old_triangles[:, 2] - old_triangles[:, 0]), axis=1)
    newly_degenerate = (double_area < 1e-12) & (old_area >= 1e-12)
    if newly_degenerate.any():
        raise ValueError('Edit created degenerate triangles; reduce magnitude or inspect the region')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    write_mesh(destination / 'edited_skull.obj', edited, skull.faces)
    np.save(destination / 'edit_weights.npy', weights, allow_pickle=False)
    record = {'kind': kind, 'strength_as_fraction_of_face_height': strength,
              'face_height_in_source_units': height, 'axes': {'x': 'lateral', 'y': 'superior', 'z': 'anterior'},
              'changed_vertices': int(np.count_nonzero(np.linalg.norm(displacement, axis=1) > 0)),
              'max_vertex_displacement': float(np.linalg.norm(displacement, axis=1).max()),
              'newly_degenerate_triangles': int(newly_degenerate.sum()),
              'orientation_guard': orientation,
              'topology_and_vertex_order': 'preserved',
              'meaning': 'synthetic_software_demonstration_edit_not_a_clinical_plan_or_observation'}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face', required=True)
    parser.add_argument('--skull', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=('advance', 'retract', 'narrow'), default='advance')
    parser.add_argument('--strength', type=float, default=.06)
    parser.add_argument('--region', help='Optional saved per-skull-vertex weights')
    args = parser.parse_args()
    mandibular_edit(args.face, args.skull, args.output, args.kind, args.strength, args.region)


if __name__ == '__main__':
    main()
