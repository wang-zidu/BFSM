"""Display-only colors approximating the paper illustrations, not recovered materials."""
import numpy as np

FACE = '#91bdd4'
SKULL = '#a3a7ad'
EDITED_SKULL = '#e3bb55'


def material_hex(label):
    """Choose face, reference skull or whole edited/Plan skull from the row label."""
    name = str(label).lower()
    if 'face' in name or '脸' in name or '面部' in name:
        return FACE
    if any(word in name for word in ('edited', 'plan', 'advance', 'retract', 'narrow', '编辑', '计划')):
        return EDITED_SKULL
    if 'skull' in name or '骨' in name:
        return SKULL
    return FACE


def material_rgb(label):
    color = material_hex(label)
    return np.array([int(color[index:index + 2], 16) for index in (1, 3, 5)]) / 255.


def mesh_colors(mesh, label, neutral=False):
    """Preserve actual texture/vertex colors; give uncolored geometry semantic colors."""
    fallback = np.broadcast_to(material_rgb(label), (len(mesh.vertices), 3))
    visual = mesh.visual
    if neutral or visual.kind is None or not visual.defined:
        return fallback
    visual = visual.to_color() if visual.kind == 'texture' else visual
    colors = np.asarray(visual.vertex_colors)[:, :3] / 255.
    if colors.shape != fallback.shape or not np.isfinite(colors).all():
        return fallback
    return colors
