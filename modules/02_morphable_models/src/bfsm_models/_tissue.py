import numpy as np
import trimesh
import torch
import os
from datetime import datetime
import random

from pytorch3d.io import load_objs_as_meshes
from bfsm_core.parametric import ParametricFaceSkullModel
from bfsm_core.palette import material_rgb




def extract_vertex_colors_from_obj(file_path: str) -> np.ndarray:
    colors = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) == 7:
                    rgb = [float(x) for x in parts[-3:]]
                    colors.append(rgb)

    return np.array(colors, dtype=np.float32)


def process_face_skull(face_to_skull_ind, face_obj_path, skull_obj_path, require_colors=True):
    """
    Process face-skull correspondence and return a dictionary of numpy arrays.

    Args:
        face_to_skull_ind (np.ndarray): Face/skull vertex-index pairs
        face_obj_path (str): Path to the face OBJ file.
        skull_obj_path (str): Path to the skull OBJ file.

    Returns:
        dict[str, np.ndarray]: Dictionary containing:
            - "d_i": displacement vectors between corresponding vertices
            - "face_v": face mesh vertex coordinates
            - "skull_v": skull mesh vertex coordinates
            - "face_t": vertex texture colors
            - "t_base": flattened base texture
    """
    # 2. Load face and skull meshes
    topology_face = load_objs_as_meshes([face_obj_path], device='cpu')
    topology_skull = load_objs_as_meshes([skull_obj_path], device='cpu')
    topology_face_v = topology_face.verts_packed().numpy().copy()
    topology_skull_v = topology_skull.verts_packed().numpy().copy()

    # 3. Extract vertex color information from the face OBJ
    mesh_vertices_texture = extract_vertex_colors_from_obj(face_obj_path)
    if mesh_vertices_texture.shape != topology_face_v.shape:
        if require_colors:
            raise ValueError(f"Texture shape mismatch: {mesh_vertices_texture.shape} vs {topology_face_v.shape}")
        mesh_vertices_texture = np.broadcast_to(material_rgb('face'), topology_face_v.shape).astype(topology_face_v.dtype).copy()

    # 4. Compute displacement vectors (skull - face)
    d_i = topology_skull_v[face_to_skull_ind[:, 1]] - topology_face_v[face_to_skull_ind[:, 0]]
    d_i = d_i.reshape(1, -1)

    # 5. Construct and return the result dictionary
    result = {
        "d_i": d_i, # tissue
        "face_v": topology_face_v,
        "skull_v": topology_skull_v,
        "face_t": mesh_vertices_texture,
        "t_base": mesh_vertices_texture.reshape(1, -1),
    }

    return result



def change_texture(
    FaceSkullMorphableModel,
    face_tri,

    output_dir, 
    final_vertices,

    output_name=None,
    shapeMu = None,
    texMu = None,
    delta = 0.5
):

    os.makedirs(output_dir, exist_ok=True)
    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    texBase = FaceSkullMorphableModel['texBase']

    num_tex_components = texBase.shape[1]

    alpha_color = np.expand_dims(np.random.normal(0, delta, num_tex_components), 0)

    new_tex = texMu + np.einsum('ij,aj->ai', texBase, alpha_color)
    new_tex = new_tex.reshape(-1, 3)


    final_mesh = trimesh.Trimesh(vertices=final_vertices, faces=face_tri, vertex_colors=new_tex, process=False, maintain_order=True)
    final_mesh.export(os.path.join(output_dir, f"{output_name}.obj"))

    return os.path.join(output_dir, f"{output_name}.obj")


def change_tissue(
    FaceSkullMorphableModel,
    TissueMorphableModel,
    face_tri,
    face_to_skull_ind_new,

    skull_path, 
    output_dir, 
    alpha_shape=None, 
    alpha_color=None, 
    alpha_scale=None, 
    output_name=None,
    use_texture=True,
    shapeMu = None,
    texMu = None,
    delta_shape = 0.5,
    model_path='./assets/BFSM/FaceSkullMorphableModel.npy',
    device='cuda',
    legacy_cache=False,
):
    os.makedirs(output_dir, exist_ok=True)
    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    shapeBase = TissueMorphableModel['shapeBase']
    num_shape_components = shapeBase.shape[1]

    if alpha_shape is None:
        alpha_shape = np.expand_dims(np.random.normal(0, delta_shape, num_shape_components), 0)
    else:
        alpha_shape = np.expand_dims(alpha_shape, 0)

    if use_texture:
        texBase = FaceSkullMorphableModel['texBase']
        num_tex_components = texBase.shape[1]
        if alpha_color is None:
            alpha_color = np.expand_dims(np.random.normal(0, 0.15, num_tex_components), 0)
        else:
            alpha_color = np.expand_dims(alpha_color, 0)

    if alpha_scale is None:
        alpha_scale = random.gauss(mu=1.0, sigma=0.15)

    new_shape = (shapeMu + np.einsum('ij,aj->ai', shapeBase, alpha_shape)) * alpha_scale
    new_shape = new_shape.reshape(-1, 3)

    if use_texture:
        new_tex = (texMu + np.einsum('ij,aj->ai', texBase, alpha_color)).reshape(-1, 3)
    else:
        new_tex = np.broadcast_to(material_rgb('face'), (int(np.max(face_tri)) + 1, 3)).copy()

    base_skull = load_objs_as_meshes([skull_path], device='cpu')
    base_skull_mesh = trimesh.Trimesh(vertices=base_skull.verts_packed().numpy().copy(), 
                                      faces=base_skull.faces_packed().numpy().copy(), 
                                      process=False, maintain_order=True)

    v_start = base_skull_mesh.vertices[face_to_skull_ind_new[:, 1]]
    v_end = v_start - new_shape

    cloud = trimesh.PointCloud(vertices=v_end)
    cloud.export(os.path.join(output_dir, f"{output_name}_tissue_points.ply"))

    face_skull_model = ParametricFaceSkullModel(model_path=model_path)
    face_skull_model.to(torch.device(device))

    gt_f_v = torch.tensor(v_end, device=device, dtype=torch.float32).unsqueeze(0)
    face_to_skull_ind_new_torch = torch.from_numpy(face_to_skull_ind_new[:, 0]).to(torch.long).to(gt_f_v.device)

    coeffs_dict = {
        'id': torch.zeros((1, face_skull_model.n_components), requires_grad=True, device=device),
        'tex': torch.zeros((1, face_skull_model.n_components), requires_grad=True, device=device),
        'angle': torch.zeros((1, 3), requires_grad=True, device=device),
        'scale': torch.ones((1, 1), requires_grad=True, device=device),
        'trans': torch.zeros((1, 3), requires_grad=True, device=device),
        'gamma': torch.zeros((1, 27), requires_grad=True, device=device),
    }

    schedule = [
        ('step1_rigid', 500, 0.01, ['angle', 'scale', 'trans', 'gamma', 'tex']),
        ('step2_rigid', 500, 0.005, ['angle', 'scale', 'trans', 'gamma', 'tex']),
        ('step3_non-rigid', 500, 0.01, ['angle', 'scale', 'trans', 'gamma', 'tex', 'id']),
        ('step4_non-rigid', 1500, 0.005, ['angle', 'scale', 'trans', 'gamma', 'tex', 'id']),
        ('step5_non-rigid', 2000, 0.0001, ['angle', 'scale', 'trans', 'gamma', 'tex', 'id'])
    ]


    for stage, steps, lr_opt, keys in schedule:
        optimizer = torch.optim.Adam([coeffs_dict[k] for k in keys], lr=lr_opt)
        for iii in range(steps):
            coef_dict = {k: v for k, v in coeffs_dict.items()}
            f_v_ori, f_c, s_v = face_skull_model.compute_for_all_shape(coef_dict)
            f_v = f_v_ori[:, face_to_skull_ind_new_torch, :]

            loss_v = torch.mean((f_v - gt_f_v) ** 2)
            loss = loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iii+1) % 50 == 0:
                print(f"{stage} Iter {iii+1}/{steps}, Loss: {loss.item():.6f}")

    if not legacy_cache:
        with torch.no_grad():
            f_v_ori, _, _ = face_skull_model.compute_for_all_shape(coeffs_dict)
    final_vertices = f_v_ori.detach().cpu().numpy()[0]
    np.savez_compressed(os.path.join(output_dir, f'{output_name}_parameters.npz'),
                        **{key: value.detach().cpu().numpy() for key, value in coeffs_dict.items()},
                        tissue_coefficients=alpha_shape, tissue_scale=np.asarray(alpha_scale),
                        target_points=v_end, face_vertices=final_vertices)
    final_mesh = trimesh.Trimesh(vertices=final_vertices, faces=face_tri, vertex_colors=new_tex, process=False, maintain_order=True)
    final_mesh.export(os.path.join(output_dir, f"{output_name}.obj"))

    return os.path.join(output_dir, f"{output_name}.obj")

