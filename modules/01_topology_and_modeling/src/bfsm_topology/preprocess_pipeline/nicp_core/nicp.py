import numpy as np
import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import corresponding_points_alignment, knn_gather, knn_points
from tqdm import tqdm

from .local_affine import LocalAffine
from .utils import batch_vertex_sample, convert_mesh_to_pcl, mesh_boundary


def non_rigid_icp_mesh2mesh(template_mesh, target_mesh, template_lm_index, target_lm_index, config, device=torch.device("cuda:0")):
    target_pcl = convert_mesh_to_pcl(target_mesh)
    pcl_normal = target_mesh.verts_normals_padded()
    return non_rigid_icp_mesh2pcl(template_mesh, target_pcl, template_lm_index, target_lm_index, config, pcl_normal, device)


def non_rigid_icp_mesh2pcl(
    template_mesh,
    target_pcl,
    template_lm_index,
    target_lm_index,
    config,
    pcl_normal=None,
    device=torch.device("cuda:0"),
    out_affine=False,
    in_affine=None,
):
    template_mesh = template_mesh.to(device)
    target_pcl = target_pcl.to(device)
    template_lm_index = template_lm_index.to(device)
    target_lm_index = target_lm_index.to(device)
    template_vertex = template_mesh.verts_padded()
    target_vertex = target_pcl.points_padded()
    assert target_vertex.shape[0] == 1

    boundary_mask = mesh_boundary(template_mesh.faces_padded()[0], template_vertex.shape[1]).unsqueeze(0).unsqueeze(2)
    inner_mask = torch.logical_not(boundary_mask)
    target_lm = batch_vertex_sample(target_lm_index, target_vertex)
    template_lm = batch_vertex_sample(template_lm_index, template_vertex)
    r, t, s = corresponding_points_alignment(template_lm, target_lm, estimate_scale=True)
    transformed_vertex = s[:, None, None] * torch.bmm(template_vertex, r) + t[:, None, :]

    template_edges = template_mesh.edges_packed()
    local_affine_model = LocalAffine(template_vertex.shape[1], template_vertex.shape[0], template_edges).to(device) if in_affine is None else in_affine
    optimizer = torch.optim.AdamW([{"params": local_affine_model.parameters()}], lr=1e-4, amsgrad=True)

    inner_iter = config["inner_iter"]
    outer_iter = config["outer_iter"]
    log_iter = config["log_iter"]
    milestones = set(config["milestones"])
    stiffness_weights = np.array(config["stiffness_weights"])
    landmark_weights = np.array(config["landmark_weights"])
    laplacian_weight = config["laplacian_weight"]
    w_idx = 0

    for i in tqdm(range(outer_iter)):
        new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num=0, return_stiff=True)
        new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
        old_verts = new_deformed_verts
        new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
        target_sample_verts = target_vertex
        knn = knn_points(new_deformed_verts, target_sample_verts)
        close_points = knn_gather(target_sample_verts, knn.idx)[:, :, 0]
        inner_loop = range(100) if (i == 0 and in_affine is None) else range(inner_iter)
        for _ in inner_loop:
            optimizer.zero_grad()
            vert_distance = (new_deformed_verts - close_points) ** 2
            vert_distance_mask = torch.sum(vert_distance, dim=2) < 0.04**2
            weight_mask = torch.logical_and(inner_mask, vert_distance_mask.unsqueeze(2))
            vert_distance = weight_mask * vert_distance
            landmark_distance = (new_deformed_lm - target_lm) ** 2
            bsize = vert_distance.shape[0]
            vert_sum = torch.sum(vert_distance.view(bsize, -1)) / bsize
            landmark_sum = torch.sum(landmark_distance.view(bsize, -1)) * landmark_weights[w_idx] / bsize
            stiffness_sum = torch.sum(stiffness.view(bsize, -1)) * stiffness_weights[w_idx] / bsize
            laplacian_loss = mesh_laplacian_smoothing(new_deform_mesh) * laplacian_weight
            loss = torch.sqrt(vert_sum + landmark_sum + stiffness_sum) + laplacian_loss
            loss.backward()
            optimizer.step()
            new_deformed_verts, stiffness = local_affine_model(transformed_vertex, pool_num=0, return_stiff=True)
            new_deformed_lm = batch_vertex_sample(template_lm_index, new_deformed_verts)
            new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
        distance = torch.mean(torch.sqrt(torch.sum((old_verts - new_deformed_verts) ** 2, dim=2)))
        if i % log_iter == 0:
            print(distance, stiffness_sum.item(), landmark_sum.item(), vert_sum.item(), laplacian_loss.item())
        if i in milestones:
            w_idx += 1

    new_deform_mesh = template_mesh.update_padded(new_deformed_verts)
    if out_affine:
        return new_deform_mesh, local_affine_model
    return new_deform_mesh
