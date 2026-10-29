import torch
from pytorch3d.structures import Pointclouds


def normalize_mesh(in_mesh):
    assert in_mesh.verts_padded().shape[0] == 1
    aabb = in_mesh.get_bounding_boxes()
    distance = aabb[:, :, 1] - aabb[:, :, 0]
    max_distance, _ = torch.max(distance, dim=1)
    mesh_verts = in_mesh.verts_padded()
    center = torch.mean(mesh_verts, dim=1, keepdim=True).repeat(1, mesh_verts.shape[1], 1)
    offset = -1 * center.reshape(-1, 3)
    scale = 1 / max_distance
    out_mesh = in_mesh.offset_verts(offset)
    out_mesh.scale_verts_(scale)
    return out_mesh, (offset, scale)


def mesh_boundary(in_faces, num_verts: int):
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
    in_xy_hash = torch.minimum(in_x * num_verts + in_y, in_y * num_verts + in_x)
    in_xz_hash = torch.minimum(in_x * num_verts + in_z, in_z * num_verts + in_x)
    in_yz_hash = torch.minimum(in_y * num_verts + in_z, in_z * num_verts + in_y)
    output, count = torch.unique(torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim=0), return_counts=True, dim=0)
    boundary_edge = output[count == 1]
    boundary_mask = torch.zeros(num_verts).bool().to(in_faces.device)
    boundary_mask[boundary_edge // num_verts] = True
    boundary_mask[boundary_edge % num_verts] = True
    return boundary_mask


def convert_mesh_to_pcl(in_mesh):
    return Pointclouds(in_mesh.verts_padded())


def batch_vertex_sample(batch_idx, vertex):
    batch_idx_expand = batch_idx.unsqueeze(2).expand(batch_idx.shape[0], batch_idx.shape[1], vertex.shape[2])
    return torch.gather(vertex, 1, batch_idx_expand)
