"""Linear BFM09 decoder with the fitting pipeline's 257-value coefficient layout."""
import torch
from .BaseModel import BaseReconModel


class BFM09ReconModel(BaseReconModel):
    def __init__(self, model_dict, **kwargs):
        super().__init__(**kwargs)
        fields = {"meanshape": "meanshape", "idBase": "idBase", "expBase": "exBase",
                  "meantex": "meantex", "texBase": "texBase"}
        for attribute, key in fields.items():
            setattr(self, attribute, torch.tensor(model_dict[key], dtype=torch.float32, device=self.device))
        self.skinmask = torch.tensor(model_dict["skinmask"], device=self.device)
        for attribute, key in (("tri", "tri"), ("point_buf", "point_buf"), ("kp_inds", "keypoints")):
            indices = torch.tensor(model_dict[key] - 1, dtype=torch.int64, device=self.device)
            setattr(self, attribute, indices.squeeze() if attribute == "kp_inds" else indices)

    def init_coeff_dims(self):
        self.id_dims, self.exp_dims, self.tex_dims = 80, 64, 80

    def split_coeffs(self, coeffs):
        widths = (self.id_dims, self.exp_dims, self.tex_dims, 3, 27, coeffs.shape[1] - 254)
        return torch.split(coeffs, widths, dim=1)

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation):
        return torch.cat((id_coeff, exp_coeff, tex_coeff, angles, gamma, translation), dim=1)

    def get_vs(self, id_coeff, exp_coeff):
        deformation = id_coeff @ self.idBase.T + exp_coeff @ self.expBase.T
        vertices = (deformation + self.meanshape).reshape(len(id_coeff), -1, 3)
        origin = self.meanshape.reshape(1, -1, 3).mean(dim=1, keepdim=True)
        return vertices - origin

    def get_color(self, tex_coeff):
        return (tex_coeff @ self.texBase.T + self.meantex).reshape(len(tex_coeff), -1, 3)

    def get_lms(self, vs):
        return vs[:, self.kp_inds]

    def get_skinmask(self):
        return self.skinmask

    def forward(self, coeffs, render=True):
        identity, expression, texture, pose, light, shift = self.split_coeffs(coeffs)
        local = self.get_vs(identity, expression)
        rotation = self.compute_rotation_matrix(pose)
        posed = self.rigid_transform(local, rotation, shift)
        landmarks = self.project_vs(self.get_lms(posed))
        image_landmarks = torch.stack((landmarks[..., 0], self.img_size - landmarks[..., 1]), dim=-1)
        result = {"lms_proj": image_landmarks}
        if render:
            from pytorch3d.renderer import TexturesVertex
            from pytorch3d.structures import Meshes

            albedo = self.get_color(texture)
            normals = self.compute_norm(local, self.tri, self.point_buf) @ rotation
            colors = self.add_illumination(albedo, normals, light)
            mesh = Meshes(verts=posed, faces=self.tri.unsqueeze(0).repeat(len(coeffs), 1, 1),
                          textures=TexturesVertex(verts_features=colors))
            result.update(mesh=mesh, rendered_img=self.renderer(mesh).clamp(0, 255),
                          face_texture=albedo, vs=posed, tri=self.tri, color=colors)
        return result
