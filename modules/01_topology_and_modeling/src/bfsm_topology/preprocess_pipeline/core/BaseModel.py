"""Camera, differential geometry and lighting for a row-vector face model.
"""
import math
import torch
from torch import nn


class BaseReconModel(nn.Module):
    def __init__(self, batch_size=1, focal=1015, img_size=224, device="cuda:0"):
        super().__init__()
        self.batch_size, self.focal, self.img_size = batch_size, focal, img_size
        self.device = torch.device(device)
        self.p_mat = self._get_p_mat(self.device)
        self.reverse_z = self._get_reverse_z(self.device)
        self.camera_pos = self._get_camera_pose(self.device)
        self.renderer = self._get_renderer(self.device)
        self.init_coeff_dims()
        self.init_coeff_tensors()

    def _get_p_mat(self, device):
        center = self.img_size // 2
        return torch.tensor([[[self.focal, 0, center], [0, self.focal, center], [0, 0, 1]]],
                            dtype=torch.float32, device=device)

    def _get_reverse_z(self, device):
        return torch.diag(torch.tensor([1., 1., -1.], dtype=torch.float32, device=device)).unsqueeze(0)

    def _get_camera_pose(self, device):
        return torch.tensor([[[0., 0., 10.]]], device=device)

    def _get_renderer(self, device):
        from pytorch3d import renderer as rendering
        from ..rasterization import MeshRasterizer

        rotation, position = rendering.look_at_view_transform(10, 0, 0)
        camera = rendering.FoVPerspectiveCameras(
            device=device, R=rotation, T=position, znear=.01, zfar=50,
            fov=math.degrees(2 * math.atan((self.img_size // 2) / self.focal)))
        lighting = rendering.PointLights(
            device=device, location=((0., 0., 100000.),),
            ambient_color=((1., 1., 1.),), diffuse_color=((0., 0., 0.),),
            specular_color=((0., 0., 0.),))
        raster = MeshRasterizer(cameras=camera, raster_settings=rendering.RasterizationSettings(
            image_size=self.img_size, blur_radius=0., faces_per_pixel=1))
        shader = rendering.SoftPhongShader(
            device=device, cameras=camera, lights=lighting,
            blend_params=rendering.BlendParams(background_color=[0, 0, 0]))
        return rendering.MeshRenderer(rasterizer=raster, shader=shader)

    def compute_norm(self, vs, tri, point_buf):
        """Area-weighted normals; adjacency padding references the appended zero face."""
        # Separate gathers retain the established gradient accumulation order.
        first, second, third = (vs[:, tri[:, corner], :] for corner in range(3))
        area_vectors = torch.linalg.cross(first - second, second - third, dim=-1)
        padded = torch.cat((area_vectors, area_vectors.new_zeros((len(vs), 1, 3))), dim=1)
        summed = padded[:, point_buf].sum(dim=2)
        return summed / torch.linalg.vector_norm(summed, dim=-1, keepdim=True)

    def project_vs(self, vs):
        camera_coordinates = vs @ self.reverse_z + self.camera_pos
        homogeneous = camera_coordinates @ self.p_mat.transpose(-1, -2)
        return homogeneous[..., :2] / homogeneous[..., 2:3]

    def compute_rotation_matrix(self, angles):
        """Return (Rz Ry Rx)^T for points represented as row vectors."""
        sine = angles.sin().to(torch.get_default_dtype())
        cosine = angles.cos().to(torch.get_default_dtype())
        zero, one = torch.zeros_like(sine[:, 0]), torch.ones_like(sine[:, 0])
        sx, sy, sz = sine.unbind(dim=1)
        cx, cy, cz = cosine.unbind(dim=1)
        def matrix(entries):
            return torch.stack(entries, dim=1).reshape(-1, 3, 3)
        rx = matrix((one, zero, zero, zero, cx, -sx, zero, sx, cx))
        ry = matrix((cy, zero, sy, zero, one, zero, -sy, zero, cy))
        rz = matrix((cz, -sz, zero, sz, cz, zero, zero, zero, one))
        return (rz @ ry @ rx).transpose(1, 2)

    def add_illumination(self, face_texture, norm, gamma):
        """RGB second-order spherical harmonics, with 0.8 ambient offset."""
        x, y, z = norm.unbind(dim=-1)
        root = math.sqrt
        a, b, c = math.pi, 2 * math.pi / root(3), 2 * math.pi / root(8)
        u, v = 1 / root(4 * math.pi), root(3) / root(4 * math.pi)
        w = 3 * root(5) / root(12 * math.pi)
        harmonics = torch.stack((
            torch.ones_like(x, dtype=torch.float32) * a * u,
            -b * v * y, b * v * z, -b * v * x,
            c * w * x * y, -c * w * y * z,
            c * w * (.5 / root(3)) * (3 * z.square() - 1),
            -c * w * x * z, c * w * .5 * (x.square() - y.square())), dim=-1)
        channels = gamma.reshape(-1, 3, 9)
        ambient = channels.new_zeros((1, 1, 9))
        ambient[..., 0] = .8
        return face_texture * (harmonics @ (channels + ambient).transpose(1, 2))

    @staticmethod
    def rigid_transform(vs, rot, trans):
        return vs @ rot + trans.reshape(-1, 1, 3)

    def init_coeff_tensors(self, id_coeff=None, tex_coeff=None):
        """Create leaf tensors; identity and texture are shared by the batch."""
        dimensions = {"id": self.id_dims, "tex": self.tex_dims, "exp": self.exp_dims,
                      "rot": 3, "trans": 3, "gamma": 27}
        seeds = {"id": id_coeff, "tex": tex_coeff}
        for name, width in dimensions.items():
            initial = seeds.get(name)
            options = dict(dtype=torch.float32, device=self.device, requires_grad=True)
            if initial is None:
                count = 1 if name in seeds else self.batch_size
                value = torch.zeros((count, width), **options)
            else:
                value = torch.tensor(initial, **options)
            setattr(self, name + "_tensor", value)

    def get_packed_tensors(self):
        common = [getattr(self, name + "_tensor").repeat(self.batch_size, 1) for name in ("id", "tex")]
        return self.merge_coeffs(common[0], self.exp_tensor, common[1], self.rot_tensor,
                                 self.gamma_tensor, self.trans_tensor)

    def get_rot_tensor(self):
        return self.rot_tensor

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_tex_tensor(self):
        return self.tex_tensor

    def get_id_tensor(self):
        return self.id_tensor

    def get_gamma_tensor(self):
        return self.gamma_tensor
