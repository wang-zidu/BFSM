from pathlib import Path
import numpy as np
import torch
import cv2
import torch.nn.functional as F
# from bfsm_core.rendering.nv_diffrast import MeshRenderer, MeshRenderer_UV
from . import networks

from bfsm_core.parametric import ParametricFaceSkullModel


def get_colors_from_uv(colors, uv_coords):
    res = bilinear_interpolate_numpy(colors, uv_coords[:, 0], uv_coords[:, 1])
    return res

def bilinear_interpolate_numpy(img, x, y):

    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def bilinear_interpolate(img, x, y):
    
    x = torch.clamp(x, 0, img.shape[1] - 1)
    y = torch.clamp(y, 0, img.shape[0] - 1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    x0 = torch.clamp(x0, 0, img.shape[1] - 1)
    x1 = torch.clamp(x1, 0, img.shape[1] - 1)
    y0 = torch.clamp(y0, 0, img.shape[0] - 1)
    y1 = torch.clamp(y1, 0, img.shape[0] - 1)

    i_a = img[y0, x0]
    i_b = img[y1, x0]
    i_c = img[y0, x1]
    i_d = img[y1, x1]

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    return wa.unsqueeze(-1) * i_a + wb.unsqueeze(-1) * i_b + wc.unsqueeze(-1) * i_c + wd.unsqueeze(-1) * i_d


def optimize_face_model(face_skull_model, gt_f_v, face_to_skull_ind_new_torch, steps_config, legacy_cache=False):

    """
    optimizing face-skull model coefficients.

    Args:
        face_skull_model: An instance of ParametricFaceSkullModel.
        gt_f_v (torch.Tensor): Target vertices with shape [1, N, 3].
        face_to_skull_ind_new_torch (torch.Tensor): Face-vertex indices used for the geometry-fitting loss.
        steps_config (list[dict]): A list of configuration dictionaries, each containing:
            - name (str): Name of the optimization stage.
            - lr (float): Learning rate.
            - steps (int): Number of optimization iterations.
            - params (list[str]): List of parameter names (keys from coeffs_dict) to optimize.
    """

    # Initialize learnable coefficients
    device = gt_f_v.device
    batch_size = gt_f_v.shape[0]
    coeffs_dict = {
        'id': torch.zeros((batch_size, face_skull_model.n_components), requires_grad=True, device=device),
        'tex': torch.zeros((batch_size, face_skull_model.n_components), requires_grad=True, device=device),
        'angle': torch.zeros((batch_size, 3), requires_grad=True, device=device),
        'gamma': torch.zeros((batch_size, 27), requires_grad=True, device=device),
        'scale': torch.ones((batch_size, 1), requires_grad=True, device=device),
        'trans': torch.zeros((batch_size, 3), requires_grad=True, device=device),
    }

    for step in steps_config:
        print(f"=== Starting {step['name']} ===")
        optimizer = torch.optim.Adam(
            [coeffs_dict[param] for param in step['params']],
            lr=step['lr']
        )
        for iii in range(step['steps']):
            f_v_ori, f_c, s_v = face_skull_model.compute_for_all_shape(coeffs_dict)
            f_v = f_v_ori[:, face_to_skull_ind_new_torch, :]

            loss_v = torch.mean((f_v - gt_f_v) ** 2)
            loss_c = 0 * loss_v  # can be extended to include color or other losses
            loss = loss_v + loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iii + 1) % 100 == 0:
                print(f"{step['name']} Iteration {iii + 1}, Loss: {loss.item():.6f}, "
                      f"loss_v: {loss_v.item():.6f}, loss_c: {loss_c.item():.6f}")

    if not legacy_cache:
        with torch.no_grad():
            _, _, s_v = face_skull_model.compute_for_all_shape(coeffs_dict)
    return coeffs_dict, s_v


class face_model:
    def __init__(self, args):

        self.args = args

        self.device = self.args.device
        selected_device = torch.device(self.device)
        if selected_device.type == "cuda" and selected_device.index is not None:
            torch.cuda.set_device(selected_device)
        self.assets = Path(getattr(args, "assets", "assets"))
        model = np.load(self.assets / "face_model.npy",allow_pickle=True).item()

        # mean shape, size (107127, 1)
        self.u = torch.tensor(model['u'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face identity bases, size (107127, 80)
        self.id = torch.tensor(model['id'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face expression bases, size (107127, 64)
        self.exp = torch.tensor(model['exp'], requires_grad=False, dtype=torch.float32, device=self.device)
        # mean albedo, size (107127, 1)
        self.u_alb = torch.tensor(model['u_alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face albedo bases, size (107127, 80)
        self.alb = torch.tensor(model['alb'], requires_grad=False, dtype=torch.float32, device=self.device)
        # for computing vertex normals, size (35709, 8), see https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/132
        self.point_buf = torch.tensor(model['point_buf'], requires_grad=False, dtype=torch.int64, device=self.device)
        # triangle faces, size (70789, 3)
        self.tri = torch.tensor(model['tri'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex uv coordinates, size (35709, 3), range (0, 1.)
        self.uv_coords = torch.tensor(model['uv_coords'], requires_grad=False, dtype=torch.float32, device=self.device)
        
        if args.extractTex:
            uv_coords_numpy = process_uv(model['uv_coords'].copy(), 1024, 1024)
            self.uv_coords_torch = (torch.tensor(uv_coords_numpy, requires_grad=False, dtype=torch.float32, device=self.device) / 1023 - 0.5) * 2
            if self.device == 'cpu':
                from bfsm_core.rendering.cpu_renderer import MeshRenderer_UV_cpu
                self.uv_renderer = MeshRenderer_UV_cpu(
                        rasterize_size=int(1024.)
                )
                self.uv_coords_torch = self.uv_coords_torch + 1e-6 # For CPU renderer, a slight perturbation may be needed to avoid certain rendering artifacts. Users can comment out 1e-6 to compare different texture effects.
            else:
                from bfsm_core.rendering.nv_diffrast import MeshRenderer_UV
                self.uv_renderer = MeshRenderer_UV(
                        rasterize_size=int(1024.)
                )
            self.uv_coords_numpy = uv_coords_numpy.copy()
            self.uv_coords_numpy[:,1] = 1024 - self.uv_coords_numpy[:,1] - 1

        # vertex indices for 68 landmarks, size (68,)
        if self.args.ldm68:
            self.ldm68 = torch.tensor(model['ldm68'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 106 landmarks, size (106,)
        if self.args.ldm106 or self.args.ldm106_2d:
            self.ldm106 = torch.tensor(model['ldm106'], requires_grad=False, dtype=torch.int64, device=self.device)
        # vertex indices for 134 landmarks, size (134,)
        if self.args.ldm134:
            self.ldm134 = torch.tensor(model['ldm134'], requires_grad=False, dtype=torch.int64, device=self.device)

        # segmentation annotation indices for 8 parts, [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
        if self.args.seg_visible:
            self.annotation = model['annotation']

        # segmentation triangle faces for 8 parts
        if self.args.seg:
            self.annotation_tri = [torch.tensor(i, requires_grad=False, dtype=torch.int64, device=self.device) for i in model['annotation_tri']]

        # face profile parallel, list
        if self.args.ldm106_2d:
            self.parallel = model['parallel']
            # parallel for profile matching
            self.v_parallel = - torch.ones(35709, device=self.device).type(torch.int64)
            for i in range(len(self.parallel)):
                self.v_parallel[self.parallel[i]]=i

        # focal = 1015, center = 112
        self.persc_proj = torch.tensor([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], requires_grad=False, dtype=torch.float32, device=self.device).reshape([3, 3]).transpose(0,1)
        self.camera_distance = 10.0
        self.init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0], requires_grad=False, dtype=torch.float32, device=self.device).reshape([1, 1, -1])
        self.SH_a = torch.tensor([np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)], requires_grad=False, dtype=torch.float32, device=self.device)
        self.SH_c = torch.tensor([1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)], requires_grad=False, dtype=torch.float32, device=self.device)
        if self.device == 'cpu':
            from bfsm_core.rendering.cpu_renderer import MeshRenderer_cpu
            self.renderer = MeshRenderer_cpu(
                        rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(1024.)
            )
        else:
            from bfsm_core.rendering.nv_diffrast import MeshRenderer
            self.renderer = MeshRenderer(
                        rasterize_fov=2 * np.arctan(112. / 1015) * 180 / np.pi, znear=5., zfar=15., rasterize_size=int(1024.)
            )

        if args.backbone == 'deep3dcloseeye':
            self.net_recon = networks.define_net_recon(
                net_recon='resnet50', use_last_fc=False, init_path=None
            )
            self.net_recon.load_state_dict(torch.load(self.assets / "epoch_20_deep3d_closeeye.pth", map_location=torch.device('cpu'), weights_only=False)['net_recon'])
            self.net_recon = self.net_recon.to(self.device)
            self.net_recon.eval()
            print('use epoch_20_deep3d_closeeye as backbone')

        self.input_img = None

        self.steps_config = [
            {'name': 'step1_rigid', 'lr': 0.01, 'steps': 500, 'params': ['angle', 'scale', 'trans', 'gamma', 'tex']},
            {'name': 'step2_rigid', 'lr': 0.005, 'steps': 500, 'params': ['angle', 'scale', 'trans', 'gamma', 'tex']},
            {'name': 'step3_non-rigid', 'lr': 0.01, 'steps': 500, 'params': ['angle', 'scale', 'trans', 'gamma', 'tex', 'id']},
            {'name': 'step4_non-rigid', 'lr': 0.005, 'steps': 1500, 'params': ['angle', 'scale', 'trans', 'gamma', 'tex', 'id']},
            {'name': 'step5_non-rigid', 'lr': 0.0001, 'steps': 2000, 'params': ['angle', 'scale', 'trans', 'gamma', 'tex', 'id']},
        ]
        self.face_skull_model = ParametricFaceSkullModel(model_path=getattr(args, "fsmm", None) or self.assets / "BFSM/FaceSkullMorphableModel.npy")
        self.face_skull_model.to(self.device)

        self.face_to_skull_ind_new_torch = torch.arange(35709, dtype=torch.int64).to(self.device)
        self.skull_faces = self.face_skull_model.skull_tri.to(dtype=torch.int64)
        self.skull_faces = [self.skull_faces]

        if getattr(args, 'skull_seg', False):
            from bfsm_models.regions import load_regions, REGIONS
            import bfsm_models.regions as regions
            masks = load_regions(Path(regions.__file__).with_name('skull_regions.npz'),
                                 self.skull_faces[0].detach().cpu().numpy(),
                                 self.face_skull_model.mean_skull.shape[0] // 3)
            self.skull_region_masks = [masks[name] for name in REGIONS]


    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3), face vertices without rotation or translation

        Parameters:
            alpha_id         -- torch.tensor, size (B, 80), identity parameter
            alpha_exp        -- torch.tensor, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_albedo(self, alpha_alb, normalize=True):
        """
        Return:
            face_albedo     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.), without lighting

        Parameters:
            alpha_alb        -- torch.tensor, size (B, 80), albedo parameter
        """
        batch_size = alpha_alb.shape[0]
        face_albedo = torch.einsum('ij,aj->ai', self.alb, alpha_alb) + self.u_alb.reshape([1, -1])
        if normalize:
            face_albedo = face_albedo / 255.
        return face_albedo.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_texture(self, face_albedo, face_norm, alpha_sh):
        """
        Return:
            face_texture        -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_albedo         -- torch.tensor, size (B, N, 3), from albedo model, range (0, 1.)
            face_norm           -- torch.tensor, size (B, N, 3), rotated face normal
            alpha_sh            -- torch.tensor, size (B, 27), SH parameter
        """
        batch_size = alpha_sh.shape[0]
        v_num = face_albedo.shape[1]
        a = self.SH_a
        c = self.SH_c
        alpha_sh = alpha_sh.reshape([batch_size, 3, 9])
        alpha_sh = alpha_sh + self.init_lit
        alpha_sh = alpha_sh.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ alpha_sh[..., :1]
        g = Y @ alpha_sh[..., 1:2]
        b = Y @ alpha_sh[..., 2:]
        face_texture = torch.cat([r, g, b], dim=-1) * face_albedo
        return face_texture

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of torch.tensors

        Parameters:
            alpha          -- torch.tensor, size (B, 256)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            'id': alpha_id,
            'exp': alpha_exp,
            'alb': alpha_alb,
            'angle': alpha_a,
            'sh': alpha_sh,
            'trans': alpha_t
        }

    def get_landmarks_68(self, v2d):
        """
        Return:
            landmarks_68_2d         -- torch.tensor, size (B, 68, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm68]

    def get_landmarks_106(self, v2d):
        """
        Return:
            landmarks_106_2d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm106]

    def get_landmarks_134(self, v2d):
        """
        Return:
            landmarks_134            -- torch.tensor, size (B, 134, 2)

        Parameters:
            v2d                      -- torch.tensor, size (B, N, 2)
        """
        return v2d[:, self.ldm134]

    def get_landmarks_106_2d(self, v2d, face_shape, alpha_dict):
        """
        Return:
            landmarks_106_2d         -- torch.tensor, size (B, 106, 2)

        Parameters:
            v2d                     -- torch.tensor, size (B, N, 2)
            face_shape              -- torch.tensor, size (B, N, 3), face vertices without rotation or translation
            alpha_dict              -- a dict of torch.tensors
        """

        temp_angle = alpha_dict['angle'].clone()
        temp_angle[:,2] = 0
        rotation_without_roll = self.compute_rotation(temp_angle)
        v2d_without_roll = self.to_image(self.to_camera(self.transform(face_shape, rotation_without_roll, alpha_dict['trans'])))

        visible_parallel = self.v_parallel.clone()
        # visible_parallel[visible_idx == 0] = -1

        ldm106_dynamic=self.ldm106.clone()
        for i in range(16):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = 1e5
            ldm106_dynamic[i]=torch.argmin(temp)

        for i in range(17,33):
            temp=v2d_without_roll.clone()[:,:,0]
            temp[:,visible_parallel!=i] = -1e5
            ldm106_dynamic[i]=torch.argmax(temp)

        return v2d[:, ldm106_dynamic]

    def add_directionlight(self, normals, lights = None):
        """Average Lambertian RGB irradiance from directional lights.

        Lights contain direction XYZ followed by RGB intensity. The surface
        cosine is clamped before intensity weighting and averaging.
        """
        if lights is None:
            directions = torch.tensor(
                ((-1, 1, 1), (1, 1, 1), (-1, -1, 1), (1, -1, 1), (0, 0, 1)),
                dtype=torch.float32, device=normals.device,
            ).unsqueeze(0).expand(normals.shape[0], -1, -1)
            intensities = torch.ones_like(directions)
        else:
            directions, intensities = lights[..., :3], lights[..., 3:]
        rays = directions.unsqueeze(2).expand(-1, -1, normals.shape[1], -1)
        unit_rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(1e-12)
        cosines = (normals[:, None] * unit_rays).sum(-1).clamp(0., 1.)
        return (cosines[..., None] * intensities[:, :, None]).mean(1)

    def compute_gray_shading_with_directionlight(self, face_texture, normals):
        """Apply the original five-light neutral preview at intensity 1.7."""
        directions = torch.tensor(
            ((-1, 1, 1), (1, 1, 1), (-1, -1, 1), (1, -1, 1), (0, 0, 1)),
            dtype=torch.float32, device=face_texture.device,
        ).unsqueeze(0).expand(normals.shape[0], -1, -1)
        lights = torch.cat((directions, torch.full_like(directions, 1.7)), dim=-1)
        return face_texture * self.add_directionlight(normals, lights)

    def segmentation(self, v3d):

        seg = torch.zeros(224, 224, 8).to(v3d.device)
        for i in range(8):
            mask, _, _, _ = self.renderer(v3d.clone(), self.annotation_tri[i])
            seg[:,:,i] = F.interpolate(mask.float().reshape(1, 1, *mask.shape[-2:]), size=(224, 224), mode="nearest").squeeze()
        return seg

    def segmentation_visible(self, v3d, visible_idx):

        seg = torch.zeros(224, 224, 8).to(v3d.device)
        for i in range(8):
            temp = torch.zeros_like(v3d)
            temp[:,self.annotation[i],:] = 1
            temp[:,visible_idx == 0,:] = 0
            _, _, temp_image, _ = self.renderer(v3d.clone(), self.tri, temp.clone())
            temp_image = temp_image.mean(axis=1)
            mask = torch.where(temp_image >= 0.5, torch.tensor(1.0).to(v3d.device), torch.tensor(0.0).to(v3d.device))
            seg[:,:,i] = F.interpolate(mask.float().reshape(1, 1, *mask.shape[-2:]), size=(224, 224), mode="nearest").squeeze()
        return seg

    def segmentation_skull(self, skull_vertex):
        # Rasterize all regions against the complete skull depth buffer.
        # Mixed-boundary pixels take the largest interpolated membership.
        memberships = []
        for region in self.skull_region_masks:
            features = torch.as_tensor(region, device=skull_vertex.device,
                                       dtype=skull_vertex.dtype)[None, :, None].expand(-1, -1, 3)
            _, _, rendered, _ = self.renderer(skull_vertex.clone(), self.skull_faces[0], features.contiguous())
            memberships.append(F.interpolate(rendered[:, :1], size=(224, 224),
                                              mode='nearest')[0, 0])
        membership = torch.stack(memberships)
        labels = membership.argmax(0) + 1
        labels[membership.max(0).values <= 0] = 0
        return labels.detach().cpu().numpy().astype(np.uint8)


    def forward(self, no_pca = False):
        assert self.net_recon.training == False
        with torch.no_grad():
            alpha = self.net_recon(self.input_img)

        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict['id'], alpha_dict['exp'])
        rotation = self.compute_rotation(alpha_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, alpha_dict['trans'])

        # Face vertices in 3D camera coordinates.
        v3d = self.to_camera(face_shape_transformed)

        # Face vertices projected onto the 2D image plane.
        v2d = self.to_image(v3d)

        # compute face texture with albedo and lighting
        face_albedo = self.compute_albedo(alpha_dict['alb'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_texture = self.compute_texture(face_albedo, face_norm_roted, alpha_dict['sh'])

        # render shape with texture
        _, _, pred_image, visible_idx_renderer = self.renderer(v3d.clone(), self.tri, torch.clamp(face_texture, 0, 1).clone(), visible_vertice = True)

        # render shape
        gray_shading = self.compute_gray_shading_with_directionlight(torch.ones_like(face_albedo)*0.78,face_norm_roted)
        mask, _, pred_image_shape, _ = self.renderer(v3d.clone(), self.tri, gray_shading.clone())


        gt_f_v = v3d.detach().clone()
        gt_f_v[..., -1] = 10 - gt_f_v[..., -1]
        coeffs_dict_result_skull, s_v = optimize_face_model(self.face_skull_model, gt_f_v, self.face_to_skull_ind_new_torch.to(gt_f_v.device), self.steps_config, legacy_cache=getattr(self.args, "legacy_cache", False))
        skull_norm_roted = self.face_skull_model.vertex_normals(s_v, self.skull_faces)
        skull_vertex = self.to_camera(s_v)
        skull_color = self.add_directionlight(skull_norm_roted)
        mask_skull, _, pred_image_skull_shape, _ = self.renderer(skull_vertex.clone(), self.skull_faces[0], skull_color.clone())



        result_dict = {
            'v3d': v3d.detach().cpu().numpy(),
            'fsmm_coefficients': {k: v.detach().cpu().numpy() for k, v in coeffs_dict_result_skull.items()},
            'fsmm_face': self.to_camera(self.face_skull_model.compute_for_all_shape(coeffs_dict_result_skull)[0]).detach().cpu().numpy(),
            'v2d': v2d.detach().cpu().numpy(),
            'face_texture': np.clip(face_texture.detach().cpu().numpy(), 0, 1),
            'tri': self.tri.detach().cpu().numpy(),
            'uv_coords': self.uv_coords.detach().cpu().numpy(),
            'render_shape': pred_image_shape.detach().cpu().permute(0, 2, 3, 1).numpy(),
            'render_face': pred_image.detach().cpu().permute(0, 2, 3, 1).numpy(),
            'render_mask': mask.detach().cpu().permute(0, 2, 3, 1).numpy(),

            'v3d_skull': skull_vertex.detach().cpu().numpy(),
            'skull_colors': skull_color.detach().cpu().numpy(),
            'render_skull_shape': pred_image_skull_shape.detach().cpu().permute(0, 2, 3, 1).numpy(),
            'render_skull_mask': mask_skull.detach().cpu().permute(0, 2, 3, 1).numpy(),
            # 'coeffs_dict_result_skull': coeffs_dict_result_skull.detach().cpu().numpy(),
            'skull_tri': self.skull_faces[0].detach().cpu().numpy(),
        }

        # Determine visible vertices from normals and the renderer.
        if self.args.seg_visible or self.args.extractTex:
            visible_idx = torch.zeros(35709).type(torch.int64).to(v3d.device)
            visible_idx[visible_idx_renderer.type(torch.int64)] = 1
            visible_idx[(face_norm_roted[..., 2] < 0)[0]] = 0
            # result_dict['visible_idx'] = visible_idx

        # landmarks 68 3d
        if self.args.ldm68:
            v2d_68 = self.get_landmarks_68(v2d)
            result_dict['ldm68'] = v2d_68.detach().cpu().numpy()

        # landmarks 106 3d
        if self.args.ldm106:
            v2d_106 = self.get_landmarks_106(v2d)
            result_dict['ldm106'] = v2d_106.detach().cpu().numpy()

        # landmarks 106 2d
        if self.args.ldm106_2d:
            # v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict, visible_idx)
            v2d_106_2d = self.get_landmarks_106_2d(v2d, face_shape, alpha_dict)
            result_dict['ldm106_2d'] = v2d_106_2d.detach().cpu().numpy()

        # landmarks 134
        if self.args.ldm134:
            v2d_134 = self.get_landmarks_134(v2d)
            result_dict['ldm134'] = v2d_134.detach().cpu().numpy()

        # segmentation in 2d without visible mask
        if self.args.seg:
            seg = self.segmentation(v3d)
            result_dict['seg'] = seg.detach().cpu().numpy()

        # segmentation in 2d with visible mask
        if self.args.seg_visible:
            seg_visible = self.segmentation_visible(v3d, visible_idx)
            result_dict['seg_visible'] = seg_visible.detach().cpu().numpy()

        if getattr(self.args, 'skull_seg', False):
            result_dict['skull_seg'] = self.segmentation_skull(skull_vertex)

        # Blend image colors with BFM colors in UV space using the visibility mask.
        if self.args.extractTex:
            _, _, uv_color_pca, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, (torch.clamp(face_texture, 0, 1)).clone())
            img_colors = bilinear_interpolate(self.input_img.permute(0, 2, 3, 1).detach()[0], v2d[0, :, 0].detach(), 223 - v2d[0, :, 1].detach())
            _, _, uv_color_img, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, img_colors.unsqueeze(0).clone())
            _, _, uv_weight, _ = self.uv_renderer(self.uv_coords_torch.unsqueeze(0).clone(), self.tri, (1 - torch.stack((visible_idx,)*3, axis=-1).unsqueeze(0).type(torch.float32).to(self.tri.device)).clone())

            # median_filtered_w = cv2.medianBlur((uv_weight.detach().cpu().permute(0, 2, 3, 1).numpy()[0]*255).astype(np.uint8), 31)/255.

            uv_color_pca = uv_color_pca.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            uv_color_img = uv_color_img.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

            # res_colors = ((1 - median_filtered_w) * np.clip(uv_color_img, 0, 1) + median_filtered_w * np.clip(uv_color_pca, 0, 1))
            # result_dict['extractTex_uv'] = res_colors
            
            if no_pca:
                # Blend with horizontally flipped image colors.
                res_colors = cv2.seamlessClone((uv_color_img*255).astype(np.uint8), cv2.flip((uv_color_img*255).astype(np.uint8), 1), ((1 - uv_weight.detach().cpu().permute(0, 2, 3, 1).numpy()[0])*255)[:,:,0].astype(np.uint8), (512, 512), cv2.NORMAL_CLONE) / 255.
            else:
                # use pca-texture for texture blending
                res_colors = cv2.seamlessClone((uv_color_img*255).astype(np.uint8), (uv_color_pca*255).astype(np.uint8), ((1 - uv_weight.detach().cpu().permute(0, 2, 3, 1).numpy()[0])*255)[:,:,0].astype(np.uint8), (512, 512), cv2.NORMAL_CLONE) / 255.
        
            v_colors = get_colors_from_uv(res_colors.copy(), self.uv_coords_numpy.copy())
            result_dict['extractTex'] = v_colors

        return result_dict
