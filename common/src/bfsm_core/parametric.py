from pathlib import Path
from .paths import ROOT
from .model import validate_model_arrays
import numpy as np
import torch
import torch.nn.functional as F


def resolve_model_path(path):
    result = Path(path)
    root = ROOT
    if not result.is_file() and result.resolve().parent == (root / 'models').resolve():
        result = root / 'assets' / 'models' / result.name
    return result


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class ParametricFaceSkullModel:
    def __init__(self, 
                recenter=True,
                model_path="./assets/BFSM/FaceSkullMorphableModel.npy",
                load_texture=False,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                ):

        model_path = resolve_model_path(model_path)
        loaded = np.load(model_path, allow_pickle=model_path.suffix == '.npy')
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                FaceSkullMorphableModel = {key: loaded[key] for key in loaded.files}
        else:
            FaceSkullMorphableModel = loaded.item()

        face_count, _ = validate_model_arrays(FaceSkullMorphableModel)
        if face_count != 35709:
            raise ValueError('The image decoder requires the 35709-vertex BFSM face')

        self.mean_shape = FaceSkullMorphableModel['shapeMU'].astype(np.float32)#[:,:107127]

        self.id_base = FaceSkullMorphableModel['shapeBase'].astype(np.float32)[:107127,:]

        self.skull_base = FaceSkullMorphableModel['shapeBase'].astype(np.float32)[107127:,:]

        self.n_components = self.id_base.shape[1]
        self.mean_shape = self.mean_shape.reshape(-1, 1)
        if load_texture:
            self.mean_tex = FaceSkullMorphableModel['texMU'].astype(np.float32)
            self.tex_base = FaceSkullMorphableModel['texBase'].astype(np.float32)


        self.face_buf = FaceSkullMorphableModel['face_buf']
        self.point_buf = FaceSkullMorphableModel['point_buf']
        self.skull_tri = FaceSkullMorphableModel['skull_tri']
        
        if recenter:
            print('recenter')
            mean_shape = self.mean_shape.reshape([-1, 3])
            center = FaceSkullMorphableModel.get(
                'recenter_offset', np.mean(mean_shape, axis=0, keepdims=True))
            mean_shape = mean_shape - np.asarray(center, dtype=mean_shape.dtype).reshape(1, 3)
            if not np.isfinite(mean_shape).all():
                raise ValueError('Recentered model geometry must remain finite in float32')
            self.mean_shape = mean_shape.reshape([-1, 1])
        
        self.mean_skull = self.mean_shape[107127:,:]
        self.mean_shape = self.mean_shape[:107127,:]

        self.device = 'cpu'
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))



    def compute_shape_scale(self, id_coeff, scale_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        # exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        # face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        face_shape = scale_coeff*(id_part + self.mean_shape.reshape([1, -1]))
        return face_shape.reshape([batch_size, -1, 3])


    def compute_skull_scale(self, id_coeff, scale_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.skull_base, id_coeff)
        # exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        # face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        skull_shape = scale_coeff*(id_part + self.mean_skull.reshape([1, -1]))
        return skull_shape.reshape([batch_size, -1, 3])

    # def compute_texture(self, tex_coeff, normalize=True):
    def compute_texture(self, tex_coeff, normalize=False):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])


    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm




    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
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
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color
    
    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
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




    def compute_for_all_shape(self, coef_dict):

        # coef_dict = self.split_coeff(coeffs)
        # face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        face_shape = self.compute_shape_scale(coef_dict['id'],coef_dict['scale'])
        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])

        face_color = None
        if hasattr(self, 'tex_base'):
            face_texture = self.compute_texture(coef_dict['tex'])
            face_norm = self.compute_norm(face_shape)
            face_norm_roted = face_norm @ rotation
            face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        skull_shape = self.compute_skull_scale(coef_dict['id'],coef_dict['scale'])
        skull_shape_transformed = self.transform(skull_shape, rotation, coef_dict['trans'])
        # skull_vertex = self.to_camera(skull_shape_transformed)
        # skull_proj = self.to_image(skull_vertex)
        # skull_norm = self.compute_norm_skull(skull_shape)
        # skull_norm_roted = skull_norm @ rotation
        # skull_color = self.add_directionlight(skull_norm_roted)

        return face_shape_transformed, face_color, skull_shape_transformed

    def vertex_normals(self, vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: list of size [batch size], each containing a [nfi, 3] tensor for the face indices
        :return: [batch size, number of vertices, 3]
        """
        assert (vertices.ndimension() == 3)
        bs, nv = vertices.shape[:2]
        device = vertices.device

        # Initialize a tensor to store the normals
        normals = torch.zeros((bs * nv, 3)).to(device)

        # Flatten vertices to [batch size * number of vertices, 3]
        vertices_flat = vertices.reshape(bs * nv, 3)

        # Flatten faces list into a single tensor containing all the face indices
        faces_all = []
        for b in range(bs):
            faces_b = faces[b]  # [nfi, 3] for the b-th batch
            face_offset = b * nv  # Offset for the current batch
            faces_b = faces_b + face_offset  # Add offset to indices
            faces_all.append(faces_b)
    
        faces_all = torch.cat(faces_all, dim=0) # Concatenate all faces into one tensor of shape [f, 3]

        # Extract vertices corresponding to each face
        vertices_faces = vertices_flat[faces_all]  # Shape [f, 3, 3]

        # Compute the normals for each face and accumulate them in the corresponding vertices
        normals.index_add_(0, faces_all[:, 1],
                           torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
        normals.index_add_(0, faces_all[:, 2],
                           torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
        normals.index_add_(0, faces_all[:, 0],
                           torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

        # Normalize the normals
        normals = F.normalize(normals, eps=1e-6, dim=1)

        # Reshape normals back to [batch size, number of vertices, 3]
        normals = normals.reshape(bs, nv, 3)

        return normals