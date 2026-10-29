import torch
import torch.nn.functional as F
# import kornia
# from kornia.geometry.camera import pixel2cam
import numpy as np
from typing import List
import nvdiffrast.torch as dr
# from scipy.io import loadmat
from torch import nn

def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224,
                use_opengl=False):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
                torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.use_opengl = use_opengl
        self.ctx = None
    
    def forward(self, vertex, tri, feat = None, visible_vertice = False):
        """
        This renderer supports one instance (B=1); process multiple meshes separately.

        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), shared triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        if (not isinstance(vertex, torch.Tensor) or vertex.ndim != 3
                or vertex.shape[0] != 1 or vertex.shape[1] < 1 or vertex.shape[2] not in (3, 4)
                or vertex.dtype != torch.float32 or not bool(torch.isfinite(vertex).all())):
            raise ValueError('Renderer vertices must be finite float32 coordinates with shape (1, N, 3 or 4)')
        if (not isinstance(self.rasterize_size, (int, np.integer)) or isinstance(self.rasterize_size, bool)
                or self.rasterize_size < 1):
            raise ValueError('Raster size must be a positive integer')
        if feat is not None and (not isinstance(feat, torch.Tensor) or feat.ndim != 3
                or feat.shape[:2] != vertex.shape[:2] or feat.shape[2] < 1
                or feat.dtype != torch.float32 or feat.device != vertex.device
                or not bool(torch.isfinite(feat).all())):
            raise ValueError('Renderer features must be finite float32 (1, N, C) values on the vertex device')
        if not isinstance(tri, torch.Tensor) or tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError('Renderer requires one shared (M, 3) triangle tensor')
        if tri.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Renderer triangle indices must have integer dtype')
        if tri.numel() and (bool((tri < 0).any()) or bool((tri >= vertex.shape[1]).any())):
            raise ValueError('Renderer triangle indices are outside the vertex array')
        if tri.device != vertex.device:
            raise ValueError('Renderer triangles must be on the vertex device')
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex @ ndc_proj.t()
        if not bool(torch.isfinite(vertex_ndc).all()):
            raise ValueError('Renderer projection produced non-finite coordinates')
        if self.ctx is None:
            if self.use_opengl:
                # self.ctx = dr.RasterizeGLContext(device=device)
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "opengl"
            else:
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "cuda"
            print("create %s ctx on device cuda:%d"%(ctx_str, device.index))
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)
        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        unique_visible_verts_idx = None
        if visible_vertice:
            # assert bacth_size = 1
            visible_faces = torch.unique(rast_out[:, :, :, 3]) - 1
            visible_faces = visible_faces[visible_faces!=-1]
            visible_verts_idx = tri[visible_faces.long()]
            unique_visible_verts_idx = torch.unique(visible_verts_idx)
        
        return mask, depth, image, unique_visible_verts_idx

class MeshRenderer_UV(nn.Module):
    def __init__(self,

                rasterize_size=224,
                use_opengl=False):
        super(MeshRenderer_UV, self).__init__()

        self.rasterize_size = rasterize_size
        self.use_opengl = use_opengl
        self.ctx = None
    
    def forward(self, vertex, tri, feat = None, visible_vertice = False):
        """
        This renderer supports one instance (B=1); process multiple meshes separately.

        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), shared triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        if (not isinstance(vertex, torch.Tensor) or vertex.ndim != 3
                or vertex.shape[0] != 1 or vertex.shape[1] < 1 or vertex.shape[2] not in (3, 4)
                or vertex.dtype != torch.float32 or not bool(torch.isfinite(vertex).all())):
            raise ValueError('Renderer vertices must be finite float32 coordinates with shape (1, N, 3 or 4)')
        if (not isinstance(self.rasterize_size, (int, np.integer)) or isinstance(self.rasterize_size, bool)
                or self.rasterize_size < 1):
            raise ValueError('Raster size must be a positive integer')
        if feat is not None and (not isinstance(feat, torch.Tensor) or feat.ndim != 3
                or feat.shape[:2] != vertex.shape[:2] or feat.shape[2] < 1
                or feat.dtype != torch.float32 or feat.device != vertex.device
                or not bool(torch.isfinite(feat).all())):
            raise ValueError('Renderer features must be finite float32 (1, N, C) values on the vertex device')
        if not isinstance(tri, torch.Tensor) or tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError('Renderer requires one shared (M, 3) triangle tensor')
        if tri.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Renderer triangle indices must have integer dtype')
        if tri.numel() and (bool((tri < 0).any()) or bool((tri >= vertex.shape[1]).any())):
            raise ValueError('Renderer triangle indices are outside the vertex array')
        if tri.device != vertex.device:
            raise ValueError('Renderer triangles must be on the vertex device')
        device = vertex.device
        rsize = int(self.rasterize_size)
        # ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 


        vertex_ndc = vertex # @ ndc_proj.t()
        if not bool(torch.isfinite(vertex_ndc).all()):
            raise ValueError('Renderer projection produced non-finite coordinates')
        if self.ctx is None:
            if self.use_opengl:
                # self.ctx = dr.RasterizeGLContext(device=device)
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "opengl"
            else:
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "cuda"
            print("create %s ctx on device cuda:%d"%(ctx_str, device.index))
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri) 
        depth = depth.permute(0, 3, 1, 2)
        mask =  (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        unique_visible_verts_idx = None
        if visible_vertice:
            # assert bacth_size = 1
            visible_faces = torch.unique(rast_out[:, :, :, 3]) - 1
            visible_faces = visible_faces[visible_faces!=-1]
            visible_verts_idx = tri[visible_faces.long()]
            unique_visible_verts_idx = torch.unique(visible_verts_idx)
        
        return mask, depth, image, unique_visible_verts_idx