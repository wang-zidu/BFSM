"""Keep every mesh triangle available to PyTorch3D's coarse raster bins."""
from dataclasses import replace

from pytorch3d.renderer import MeshRasterizer as _MeshRasterizer


class MeshRasterizer(_MeshRasterizer):
    def forward(self, meshes_world, **kwargs):
        settings = kwargs.get("raster_settings", self.raster_settings)
        capacity = int(meshes_world.num_faces_per_mesh().max().item())
        kwargs["raster_settings"] = replace(
            settings, max_faces_per_bin=max(capacity, settings.max_faces_per_bin or 0)
        )
        return super().forward(meshes_world, **kwargs)
