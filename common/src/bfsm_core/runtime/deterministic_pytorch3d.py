"""Opt-in reproducible GPU rasterization for PyTorch3D 0.7.9.

Upstream fine rasterization receives sorted coarse candidates so equal-depth
ties select a stable face. Raster gradients first write independent
pixel contributions using upstream CUDA formulas, then use PyTorch's
deterministic index_add implementation.
"""
from pathlib import Path
import os

import torch

_extension = None
_originals = None


def _load():
    global _extension
    if _extension is not None:
        return _extension
    source = os.environ.get("VT_PYTORCH3D_SOURCE")
    if not source:
        raise RuntimeError("Set VT_PYTORCH3D_SOURCE to the pinned PyTorch3D source checkout")
    include = Path(source).resolve() / "pytorch3d" / "csrc"
    if not (include / "utils/geometry_utils.cuh").is_file():
        raise RuntimeError("VT_PYTORCH3D_SOURCE does not contain required upstream headers")
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        raise RuntimeError("Set TORCH_EXTENSIONS_DIR to a writable local cache directory")
    from torch.utils.cpp_extension import load
    here = Path(__file__).resolve().parent
    _extension = load(
        name="vt_deterministic_raster_v1",
        sources=[str(here / "binding.cpp"), str(here / "pixel_contributions.cu")],
        extra_include_paths=[str(include)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17",
                          "--static-global-template-stub=false",
                          "--device-entity-has-hidden-visibility=false"],
        verbose=os.environ.get("VT_EXTENSION_VERBOSE") == "1",
    )
    return _extension


def _raster_forward(face_verts, first_idx, num_faces, clipped_neighbors, image_size,
                    blur_radius, faces_per_pixel, bin_size, max_faces_per_bin,
                    perspective_correct, clip_barycentric_coords, cull_backfaces):
    from pytorch3d import _C
    args = (face_verts, first_idx, num_faces, clipped_neighbors, image_size,
            blur_radius, faces_per_pixel, bin_size, max_faces_per_bin,
            perspective_correct, clip_barycentric_coords, cull_backfaces)
    if (not torch.are_deterministic_algorithms_enabled() or not face_verts.is_cuda
            or bin_size <= 0 or face_verts.numel() == 0):
        return _originals[2](*args)
    bins = _C._rasterize_meshes_coarse(
        face_verts, first_idx, num_faces, image_size, blur_radius, bin_size,
        max_faces_per_bin)
    # Upstream coarse writes packed prefixes. Refuse to discard any candidate
    # if that contract changes. Removing the common all-padding tail is exact.
    valid = bins >= 0
    counts = valid.sum(dim=-1)
    positions = torch.arange(bins.shape[-1], device=bins.device)
    if not torch.equal(valid, positions < counts[..., None]):
        raise RuntimeError("PyTorch3D coarse candidates are not packed prefixes")
    keep = max(1, int(counts.max()))
    bins = bins[..., :keep]
    # Send padding after valid faces. Do not perturb depth, barycentrics or the
    # upstream fine formulas: only equal-depth candidate priority is canonical.
    sentinel = face_verts.shape[0]
    ordered = torch.where(bins < 0, sentinel, bins).sort(dim=-1).values
    ordered = torch.where(ordered == sentinel, -1, ordered).contiguous()
    return _C._rasterize_meshes_fine(
        face_verts, ordered, clipped_neighbors, image_size, blur_radius, bin_size,
        faces_per_pixel, perspective_correct, clip_barycentric_coords, cull_backfaces)


def _raster_backward(face_verts, pix_to_face, grad_z, grad_bary, grad_dist,
                     perspective_correct, clip_barycentric_coords):
    if not face_verts.is_cuda:
        return _originals[0](face_verts, pix_to_face, grad_z, grad_bary, grad_dist,
                             perspective_correct, clip_barycentric_coords)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Enable torch.use_deterministic_algorithms(True) before backward")
    shape = pix_to_face.shape
    grad_z = torch.zeros(shape, device=face_verts.device) if grad_z is None else grad_z
    grad_dist = torch.zeros(shape, device=face_verts.device) if grad_dist is None else grad_dist
    grad_bary = torch.zeros((*shape, 3), device=face_verts.device) if grad_bary is None else grad_bary
    contributions = _load().pixel_contributions(
        face_verts.contiguous(), pix_to_face.contiguous(), grad_z.contiguous(),
        grad_bary.contiguous(), grad_dist.contiguous(), perspective_correct,
        clip_barycentric_coords)
    ids = pix_to_face.reshape(-1)
    valid = ids >= 0
    result = torch.zeros((face_verts.shape[0], 9), device=face_verts.device, dtype=face_verts.dtype)
    result.index_add_(0, ids[valid], contributions[valid])
    return result.reshape_as(face_verts)


def _interp_backward(pix_to_face, bary, attrs, grad):
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Enable torch.use_deterministic_algorithms(True) before backward")
    ids = pix_to_face.reshape(-1)
    valid = ids >= 0
    ids_valid = ids[valid]
    b = bary.reshape(-1, 3)[valid]
    g = grad.reshape(ids.numel(), -1)[valid]
    grad_bary = torch.zeros_like(bary).reshape(-1, 3)
    grad_bary[valid] = (attrs[ids_valid] * g[:, None, :]).sum(-1)
    grad_attrs = torch.zeros_like(attrs)
    grad_attrs.index_add_(0, ids_valid, b[:, :, None] * g[:, None, :])
    return grad_bary.reshape_as(bary), grad_attrs


def install():
    """Install process-local adapters; caller enables strict determinism.

    Call once before running the pipeline. This changes the
    summation order relative to upstream atomicAdd and canonicalizes equal-depth
    face priority. Differentiation formulas and non-tied forward values are unchanged.
    """
    global _originals
    import pytorch3d
    from pytorch3d import _C
    if pytorch3d.__version__ != "0.7.9":
        raise RuntimeError("Adapter validated only for PyTorch3D 0.7.9")
    _load()
    if _originals is None:
        _originals = (_C.rasterize_meshes_backward, _C.interp_face_attrs_backward,
                      _C.rasterize_meshes)
        _C.rasterize_meshes = _raster_forward
        _C.rasterize_meshes_backward = _raster_backward
        _C.interp_face_attrs_backward = _interp_backward


def uninstall():
    """Restore upstream functions in this process."""
    global _originals
    if _originals is not None:
        from pytorch3d import _C
        (_C.rasterize_meshes_backward, _C.interp_face_attrs_backward,
         _C.rasterize_meshes) = _originals
        _originals = None
