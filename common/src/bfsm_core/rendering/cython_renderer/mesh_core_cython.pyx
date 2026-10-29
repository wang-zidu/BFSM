import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "mesh_core.h":
    void _MeshRenderer_cpu_core(
        float* image, float* vertices, int* triangles,
        float* colors,
        float* depth_buffer, int* triangle_buffer, float* barycentric_weight,
        int nver, int ntri,
        int h, int w, int c, bint perspective)

def MeshRenderer_cpu_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] colors not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                np.ndarray[int, ndim=2, mode = "c"] triangle_buffer not None,
                np.ndarray[float, ndim=3, mode = "c"] barycentric_weight not None,
                int nver, int ntri,
                int h, int w, int c, bint perspective=False
                ):
    # Validate the actual buffers before passing unchecked pointers to C++.
    if nver < 0 or ntri < 0 or h < 1 or w < 1 or c != 3:
        raise ValueError('Invalid raster dimensions or channel count')
    if (vertices.shape[0] != nver or vertices.shape[1] != 3
            or triangles.shape[0] != ntri or triangles.shape[1] != 3
            or colors.shape[0] != nver or colors.shape[1] != c
            or image.shape[0] != h or image.shape[1] != w or image.shape[2] != c
            or depth_buffer.shape[0] != h or depth_buffer.shape[1] != w
            or triangle_buffer.shape[0] != h or triangle_buffer.shape[1] != w
            or barycentric_weight.shape[0] != h or barycentric_weight.shape[1] != w
            or barycentric_weight.shape[2] != 3):
        raise ValueError('Raster buffer dimensions do not match their declared sizes')
    if ntri and (triangles.min() < 0 or triangles.max() >= nver):
        raise ValueError('Triangle index is outside the vertex buffer')
    if not np.isfinite(vertices).all() or not np.isfinite(colors).all():
        raise ValueError('Raster vertices and colors must be finite')
    if np.any(np.abs(vertices[:, :2].astype(np.float64)) > np.iinfo(np.int32).max - 1):
        raise ValueError('Projected coordinates exceed the native raster integer range')
    if perspective:
        if np.any(vertices[:, 2] >= 0):
            raise ValueError('Perspective raster depths must be negative camera Z')
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            if not np.isfinite(1.0 / vertices[:, 2]).all():
                raise ValueError('Perspective reciprocal depth must remain finite')
    for buffer in (image, depth_buffer, triangle_buffer, barycentric_weight):
        if not buffer.flags.writeable:
            raise ValueError('Raster output buffers must be writable')
    _MeshRenderer_cpu_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(colors), 
        <float*> np.PyArray_DATA(depth_buffer), <int*> np.PyArray_DATA(triangle_buffer), <float*> np.PyArray_DATA(barycentric_weight),
        nver, ntri,
        h, w, c, perspective)
