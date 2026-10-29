/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <thrust/tuple.h>
#include <cstdio>
#include <tuple>
#include "rasterize_points/rasterization_utils.cuh"
#include "utils/float_math.cuh"
#include "utils/geometry_utils.cuh"

// Upstream formulas preserved. Each pixel/face contribution writes a unique row.
__device__ thrust::tuple<float3, float3, float3> GetSingleFaceVerts(
    const float* face_verts,
    int face_idx) {
  const float x0 = face_verts[face_idx * 9 + 0];
  const float y0 = face_verts[face_idx * 9 + 1];
  const float z0 = face_verts[face_idx * 9 + 2];
  const float x1 = face_verts[face_idx * 9 + 3];
  const float y1 = face_verts[face_idx * 9 + 4];
  const float z1 = face_verts[face_idx * 9 + 5];
  const float x2 = face_verts[face_idx * 9 + 6];
  const float y2 = face_verts[face_idx * 9 + 7];
  const float z2 = face_verts[face_idx * 9 + 8];

  const float3 v0xyz = make_float3(x0, y0, z0);
  const float3 v1xyz = make_float3(x1, y1, z1);
  const float3 v2xyz = make_float3(x2, y2, z2);

  return thrust::make_tuple(v0xyz, v1xyz, v2xyz);
}

__global__ void PixelContributionsKernel(
    const float* face_verts, // (F, 3, 3)
    const int64_t* pix_to_face, // (N, H, W, K)
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const int N,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_bary, // (N, H, W, K, 3)
    const float* grad_dists, // (N, H, W, K)
    float* grad_face_verts) { // (F, 3, 3)

  // Parallelize over each pixel in images of
  // size H * W, for each image in the batch of size N.
  const auto num_threads = gridDim.x * blockDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int t_i = tid; t_i < N * H * W; t_i += num_threads) {
    // Convert linear index to 3D index
    const int n = t_i / (H * W); // batch index.
    const int pix_idx = t_i % (H * W);

    // Reverse ordering of X and Y axes.
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);
    const float2 pxy = make_float2(xf, yf);

    // Loop over all the faces for this pixel.
    for (int k = 0; k < K; k++) {
      // Index into (N, H, W, K, :) grad tensors
      // pixel index + top k index
      int i = n * H * W * K + pix_idx * K + k;

      const int f = pix_to_face[i];
      if (f < 0) {
        continue; // padded face.
      }
      // Get xyz coordinates of the three face vertices.
      const auto v012 = GetSingleFaceVerts(face_verts, f);
      const float3 v0 = thrust::get<0>(v012);
      const float3 v1 = thrust::get<1>(v012);
      const float3 v2 = thrust::get<2>(v012);

      // Only neex xy for barycentric coordinate and distance calculations.
      const float2 v0xy = make_float2(v0.x, v0.y);
      const float2 v1xy = make_float2(v1.x, v1.y);
      const float2 v2xy = make_float2(v2.x, v2.y);

      // Get upstream gradients for the face.
      const float grad_dist_upstream = grad_dists[i];
      const float grad_zbuf_upstream = grad_zbuf[i];
      const float grad_bary_upstream_w0 = grad_bary[i * 3 + 0];
      const float grad_bary_upstream_w1 = grad_bary[i * 3 + 1];
      const float grad_bary_upstream_w2 = grad_bary[i * 3 + 2];
      const float3 grad_bary_upstream = make_float3(
          grad_bary_upstream_w0, grad_bary_upstream_w1, grad_bary_upstream_w2);

      const float3 b_w = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
      const float3 b_pp = !perspective_correct
          ? b_w
          : BarycentricPerspectiveCorrectionForward(b_w, v0.z, v1.z, v2.z);

      const float3 b_w_clip =
          !clip_barycentric_coords ? b_pp : BarycentricClipForward(b_pp);

      const bool inside = b_pp.x > 0.0f && b_pp.y > 0.0f && b_pp.z > 0.0f;
      const float sign = inside ? -1.0f : 1.0f;

      auto grad_dist_f = PointTriangleDistanceBackward(
          pxy, v0xy, v1xy, v2xy, sign * grad_dist_upstream);
      const float2 ddist_d_v0 = thrust::get<1>(grad_dist_f);
      const float2 ddist_d_v1 = thrust::get<2>(grad_dist_f);
      const float2 ddist_d_v2 = thrust::get<3>(grad_dist_f);

      // Upstream gradient for barycentric coords from zbuf calculation:
      // zbuf = bary_w0 * z0 + bary_w1 * z1 + bary_w2 * z2
      // Therefore
      // d_zbuf/d_bary_w0 = z0
      // d_zbuf/d_bary_w1 = z1
      // d_zbuf/d_bary_w2 = z2
      const float3 d_zbuf_d_bwclip = make_float3(v0.z, v1.z, v2.z);

      // Total upstream barycentric gradients are the sum of
      // external upstream gradients and contribution from zbuf.
      const float3 grad_bary_f_sum =
          (grad_bary_upstream + grad_zbuf_upstream * d_zbuf_d_bwclip);

      float3 grad_bary0 = grad_bary_f_sum;

      if (clip_barycentric_coords) {
        grad_bary0 = BarycentricClipBackward(b_w, grad_bary_f_sum);
      }

      float dz0_persp = 0.0f, dz1_persp = 0.0f, dz2_persp = 0.0f;
      if (perspective_correct) {
        auto perspective_grads = BarycentricPerspectiveCorrectionBackward(
            b_w, v0.z, v1.z, v2.z, grad_bary0);
        grad_bary0 = thrust::get<0>(perspective_grads);
        dz0_persp = thrust::get<1>(perspective_grads);
        dz1_persp = thrust::get<2>(perspective_grads);
        dz2_persp = thrust::get<3>(perspective_grads);
      }

      auto grad_bary_f =
          BarycentricCoordsBackward(pxy, v0xy, v1xy, v2xy, grad_bary0);
      const float2 dbary_d_v0 = thrust::get<1>(grad_bary_f);
      const float2 dbary_d_v1 = thrust::get<2>(grad_bary_f);
      const float2 dbary_d_v2 = thrust::get<3>(grad_bary_f);

      grad_face_verts[i * 9 + 0] = dbary_d_v0.x + ddist_d_v0.x;
      grad_face_verts[i * 9 + 1] = dbary_d_v0.y + ddist_d_v0.y;
      grad_face_verts[i * 9 + 2] = grad_zbuf_upstream * b_w_clip.x + dz0_persp;
      grad_face_verts[i * 9 + 3] = dbary_d_v1.x + ddist_d_v1.x;
      grad_face_verts[i * 9 + 4] = dbary_d_v1.y + ddist_d_v1.y;
      grad_face_verts[i * 9 + 5] = grad_zbuf_upstream * b_w_clip.y + dz1_persp;
      grad_face_verts[i * 9 + 6] = dbary_d_v2.x + ddist_d_v2.x;
      grad_face_verts[i * 9 + 7] = dbary_d_v2.y + ddist_d_v2.y;
      grad_face_verts[i * 9 + 8] = grad_zbuf_upstream * b_w_clip.z + dz2_persp;
    }
  }
}


at::Tensor pixel_contributions(
    const at::Tensor& face_verts, const at::Tensor& pix_to_face,
    const at::Tensor& grad_zbuf, const at::Tensor& grad_bary,
    const at::Tensor& grad_dists, bool perspective_correct, bool clip_barycentric_coords) {
  TORCH_CHECK(face_verts.is_cuda() && face_verts.scalar_type() == at::kFloat,
              "face_verts must be CUDA float32");
  TORCH_CHECK(pix_to_face.scalar_type() == at::kLong && pix_to_face.dim() == 4,
              "pix_to_face must be int64 NHWK");
  for (const auto& t : {pix_to_face, grad_zbuf, grad_bary, grad_dists}) {
    TORCH_CHECK(t.device() == face_verts.device() && t.is_contiguous(),
                "all inputs must be contiguous on the same CUDA device");
  }
  TORCH_CHECK(face_verts.is_contiguous(), "face_verts must be contiguous");
  TORCH_CHECK(face_verts.dim() == 3 && face_verts.size(1) == 3 && face_verts.size(2) == 3,
              "face_verts must have shape (F, 3, 3)");
  TORCH_CHECK(grad_zbuf.scalar_type() == at::kFloat &&
              grad_bary.scalar_type() == at::kFloat && grad_dists.scalar_type() == at::kFloat,
              "upstream gradients must be float32");
  TORCH_CHECK(grad_zbuf.sizes() == pix_to_face.sizes() &&
              grad_dists.sizes() == pix_to_face.sizes() &&
              grad_bary.dim() == 5 && grad_bary.size(4) == 3 &&
              grad_bary.numel() == pix_to_face.numel() * 3,
              "upstream gradient shapes do not match raster outputs");
  at::cuda::CUDAGuard guard(face_verts.device());
  auto result = at::zeros({pix_to_face.numel(), 9}, face_verts.options());
  if (pix_to_face.numel() == 0 || face_verts.numel() == 0) return result;
  PixelContributionsKernel<<<1024, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
      face_verts.data_ptr<float>(), pix_to_face.data_ptr<int64_t>(),
      perspective_correct, clip_barycentric_coords,
      pix_to_face.size(0), pix_to_face.size(1), pix_to_face.size(2), pix_to_face.size(3),
      grad_zbuf.data_ptr<float>(), grad_bary.data_ptr<float>(), grad_dists.data_ptr<float>(),
      result.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}
