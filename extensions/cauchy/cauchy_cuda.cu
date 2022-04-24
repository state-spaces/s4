#include <stdio.h>
// On pytorch 1.10 and CUDA 10.2, I get compilation errors on torch/csrc/api/include/torch/nn/cloneable.h
// So we'll only include torch/python.h instead of torch/extension.h
// Similar to https://github.com/getkeops/keops/blob/3efd428b55c724b12f23982c06de00bc4d02d903/pykeops/torch_headers.h.in#L8
// #include <torch/extension.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>  // For getCurrentCUDAStream
#include <THC/THCAtomics.cuh>  // For atomicAdd on complex
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/util/complex.h>  // For scalar_value_type
#include "map.h"  // For the MAP macro, i.e. for_each over the arguments


#ifndef ITEMS_PER_THREAD_SYM_FWD_VALUES
  #define ITEMS_PER_THREAD_SYM_FWD_VALUES {2, 4, 8, 16, 32, 32, 32, 64, 64, 64}
#endif
#ifndef MAX_BLOCK_SIZE_VALUE
  #define MAX_BLOCK_SIZE_VALUE 256
#endif
#ifndef ITEMS_PER_THREAD_SYM_BWD_VALUE
  #define ITEMS_PER_THREAD_SYM_BWD_VALUE 32
#endif

static constexpr int ITEMS_PER_THREAD_FWD = 64;
static constexpr int ITEMS_PER_THREAD_BWD = 32;
static constexpr int ITEMS_PER_THREAD_SYM_FWD[] = ITEMS_PER_THREAD_SYM_FWD_VALUES;
static constexpr int MAX_BLOCK_SIZE = MAX_BLOCK_SIZE_VALUE;
static constexpr int ITEMS_PER_THREAD_SYM_BWD = ITEMS_PER_THREAD_SYM_BWD_VALUE;

template <typename T, size_t N>
using CudaAcsr = at::GenericPackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;
constexpr __host__ __device__ int div_up_const(int a, int b) { return (a + b - 1) / b; }

__host__ __device__ static inline int div_up(int a, int b) { return (a + b - 1) / b;}

template<typename scalar_t>
__device__ __forceinline__ void initalize_shared_mem(scalar_t mem[], int size) {
  // Assume that block only uses x and y coordinates, not z coordinate
  for (int t = threadIdx.x + threadIdx.y * blockDim.x; t < size; t += blockDim.x * blockDim.y) {
    mem[t] = 0;
  }
}

template <typename scalar_t, int log_N>
__global__ void cauchy_mult_fwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                            CudaAcsr<scalar_t, 1> z,
                                            CudaAcsr<scalar_t, 2> w,
                                            CudaAcsr<scalar_t, 2> out,
                                            int L) {
  constexpr int N = 1 << log_N;
  constexpr int blockDimx = div_up_const(N, ITEMS_PER_THREAD_FWD);
  constexpr int blockDimy = MAX_BLOCK_SIZE / blockDimx;
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  // TODO: generalize for N > 256
  __shared__ char s_v_char[N * sizeof(scalar_t)];
  scalar_t *s_v = (scalar_t *)&s_v_char;
  __shared__ char s_w_char[N * sizeof(scalar_t)];
  scalar_t *s_w = (scalar_t *)&s_w_char;
  __shared__ char s_z_char[blockDimy * sizeof(scalar_t)];
  scalar_t *s_z = (scalar_t *)&s_z_char;
  __shared__ char s_out_char[blockDimy * sizeof(scalar_t)];
  scalar_t *s_out = (scalar_t *)&s_out_char;
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int L_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int L_block_start = blockIdx.y * blockDim.y;
  for (int N_idx = threadIdx.x + threadIdx.y * blockDim.x; N_idx < N; N_idx += blockDim.x * blockDim.y) {
    s_v[N_idx] = v[batch_idx][N_idx];
    s_w[N_idx] = w[batch_idx][N_idx];
  }
  // for (int l = threadIdx.x + threadIdx.y * blockDim.x; l < blockDim.y && L_block_start + l < L; l += blockDim.x * blockDim.y) {
  //   s_z[l] = z[L_block_start + l];
  // }
  if (tid < blockDim.y && L_block_start + tid < L) {
    s_z[tid] = z[L_block_start + tid];
  }
  // if (threadIdx.x == 0 && L_idx < L) {
  //   s_z[threadIdx.y] = z[L_idx];
  // }
  __syncthreads();
  scalar_t result = 0;
  if (L_idx < L) {
    scalar_t t_z = s_z[threadIdx.y];
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD_FWD; ++item) {
      int N_idx = item * blockDimx + threadIdx.x;
      // result += s_v[N_idx] / (t_z - s_w[N_idx]);
      scalar_t diff_inv = scalar_t(1.0) / (t_z - s_w[N_idx]);
      result += s_v[N_idx] * diff_inv;
    }
    // #pragma unroll
    // for (int N_idx = threadIdx.x; N_idx < N; N_idx += blockDimx) {
    //     result += s_v[N_idx] / (t_z - s_w[N_idx]);
    // }
  }
  // TODO: this only works for N a power of 2
  #pragma unroll
  for (int offset = blockDimx / 2; offset > 0; offset /= 2) {
    result += WARP_SHFL_DOWN(result, offset);
  }
  // if ((L_idx < L) && (threadIdx.x == 0)) {
  //   out[batch_idx][L_idx] = result;
  // }
  if ((threadIdx.x == 0) && (L_idx < L)) {
    s_out[threadIdx.y] = result;
  }
  __syncthreads();
  if (tid < blockDim.y && L_block_start + tid < L) {
    out[batch_idx][L_block_start + tid] = s_out[tid];
  }
}

torch::Tensor cauchy_mult_fwd_cuda(torch::Tensor v,
                                   torch::Tensor z,
                                   torch::Tensor w) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  auto out = torch::empty({batch_size, L}, torch::dtype(v.dtype()).device(v.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto out_a = out.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  int block_x = div_up(N, ITEMS_PER_THREAD_FWD);
  dim3 block(block_x, MAX_BLOCK_SIZE / block_x);
  dim3 grid(batch_size, div_up(L, block.y));
  switch (N) {
  case 64:
    cauchy_mult_fwd_cuda_kernel<scalar_t, 6>
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, out_a, L);
  }
  return out;
}

template <typename scalar_t>
__global__ void cauchy_mult_bwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                            CudaAcsr<scalar_t, 1> z,
                                            CudaAcsr<scalar_t, 2> w,
                                            CudaAcsr<scalar_t, 2> dout,
                                            CudaAcsr<scalar_t, 2> dv,
                                            CudaAcsr<scalar_t, 2> dw,
                                            int L) {
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  // TODO: generalize for N > 256
  __shared__ char s_v_char[sizeof(scalar_t)];
  scalar_t *s_v = (scalar_t *)&s_v_char;
  __shared__ char s_w_char[ sizeof(scalar_t)];
  scalar_t *s_w = (scalar_t *)&s_w_char;
  __shared__ char s_dv_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *s_dv = (scalar_t *)&s_dv_char;
  __shared__ char s_dw_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *s_dw = (scalar_t *)&s_dw_char;
  int batch_idx = blockIdx.x;
  int N_idx = blockIdx.y;
  int tid = threadIdx.x;
  if (tid == 0) {
    s_v[0] = v[batch_idx][N_idx];
    s_w[0] = w[batch_idx][N_idx];
  }
  __syncthreads();
  scalar_t t_v = s_v[0];
  scalar_t t_w = s_w[0];
  scalar_t t_dv = 0;
  scalar_t t_dw = 0;
  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD_BWD; ++item) {
    int l = item * blockDim.x + threadIdx.x;
    // if (l < L) {
    scalar_t t_dout = dout[batch_idx][l];
    scalar_t diff_conj_inv = scalar_t(1.0) / std::conj(z[l] - t_w);
    scalar_t prod = t_dout * diff_conj_inv;
    t_dv += prod;
    t_dw += prod * diff_conj_inv;
    // }
  }
  // for (int item = 0; item < ITEMS_PER_THREAD_BWD / 2; ++item) {
  //   int l_1 = item * 2 * blockDim.x + threadIdx.x;
  //   int l_2 = (item * 2 + 1) * blockDim.x + threadIdx.x;
  //   scalar_t t_dout_1 = dout[batch_idx][l_1];
  //   scalar_t denom_1 = std::conj(z[l_1] - t_w);
  //   scalar_t t_dout_2 = dout[batch_idx][l_2];
  //   scalar_t denom_2 = std::conj(z[l_2] - t_w);
  //   scalar_t denom_prod_inv = scalar_t(1) / (denom_1 * denom_2);
  //   scalar_t denom_1_inv = denom_2 * denom_prod_inv;
  //   scalar_t denom_2_inv = denom_1 * denom_prod_inv;
  //   scalar_t prod_1 = t_dout_1 * denom_1_inv;
  //   scalar_t prod_2 = t_dout_2 * denom_2_inv;
  //   t_dv += prod_1 + prod_2;
  //   t_dw += prod_1 * denom_1_inv + prod_2 * denom_2_inv;
  //   t_dv += (t_dout_1 * denom_2 + t_dout_2 * denom_1) * denom_prod_inv;
  //   t_dw += (t_dout_1 * denom_2 * denom_2 + t_dout_2 * denom_1 * denom_1) * denom_prod_inv * denom_prod_inv;
  // }
  t_dv = at::native::cuda_utils::BlockReduceSum<scalar_t>(t_dv, s_dv);
  t_dw = at::native::cuda_utils::BlockReduceSum<scalar_t>(t_dw, s_dw);
  if (tid == 0) {
    dv[batch_idx][N_idx] = t_dv;
    dw[batch_idx][N_idx] = t_dw * std::conj(t_v);
  }
}


std::tuple<torch::Tensor, torch::Tensor>
cauchy_mult_bwd_cuda(torch::Tensor v,
                     torch::Tensor z,
                     torch::Tensor w,
                     torch::Tensor dout) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  auto dv = torch::empty({batch_size, N}, torch::dtype(v.dtype()).device(v.device()));
  auto dw = torch::empty({batch_size, N}, torch::dtype(w.dtype()).device(w.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto dout_a = dout.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto dv_a = dv.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto dw_a = dw.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  // Need to take max, otherwise each block has fewer than 1 full warp, causing
  // at::native::cuda_utils::BlockReduceSum to produce wrong resutl.
  // Otherwise we assume L > ITEMS_PER_THREAD_BWD * C10_WARP_SIZE
  int block_x = max(div_up(L, ITEMS_PER_THREAD_BWD), C10_WARP_SIZE);
  // TODO: assume that L is a multiple of ITEMS_PER_THREAD_BWD
  dim3 block(block_x);
  dim3 grid(batch_size, N);
  cauchy_mult_bwd_cuda_kernel<scalar_t>
    <<<grid, block, 0, stream>>>(v_a, z_a, w_a, dout_a, dv_a, dw_a, L);
  return std::make_tuple(dv, dw);
}

template <typename scalar_t, int log_N,
            int items_per_thread=ITEMS_PER_THREAD_SYM_FWD[log_N - 1]>
__global__ void cauchy_mult_sym_fwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                                CudaAcsr<scalar_t, 1> z,
                                                CudaAcsr<scalar_t, 2> w,
                                                CudaAcsr<scalar_t, 2> out,
                                                int L) {
  // Get the float type from the complex type
  // https://github.com/pytorch/pytorch/blob/bceb1db885cafa87fe8d037d8f22ae9649a1bba0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp#L213
  using float_t = typename at::scalar_value_type<scalar_t>::type;
  constexpr int N = 1 << log_N;
  constexpr int blockDimx = div_up_const(N, items_per_thread);
  constexpr int blockDimy = MAX_BLOCK_SIZE / blockDimx;
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  // __shared__ float_t s_vr[N];
  __shared__ char s_v_char[N * sizeof(scalar_t)];
  scalar_t *s_v = (scalar_t *)&s_v_char;
  __shared__ char s_w_char[N * sizeof(scalar_t)];
  scalar_t *s_w = (scalar_t *)&s_w_char;
  // __shared__ float_t s_wr[N];
  // __shared__ float_t s_wnorm[N];
  // __shared__ float_t s_vwconj_r[N];
  __shared__ char s_z_char[blockDimy * sizeof(scalar_t)];
  scalar_t *s_z = (scalar_t *)&s_z_char;
  __shared__ char s_out_char[blockDimy * sizeof(scalar_t)];
  scalar_t *s_out = (scalar_t *)&s_out_char;
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int L_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int L_block_start = blockIdx.y * blockDim.y;
  for (int N_idx = threadIdx.x + threadIdx.y * blockDim.x; N_idx < N; N_idx += blockDim.x * blockDim.y) {
    scalar_t t_v = v[batch_idx][N_idx];
    scalar_t t_w = w[batch_idx][N_idx];
    s_v[N_idx] = t_v;
    s_w[N_idx] = t_w;
    // s_vr[N_idx] = std::real(t_v);
    // s_wr[N_idx] = std::real(t_w);
    // s_wnorm[N_idx] = std::norm(t_w);
    // s_vwconj_r[N_idx] = std::real(t_v) * std::real(t_w) + std::imag(t_v) * std::imag(t_w);
    // Compiler is able to optimize this, so the two lines give idential speed;
    // s_vwconj_r[N_idx] = std::real(t_v * std::conj(t_w));
  }
  if (tid < blockDim.y && L_block_start + tid < L) {
    s_z[tid] = z[L_block_start + tid];
  }
  __syncthreads();
  scalar_t result = 0;
  if (L_idx < L) {
    scalar_t t_z = s_z[threadIdx.y];
    scalar_t t_z_sq = t_z * t_z;
    // c10::complex<double> result = 0;
    #pragma unroll
    // for (int item = 0; item < items_per_thread; ++item) {
    //   int N_idx = item * blockDimx + threadIdx.x;
    //   // int N_idx = item * blockDim.x + threadIdx.x;
    //   // scalar_t t_w = s_w[N_idx];
    //   // float t_vr = s_vr[N_idx];
    //   // scalar_t denom_inv = scalar_t(1.0) / ((t_z - t_w) * (t_z - std::conj(t_w)));
    //   // scalar_t denom_inv = scalar_t(1.0) / (t_z * t_z - 2 * t_z * std::real(t_w) + t_w * std::conj(t_w));
    //   // scalar_t denom_inv = scalar_t(1.0) / (t_z * t_z - 2 * t_z * std::real(t_w) + std::norm(t_w));
    //   // result += (t_z * std::real(t_v) - std::real(t_v) * std::real(t_w) - std::imag(t_v) * std::imag(t_w)) * denom_inv;
    //   scalar_t denom_inv = scalar_t(1.0) / (t_z_sq - 2 * t_z * s_wr[N_idx] + s_wnorm[N_idx]);
    //   result += (t_z * s_vr[N_idx] - s_vwconj_r[N_idx]) * denom_inv;
    //   // These next 2 lines assume that z is a root of unity
    //   // scalar_t denom_inv = scalar_t(1.0) / (t_z - 2 * std::real(t_w) + std::norm(t_w) * std::conj(t_z));
    //   // result += (std::real(t_v) - (std::real(t_v) * std::real(t_w) + std::imag(t_v) * std::imag(t_w)) * std::conj(t_z)) * denom_inv;
    // }
    // for (int item = 0; item < items_per_thread / 2; ++item) {
    //   int N_idx_1 = item * 2 * blockDimx + threadIdx.x;
    //   int N_idx_2 = (item * 2 + 1) * blockDimx + threadIdx.x;
    //   scalar_t denom_1 = (t_z_sq - 2 * t_z * s_wr[N_idx_1] + s_wnorm[N_idx_1]);
    //   scalar_t nom_1 = (t_z * s_vr[N_idx_1] - s_vwconj_r[N_idx_1]);
    //   scalar_t denom_2 = (t_z_sq - 2 * t_z * s_wr[N_idx_2] + s_wnorm[N_idx_2]);
    //   scalar_t nom_2 = (t_z * s_vr[N_idx_2] - s_vwconj_r[N_idx_2]);
    //   scalar_t denom_prod_inv = scalar_t(1) / (denom_1 * denom_2);
    //   result += (nom_1 * denom_2 + nom_2 * denom_1) * denom_prod_inv;
    // }
    // Combining the two terms (a/b + c/d = (ad + bc)/(bd)) seems to increase numerical errors.
    // Using nvcc --use_fast_math is yields the same speed between the versions.
    // So we don't combine the two terms.
    for (int item = 0; item < items_per_thread; ++item) {
      int N_idx = item * blockDimx + threadIdx.x;
      // scalar_t denom = (t_z_sq - 2 * t_z * s_wr[N_idx] + s_wnorm[N_idx]);
      // scalar_t nom = (t_z * s_vr[N_idx] - s_vwconj_r[N_idx]);
      // result += nom / denom;
      result += s_v[N_idx] / (t_z - s_w[N_idx]) + std::conj(s_v[N_idx]) / (t_z - std::conj(s_w[N_idx]));
    }
  }
  // TODO: this only works for N a power of 2
  #pragma unroll
  for (int offset = blockDimx / 2; offset > 0; offset /= 2) {
    result += WARP_SHFL_DOWN(result, offset);
  }
  if ((threadIdx.x == 0) && (L_idx < L)) {
    s_out[threadIdx.y] = result;
  }
  __syncthreads();
  if (tid < blockDim.y && L_block_start + tid < L) {
    // out[batch_idx][L_block_start + tid] = 2 * s_out[tid];
    out[batch_idx][L_block_start + tid] = s_out[tid];
  }
}

torch::Tensor cauchy_mult_sym_fwd_cuda(torch::Tensor v,
                                       torch::Tensor z,
                                       torch::Tensor w) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  auto out = torch::empty({batch_size, L}, torch::dtype(v.dtype()).device(v.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto out_a = out.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  int log_N = int(log2((double) N));
  int block_x = div_up(N, ITEMS_PER_THREAD_SYM_FWD[log_N - 1]);
  dim3 block(block_x, MAX_BLOCK_SIZE / block_x);
  dim3 grid(batch_size, div_up(L, block.y));
  switch (log_N) {
    #define CASE_LOG_N(log_N_val) case log_N_val:                 \
    cauchy_mult_sym_fwd_cuda_kernel<scalar_t, log_N_val>          \
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, out_a, L); break;
    MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  }
  #undef CASE_LOG_N
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename scalar_t, bool check_L_boundary>
__global__ void cauchy_mult_sym_bwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                                CudaAcsr<scalar_t, 1> z,
                                                CudaAcsr<scalar_t, 2> w,
                                                CudaAcsr<scalar_t, 2> dout,
                                                CudaAcsr<scalar_t, 3> dv,
                                                CudaAcsr<scalar_t, 3> dw,
                                                int L,
                                                int L_chunk_size) {
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  // __shared__ char s_v_char[sizeof(scalar_t)];
  // scalar_t *s_v = (scalar_t *)&s_v_char;
  __shared__ char s_w_conj_char[ sizeof(scalar_t)];
  scalar_t *s_w_conj = (scalar_t *)&s_w_conj_char;
  __shared__ char s_dv_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *s_dv = (scalar_t *)&s_dv_char;
  __shared__ char s_dw_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *s_dw = (scalar_t *)&s_dw_char;
  int batch_idx = blockIdx.x;
  int N_idx = blockIdx.y;
  int L_chunk_idx = blockIdx.z;
  int tid = threadIdx.x;
  if (tid == 0) {
    s_w_conj[0] = std::conj(w[batch_idx][N_idx]);
  }
  __syncthreads();
  scalar_t t_w_conj = s_w_conj[0];
  scalar_t t_w_conj_sq = t_w_conj * t_w_conj;
  scalar_t t_dv = 0;
  scalar_t t_dw = 0;
  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD_SYM_BWD; ++item) {
    int l = L_chunk_idx * L_chunk_size + item * blockDim.x + threadIdx.x;
    scalar_t t_dout, t_z;
    if (check_L_boundary) {
      t_dout = l < L ? dout[batch_idx][l] : 0;
      t_z = l < L ? z[l] : 1;
    } else {// Not checking boundary can speed it up quite a bit, around 30%.
      t_dout = dout[batch_idx][l];
      t_z = z[l];
    }
    scalar_t denom_1 = std::conj(t_z) - t_w_conj;
    scalar_t denom_2 = t_z - t_w_conj;
    scalar_t term_1 = t_dout / denom_1;
    scalar_t term_2 = std::conj(t_dout) / denom_2;
    t_dv += term_1 + term_2;
    t_dw += term_1 / denom_1 + term_2 / denom_2;
    // scalar_t denom_inv = scalar_t(1) / (std::norm(t_z) - 2 * std::real(t_z) * t_w_conj + t_w_conj_sq);
    // auto dout_z_real = std::real(t_dout) * std::real(t_z) - std::imag(t_dout) * std::imag(t_z);
    // Compiler is able to optimize this, so the two lines give idential speed;
    // auto dout_z_real = std::real(t_dout * t_z);
    // scalar_t dv_nom = (dout_z_real - std::real(t_dout) * t_w_conj);
    // t_dv += dv_nom * denom_inv;
    // scalar_t t_z_sq = t_z * t_z;
    // auto dout_z_sq_real = std::real(t_dout) * std::real(t_z_sq) - std::imag(t_dout) * std::imag(t_z_sq);
    // Compiler is able to optimize this, so the two lines give idential speed;
    // auto dout_z_sq_real = std::real(t_dout * t_z_sq);
    // scalar_t dw_nom = dout_z_sq_real - 2 * dout_z_real * t_w_conj + std::real(t_dout) * t_w_conj_sq;
    // t_dw += dw_nom * denom_inv * denom_inv;
  }
  t_dv = at::native::cuda_utils::BlockReduceSum<scalar_t>(t_dv, s_dv);
  t_dw = at::native::cuda_utils::BlockReduceSum<scalar_t>(t_dw, s_dw);
  if (tid == 0) {
    // dw[batch_idx][N_idx][L_chunk_idx] = 2 * t_dw * std::conj(v[batch_idx][N_idx]);
    dw[batch_idx][N_idx][L_chunk_idx] = t_dw * std::conj(v[batch_idx][N_idx]);
    // dv[batch_idx][N_idx][L_chunk_idx] = 2 * t_dv;
    dv[batch_idx][N_idx][L_chunk_idx] = t_dv;
  }
}

std::tuple<torch::Tensor, torch::Tensor>
cauchy_mult_sym_bwd_cuda(torch::Tensor v,
                         torch::Tensor z,
                         torch::Tensor w,
                         torch::Tensor dout) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  constexpr int MAX_BLOCK_SIZE = 1024;
  constexpr int MAX_L_CHUNK_SIZE = ITEMS_PER_THREAD_SYM_BWD * MAX_BLOCK_SIZE;
  const int n_L_chunks = div_up(L, MAX_L_CHUNK_SIZE);
  auto dv = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(v.dtype()).device(v.device()));
  auto dw = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(w.dtype()).device(w.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto dout_a = dout.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto dv_a = dv.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  auto dw_a = dw.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  // Each block need to have a multiple of 32 threads, otherwise
  // at::native::cuda_utils::BlockReduceSum to produce wrong result.
  // int block_x = max(div_up(L, ITEMS_PER_THREAD_SYM_BWD), C10_WARP_SIZE);
  const int L_chunk_size = min(L, MAX_L_CHUNK_SIZE);
  int block_x = div_up(L_chunk_size, ITEMS_PER_THREAD_SYM_BWD * C10_WARP_SIZE) * C10_WARP_SIZE;
  bool check_L_boundary = L != block_x * ITEMS_PER_THREAD_SYM_BWD * n_L_chunks;
  dim3 block(block_x);
  dim3 grid(batch_size, N, n_L_chunks);
  check_L_boundary
    ? cauchy_mult_sym_bwd_cuda_kernel<scalar_t, true>
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, dout_a, dv_a, dw_a, L, L_chunk_size)
    : cauchy_mult_sym_bwd_cuda_kernel<scalar_t, false>
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, dout_a, dv_a, dw_a, L, L_chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dv.sum(-1), dw.sum(-1));
}
