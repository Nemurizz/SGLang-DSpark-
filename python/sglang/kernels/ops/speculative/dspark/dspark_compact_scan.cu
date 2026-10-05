#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

namespace {
__global__ void dspark_compact_scan_kernel(const int32_t* __restrict__ verify_lens, int32_t bs, int64_t* __restrict__ req_out, int64_t* __restrict__ within_out, bool* __restrict__ valid_out, int64_t* __restrict__ verify_ids_out, const int64_t* __restrict__ draft_block_ids, const int64_t* __restrict__ draft_tokens, int32_t gamma, int64_t anchor_stride, int64_t n) {
  extern __shared__ int64_t prefix[];
  int64_t* starts = prefix; int64_t* inclusive = prefix + bs;
  if (threadIdx.x == 0) { int64_t cursor=0; for (int32_t i=0;i<bs;++i) { starts[i]=cursor; cursor += (int64_t)verify_lens[i]; inclusive[i]=cursor; } }
  __syncthreads();
  const int64_t real_total = bs > 0 ? inclusive[bs-1] : 0;
  for (int64_t row=(int64_t)threadIdx.x; row<n; row+=(int64_t)blockDim.x) {
    if (row >= real_total || bs == 0) { req_out[row]=bs; within_out[row]=0; valid_out[row]=false; verify_ids_out[row]=0; continue; }
    int32_t lo=0, hi=bs;
    while (lo < hi) { int32_t mid=lo+(hi-lo)/2; if (inclusive[mid] <= row) lo=mid+1; else hi=mid; }
    int32_t req=lo; int64_t within=row-starts[req]; req_out[row]=req; within_out[row]=within; valid_out[row]=true;
    int64_t value=draft_block_ids[(int64_t)req*anchor_stride];
    if (within > 0 && gamma > 0) value=draft_tokens[(int64_t)req*gamma + within-1];
    verify_ids_out[row]=value;
  }
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> compact_scan(at::Tensor verify_lens, at::Tensor draft_block_ids, at::Tensor draft_tokens, int64_t padded_total) {
  TORCH_CHECK(verify_lens.is_cuda() && draft_block_ids.is_cuda() && draft_tokens.is_cuda(), "DSpark compact_scan expects CUDA tensors");
  TORCH_CHECK(verify_lens.scalar_type()==at::kInt && draft_block_ids.scalar_type()==at::kLong && draft_tokens.scalar_type()==at::kLong, "DSpark compact_scan dtype mismatch");
  TORCH_CHECK(verify_lens.dim()==1 && draft_block_ids.dim()==2 && draft_tokens.dim()==2, "DSpark compact_scan rank mismatch");
  TORCH_CHECK(draft_block_ids.size(0)==verify_lens.size(0) && draft_tokens.size(0)==verify_lens.size(0), "DSpark compact_scan batch mismatch");
  TORCH_CHECK(draft_tokens.size(1)>0 && padded_total>=0, "DSpark compact_scan invalid shape");
  TORCH_CHECK(verify_lens.is_contiguous() && draft_block_ids.is_contiguous() && draft_tokens.is_contiguous(), "DSpark compact_scan tensors must be contiguous");
  auto device=verify_lens.device(); auto i64=at::TensorOptions().dtype(at::kLong).device(device); auto boolean=at::TensorOptions().dtype(at::kBool).device(device);
  auto req=at::empty({padded_total},i64); auto within=at::empty({padded_total},i64); auto valid=at::empty({padded_total},boolean); auto ids=at::empty({padded_total},i64);
  int32_t bs=(int32_t)verify_lens.numel(); int32_t gamma=(int32_t)draft_tokens.size(1); int64_t stride=draft_block_ids.stride(0); auto stream=at::cuda::getCurrentCUDAStream();
  dspark_compact_scan_kernel<<<1,256,(size_t)(2*bs)*sizeof(int64_t),stream>>>(verify_lens.data_ptr<int32_t>(),bs,req.data_ptr<int64_t>(),within.data_ptr<int64_t>(),valid.data_ptr<bool>(),ids.data_ptr<int64_t>(),draft_block_ids.data_ptr<int64_t>(),draft_tokens.data_ptr<int64_t>(),gamma,stride,padded_total);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return std::make_tuple(req,within,valid,ids);
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("compact_scan", &compact_scan, "DSpark compact/scan and verify-id gather (CUDA)"); }
