
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>

using namespace cute;
using Element = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <typename ThreadMap, typename CtaShape, int MTile, int NTile,
          int KV, int D, int L>
struct DsparkKvVisitor {
  using ElementOutput = Element;
  using ElementCompute = float;
  static constexpr int HeadsPerTile = NTile / D;
  static constexpr int ElementsPerAccess = ThreadMap::kElementsPerAccess;
  static constexpr int kElementsPerAccess = ElementsPerAccess;
  struct Arguments {
    int64_t const* meta = nullptr;
    ElementCompute const* knw = nullptr;
    ElementCompute const* cos_sin = nullptr;
    int64_t const* positions = nullptr;
    int64_t const* locs = nullptr;
    ElementCompute eps = 1e-6f;
  };
  using Params = Arguments;
  struct SharedStorage {
    alignas(16) ElementCompute raw[MTile * NTile];
    alignas(16) ElementCompute sum[MTile * HeadsPerTile];
  };
  Params const* params_ptr;
  SharedStorage* shared_ptr;

  CUTLASS_HOST_DEVICE DsparkKvVisitor() : params_ptr(nullptr), shared_ptr(nullptr) {}
  CUTLASS_HOST_DEVICE DsparkKvVisitor(Params const& params, SharedStorage& shared)
      : params_ptr(&params), shared_ptr(&shared) {}

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_size;
    Params const* params_ptr;
    SharedStorage* shared_ptr;
    int row_offset;
    int col_offset;

    CUTLASS_DEVICE Callbacks(
        GTensor&& ga, RTensor&& ra, CTensor&& ca, ProblemShape ps,
        Params const* pp, SharedStorage* sp, int ro, int co)
        : tC_gAux(cute::forward<GTensor>(ga)),
          tC_rAux(cute::forward<RTensor>(ra)),
          tC_cAux(cute::forward<CTensor>(ca)),
          problem_size(ps), params_ptr(pp), shared_ptr(sp), row_offset(ro), col_offset(co) {}

    CUTLASS_DEVICE void begin_step(int) {
      __syncthreads();
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int frg_idx,
        cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      using ConvertInput = cutlass::NumericArrayConverter<
          ElementCompute, ElementAccumulator, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      ConvertInput cvt;
      auto frg = cvt(frg_acc);
      auto dst = recast<cutlass::Array<ElementCompute, FragmentSize>>(
          coalesce(tC_rAux));
      dst(frg_idx) = frg;
      return frg_acc;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      for (int i = 0; i < size(src_v); ++i) {
        auto coord = coord_v(i);
        int m = int(get<0>(coord));
        int n = int(get<1>(coord));
        bool guard = m >= 0 && m < int(get<0>(problem_size)) &&
                     n >= 0 && n < int(get<1>(problem_size));
        if (!guard) continue;
        int lm = m % MTile;
        int ln = n % NTile;
        shared_ptr->raw[lm * NTile + ln] = ElementCompute(Element(src_v(i)));
      }
      __syncthreads();
    }

    CUTLASS_DEVICE void end_epilogue() {
      if (threadIdx.x == 0) {
        for (int lm = 0; lm < MTile; ++lm) {
          for (int h = 0; h < HeadsPerTile; ++h) {
            ElementCompute acc = 0;
            for (int d = 0; d < D; ++d) {
              ElementCompute v = shared_ptr->raw[lm * NTile + h * D + d];
              acc += v * v;
            }
            shared_ptr->sum[lm * HeadsPerTile + h] = acc;
          }
        }
      }
      __syncthreads();
      int total = MTile * NTile;
      for (int idx = int(threadIdx.x); idx < total; idx += int(blockDim.x)) {
        int lm = idx / NTile;
        int ln = idx % NTile;
        int m = row_offset + lm;
        int n = col_offset + ln;
        if (m < 0 || m >= int(get<0>(problem_size)) ||
            n < 0 || n >= int(get<1>(problem_size))) {
          continue;
        }
        int layer = n / (2 * KV);
        int within = n % (2 * KV);
        bool is_k = within < KV;
        int d = within % D;
        int h = (within % KV) / D;
        ElementCompute inv = rsqrtf(
            shared_ptr->sum[lm * HeadsPerTile + (ln / D)] /
                ElementCompute(D) + params_ptr->eps);
        ElementCompute y;
        if (!is_k) {
          y = shared_ptr->raw[lm * NTile + ln];
        } else {
          int half = D / 2;
          ElementCompute c = params_ptr->cos_sin[
              params_ptr->positions[m] * D + (d % half)];
          ElementCompute ss = params_ptr->cos_sin[
              params_ptr->positions[m] * D + half + (d % half)];
          ElementCompute x1 = (d < half)
              ? shared_ptr->raw[lm * NTile + ln] * inv *
                    params_ptr->knw[layer * D + d]
              : shared_ptr->raw[lm * NTile + ln - half] * inv *
                    params_ptr->knw[layer * D + d - half];
          ElementCompute x2 = (d < half)
              ? shared_ptr->raw[lm * NTile + ln + half] * inv *
                    params_ptr->knw[layer * D + d + half]
              : shared_ptr->raw[lm * NTile + ln] * inv *
                    params_ptr->knw[layer * D + d];
          // Match the existing Triton path: norm/weight results are rounded
          // to BF16 before the RoPE multiply-add.
          x1 = ElementCompute(Element(x1));
          x2 = ElementCompute(Element(x2));
          y = d < half ? x1 * c - x2 * ss : x2 * c + x1 * ss;
        }
        int64_t loc = params_ptr->locs[m];
        if (loc < 0) continue;
        auto base = params_ptr->meta + layer * 4;
        int64_t ptr = base[is_k ? 0 : 1];
        int64_t stride = base[is_k ? 2 : 3];
        auto dst = reinterpret_cast<Element*>(ptr + loc * stride);
        dst[h * D + d] = Element(y);
      }
    }
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) { return args; }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx, ProblemShape problem_size) {
    Tensor dummy = make_tensor(
        make_gmem_ptr(const_cast<ElementCompute*>(params_ptr->knw)),
        make_shape(size<0>(problem_size), size<1>(problem_size), 1),
        make_stride(size<1>(problem_size), 1, 0));
    Tensor tC_gAux = recast<ElementCompute>(
        group_modes<3,6>(ThreadMap::partition(dummy, thread_idx,
                                               threadblock_tile_offset)));
    Tensor tC_rAux = make_tensor_like(take<0,3>(tC_gAux));
    Tensor cAux = make_identity_tensor(dummy.shape());
    Tensor tC_cAux = outer_partition(
        group_modes<3,6>(ThreadMap::partition(cAux, thread_idx,
                                               threadblock_tile_offset)),
        Shape<Int<1>>{}, (_0{}));
    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux),
                     ProblemShape>(cute::move(tC_gAux), cute::move(tC_rAux),
                                   cute::move(tC_cAux), problem_size, params_ptr,
                                   shared_ptr,
                                   threadblock_tile_offset.m() * CtaShape::kM,
                                   threadblock_tile_offset.n() * CtaShape::kN);
  }
};

using CtaShape = cutlass::gemm::GemmShape<16, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    CtaShape, WarpShape, Element, 8, 1>;
using Visitor = DsparkKvVisitor<ThreadMap, CtaShape, 16, 128, 1024, 128, 28>;
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    Element, LayoutA, cutlass::ComplexTransform::kNone, 8,
    Element, LayoutB, cutlass::ComplexTransform::kNone, 8,
    Element, LayoutC, 8, Accumulator, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    CtaShape, WarpShape, InstructionShape, Visitor,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, cutlass::arch::OpMultiplyAdd, 1>::GemmKernel;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor cutlass_kv_probe(torch::Tensor a, torch::Tensor w,
                               torch::Tensor meta, torch::Tensor knw,
                               torch::Tensor cos_sin, torch::Tensor positions,
                               torch::Tensor locs, double eps) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && meta.is_cuda() && knw.is_cuda());
  int m = a.size(0), k = a.size(1), n = w.size(0);
  auto out = torch::empty({m, n}, a.options());
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, {},
      reinterpret_cast<Element const*>(a.data_ptr()),
      reinterpret_cast<Element const*>(w.data_ptr()), nullptr, nullptr,
      int64_t(m) * k, int64_t(n) * k, int64_t(m) * n, int64_t(m) * n,
      k, k, n, n);
  args.epilogue = {reinterpret_cast<int64_t const*>(meta.data_ptr()),
                   reinterpret_cast<float const*>(knw.data_ptr()),
                   reinterpret_cast<float const*>(cos_sin.data_ptr()),
                   reinterpret_cast<int64_t const*>(positions.data_ptr()),
                   reinterpret_cast<int64_t const*>(locs.data_ptr()),
                   float(eps)};
  Gemm op;
  auto ws = torch::empty({static_cast<long>(op.get_workspace_size(args))},
                         torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
  auto status = op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, static_cast<int>(status));
  status = op.initialize(args, ws.data_ptr(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, static_cast<int>(status));
  status = op.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, static_cast<int>(status));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &cutlass_kv_probe); }
