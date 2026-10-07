"""Read-only eager runtime attestation; never alters model tensors or outputs."""
import json
from pathlib import Path


def validate_runtime(data):
    expected = {"torch_batch_invariant": True, "fa4_batch_invariant": True,
                "backend_class": "FlashAttentionBackend", "fa_impl_ver": 4,
                "num_splits": 1, "decode_num_splits": 1, "page_size": 1,
                "kv_dtype": "torch.bfloat16", "deterministic": True,
                "graphs_disabled": True, "gemm_backend": "torch"}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("Stable FA4 runtime attestation failed: " + repr(data))


def make_hook(config):
    import torch
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import is_batch_invariant
    from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
    from sglang.srt.runtime_context import get_exec

    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    recorded = set()

    def hook(module, inputs, output):
        if len(inputs) < 4 or getattr(inputs[3], "req_pool_indices", None) is None:
            return None
        fb = inputs[3]
        mode = str(fb.forward_mode)
        if mode in recorded:
            return None
        backend = get_attn_backend()
        if hasattr(backend, "_select_backend"):
            backend = backend._select_backend(fb.forward_mode)
        data = {"profile_id": config["profile_id"], "mode": mode,
                "torch": torch.__version__, "torch_batch_invariant": is_batch_invariant_mode_enabled(),
                "fa4_batch_invariant": is_batch_invariant(), "backend_class": type(backend).__name__,
                "fa_impl_ver": getattr(backend, "fa_impl_ver", None),
                "num_splits": getattr(backend, "num_splits", None),
                "decode_num_splits": getattr(backend, "decode_num_splits", None),
                "page_size": backend.page_size,
                "kv_dtype": str(backend.token_to_kv_pool.get_key_buffer(0).dtype),
                "deterministic": get_exec().deterministic.enable_deterministic_inference,
                "graphs_disabled": all(check_cuda_graph_backend(phase, Backend.DISABLED)
                                       for phase in (Phase.PREFILL, Phase.DECODE)),
                "gemm_backend": get_bf16_gemm_backend().value,
                "scope": "runtime configuration attestation only; no tensor copies or timing"}
        validate_runtime(data)
        with (folder / f"runtime-{len(recorded):02d}.json").open("x") as stream:
            json.dump(data, stream, indent=2)
        recorded.add(mode)
        return None

    return hook
