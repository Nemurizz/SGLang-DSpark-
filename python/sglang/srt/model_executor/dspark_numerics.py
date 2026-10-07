"""Default-OFF, narrowly guarded DSpark BF16 reduction correctness probe.

NOT batch-invariance or a performance guarantee. Full precision reduction may
change original B1 rounding; callers must gate tokens AND acceptance against
preserved original controls. Does not enable deterministic mode or radix cache.
"""


def configure_bf16_full_reduction(*, enabled, is_dspark, device, dtype, quantization,
                                tp_size, graphs_disabled, deterministic, backend,
                                gpu_id=0, torch_module=None):
    if not enabled:
        return False
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if (not is_dspark or device != "cuda" or dtype != torch.bfloat16 or quantization is not None
            or tp_size != 1 or not graphs_disabled or deterministic or backend not in ("auto", "torch")
            or torch.__version__ != "2.13.0+cu130"
            or tuple(torch.cuda.get_device_capability(gpu_id)) != (12, 0)):
        raise ValueError("DSpark BF16 reduction probe only supports checked Torch2.13/cu130 SM120 BF16 TP1 graph-OFF non-deterministic unquantized torch backend")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return True
