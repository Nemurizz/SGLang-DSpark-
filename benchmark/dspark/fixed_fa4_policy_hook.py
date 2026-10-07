"""Explicit, process-local numerical host policy. OFF unless requested.

This factory IS a numerical configuration mutation, not a read-only observer.
It pins the EXISTING SM120 FA4 kernel's host-selected tile before any FA4
compile; it does not implement/edit a kernel, touch weights or change sampling.
Only the separately qualified BF16 Qwen3 40Q/8KV, HD128 profile is supported.
"""
import os


TILE = (64, 64)


def locked_options(options):
    if (options.get('head_dim'), options.get('head_dim_v'), options.get('num_head_kv'),
        options.get('qhead_per_kvhead')) != (128, 128, 8, 5):
        raise ValueError('Fixed numerical policy only supports Qwen3 40Q/8KV HD128')
    if options.get('has_bias') or options.get('is_local'):
        raise ValueError('Bias/local attention is not qualified')
    if options.get('tile_mn') not in (None, TILE):
        raise ValueError('Conflicting explicit FA4 tile')
    if not options.get('is_causal') or not options.get('pack_gqa'):
        raise ValueError('Only causal packed-GQA attention is qualified')
    return {**options, 'tile_mn': TILE}


def make_hook(config):
    import torch
    from sglang.kernels.ops.attention.fa4_sm120.runtime import Sm120ForwardHost
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend
    from sglang.srt.runtime_context import get_exec
    from stable_profile_hook import make_hook as attestation_hook

    if os.environ.get('SGLANG_DSPARK_STABLE_FA4_FIXED_TILE') != '1':
        raise ValueError('Fixed FA4 policy needs its explicit opt-in environment flag')
    if config.get('fixed_fa4_tile') != list(TILE) or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError('Unqualified fixed tile/device')
    if not get_exec().deterministic.enable_deterministic_inference or not all(
            check_cuda_graph_backend(phase, Backend.DISABLED) for phase in (Phase.PREFILL, Phase.DECODE)):
        raise ValueError('Fixed FA4 policy requires deterministic inference and all graphs OFF')
    marker = getattr(Sm120ForwardHost, '_dspark_fixed_tile_policy', None)
    if marker is None:
        if _flash_attn_fwd.compile_cache:
            raise ValueError('Policy must be installed BEFORE first FA4 compile')
        original = Sm120ForwardHost.select_config
        marker = {'tile': list(TILE), 'profile_id': config['profile_id'], 'select_calls': 0}

        def select_config(**options):
            result = original(**locked_options(options))
            if (result.tile_m, result.tile_n) != TILE:
                raise ValueError('Host did not honor fixed tile')
            marker['select_calls'] += 1
            return result

        Sm120ForwardHost.select_config = staticmethod(select_config)
        Sm120ForwardHost._dspark_fixed_tile_policy = marker
    elif marker['tile'] != list(TILE) or marker['profile_id'] != config['profile_id']:
        raise ValueError('Conflicting process-local numerical policy')
    return attestation_hook(config)
