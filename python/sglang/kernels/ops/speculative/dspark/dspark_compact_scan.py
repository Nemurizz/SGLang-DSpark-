from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Optional
import torch
from sglang.srt.environ import envs
logger=logging.getLogger(__name__)
_EXT=None
_LOAD_ERROR: Optional[BaseException]=None
_LOCK=threading.Lock()
def enabled() -> bool:
    return bool(envs.SGLANG_DSPARK_COMPACT_SCAN_CUDA.get()) and torch.cuda.is_available()
def _load_extension():
    global _EXT,_LOAD_ERROR
    if _EXT is not None or _LOAD_ERROR is not None: return _EXT
    with _LOCK:
        if _EXT is not None or _LOAD_ERROR is not None: return _EXT
        try:
            from torch.utils.cpp_extension import load
            _EXT=load(name="dspark_compact_scan_ext",sources=[str(Path(__file__).with_suffix(".cu"))],extra_cuda_cflags=["-O3"],verbose=False)
            logger.info("Loaded DSpark hand-written compact/scan CUDA kernel")
        except BaseException as exc:
            _LOAD_ERROR=exc
            logger.warning("DSpark compact/scan CUDA kernel unavailable; using Triton fallback: %s",exc)
    return _EXT
def fused_compact_scan(*,verify_lens: torch.Tensor,draft_block_ids: torch.Tensor,draft_tokens: torch.Tensor,padded_total: int):
    if not enabled(): return None
    ext=_load_extension()
    if ext is None: return None
    return ext.compact_scan(verify_lens.to(dtype=torch.int32).contiguous(),draft_block_ids.to(dtype=torch.int64).contiguous(),draft_tokens.to(dtype=torch.int64).contiguous(),int(padded_total))
