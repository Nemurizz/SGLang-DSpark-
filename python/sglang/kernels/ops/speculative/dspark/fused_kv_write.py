from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kv_norm_rope_write_kernel(
    kv_ptr,
    meta_ptr,
    knw_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    commit_lens_ptr,
    locs_row_width,
    KV: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    l = tl.program_id(1).to(tl.int64)
    if HAS_COMMIT_LENS:
        row_b = t // locs_row_width
        col = t - row_b * locs_row_width
        num_commit = tl.load(commit_lens_ptr + row_b).to(tl.int64)
        if col >= num_commit:
            return
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)
    knw1 = tl.load(knw_ptr + l * D + half_ar).to(tl.float32)
    knw2 = tl.load(knw_ptr + l * D + HALF + half_ar).to(tl.float32)

    k_buf = tl.load(meta_ptr + l * 4 + 0).to(tl.pointer_type(tl.bfloat16))
    v_buf = tl.load(meta_ptr + l * 4 + 1).to(tl.pointer_type(tl.bfloat16))
    ks0 = tl.load(meta_ptr + l * 4 + 2)
    vs0 = tl.load(meta_ptr + l * 4 + 3)

    row = kv_ptr + t * (L * 2 * KV) + l * (2 * KV)
    for h in tl.static_range(NH):
        k = tl.load(row + h * D + d_ar).to(tl.float32)
        ms = tl.sum(k * k, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        k1 = tl.load(row + h * D + half_ar).to(tl.float32) * inv * knw1
        k2 = tl.load(row + h * D + HALF + half_ar).to(tl.float32) * inv * knw2
        k1 = k1.to(tl.bfloat16).to(tl.float32)
        k2 = k2.to(tl.bfloat16).to(tl.float32)
        o1 = k1 * cos - k2 * sin
        o2 = k2 * cos + k1 * sin
        tl.store(k_buf + loc * ks0 + h * D + half_ar, o1.to(tl.bfloat16))
        tl.store(k_buf + loc * ks0 + h * D + HALF + half_ar, o2.to(tl.bfloat16))

        v = tl.load(row + KV + h * D + d_ar)
        tl.store(v_buf + loc * vs0 + h * D + d_ar, v)


@triton.jit
def _fused_kv_gemm_norm_rope_write_kernel(
    x_ptr,
    w_ptr,
    meta_ptr,
    knw_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    commit_lens_ptr,
    locs_row_width,
    K: tl.constexpr,
    KV: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one K/V head and write it directly to the cache pool."""
    t = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    if HAS_COMMIT_LENS:
        row_b = t // locs_row_width
        col = t - row_b * locs_row_width
        num_commit = tl.load(commit_lens_ptr + row_b).to(tl.int64)
        if col >= num_commit:
            return

    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)

    layer_tiles = 2 * NH
    l = tile // layer_tiles
    kind = tile - l * layer_tiles
    is_k = kind < NH
    h = kind % NH
    o0 = tile * D
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    k_ar = tl.arange(0, BLOCK_K)
    acc_lo = tl.zeros((HALF,), dtype=tl.float32)
    acc_hi = tl.zeros((HALF,), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + t * K + k0 + k_ar,
            mask=k_ar < K - k0,
            other=0.0,
        ).to(tl.float32)
        w_lo = tl.load(
            w_ptr + (o0 + half_ar)[:, None] * K + k0 + k_ar[None, :],
            mask=k_ar[None, :] < K - k0,
            other=0.0,
        ).to(tl.float32)
        w_hi = tl.load(
            w_ptr + (o0 + HALF + half_ar)[:, None] * K + k0 + k_ar[None, :],
            mask=k_ar[None, :] < K - k0,
            other=0.0,
        ).to(tl.float32)
        acc_lo += tl.sum(w_lo * x[None, :], axis=1)
        acc_hi += tl.sum(w_hi * x[None, :], axis=1)

    raw_lo = acc_lo.to(tl.bfloat16)
    raw_hi = acc_hi.to(tl.bfloat16)
    k_lo = raw_lo.to(tl.float32)
    k_hi = raw_hi.to(tl.float32)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)
    knw1 = tl.load(knw_ptr + l * D + half_ar).to(tl.float32)
    knw2 = tl.load(knw_ptr + l * D + HALF + half_ar).to(tl.float32)
    ms = (tl.sum(k_lo * k_lo, 0) + tl.sum(k_hi * k_hi, 0)) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    k1 = k_lo * inv * knw1
    k2 = k_hi * inv * knw2
    # Match the existing path's BF16 round-trip before applying RoPE.
    k1 = k1.to(tl.bfloat16).to(tl.float32)
    k2 = k2.to(tl.bfloat16).to(tl.float32)
    o1 = k1 * cos - k2 * sin
    o2 = k2 * cos + k1 * sin

    k_buf = tl.load(meta_ptr + l * 4 + 0).to(tl.pointer_type(tl.bfloat16))
    v_buf = tl.load(meta_ptr + l * 4 + 1).to(tl.pointer_type(tl.bfloat16))
    ks0 = tl.load(meta_ptr + l * 4 + 2)
    vs0 = tl.load(meta_ptr + l * 4 + 3)
    k_base = k_buf + loc * ks0 + h * D
    v_base = v_buf + loc * vs0 + h * D
    tl.store(k_base + half_ar, o1.to(tl.bfloat16), mask=is_k)
    tl.store(k_base + HALF + half_ar, o2.to(tl.bfloat16), mask=is_k)
    tl.store(v_base + half_ar, raw_lo, mask=~is_k)
    tl.store(v_base + HALF + half_ar, raw_hi, mask=~is_k)


def fused_kv_gemm_norm_rope_write(
    ctx_hidden: torch.Tensor,
    w_all: torch.Tensor,
    meta: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locs: torch.Tensor,
    num_layers: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    commit_lens: Optional[torch.Tensor] = None,
    locs_row_width: Optional[int] = None,
) -> None:
    """Hand-written stacked KV GEMM fused with norm/RoPE/cache writes."""
    T, K = ctx_hidden.shape
    if T == 0:
        return
    if head_dim != 128 or kv_size % head_dim != 0:
        raise ValueError(
            "manual fused KV kernel requires head_dim=128 and kv_size divisible "
            f"by head_dim, got head_dim={head_dim}, kv_size={kv_size}"
        )
    has_commit_lens = commit_lens is not None
    if has_commit_lens != (locs_row_width is not None):
        raise ValueError("commit_lens and locs_row_width must be passed together")
    if has_commit_lens:
        if commit_lens.numel() * locs_row_width != locs.numel():
            raise ValueError("flattened locs does not match commit_lens")
        commit_lens_arg = commit_lens.contiguous()
    else:
        locs_row_width = 1
        commit_lens_arg = locs
    ctx_hidden = ctx_hidden.contiguous()
    w_all = w_all.contiguous()
    grid = (T, num_layers * 2 * (kv_size // head_dim))
    _fused_kv_gemm_norm_rope_write_kernel[grid](
        ctx_hidden,
        w_all,
        meta,
        k_norm_weights,
        cos_sin_cache,
        positions.to(torch.int64).contiguous(),
        locs.to(torch.int64).contiguous(),
        commit_lens_arg,
        locs_row_width,
        K=K,
        KV=kv_size,
        D=head_dim,
        NH=kv_size // head_dim,
        L=num_layers,
        EPS=eps,
        HAS_COMMIT_LENS=has_commit_lens,
        BLOCK_K=128,
        num_warps=4,
    )
def fused_kv_norm_rope_write(
    kv: torch.Tensor,
    meta: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locs: torch.Tensor,
    num_layers: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    commit_lens: Optional[torch.Tensor] = None,
    locs_row_width: Optional[int] = None,
) -> None:
    """Write per-layer normed+roped K and raw V rows into the KV pools.

    Rows with loc < 0 are skipped. When commit_lens is given, locs is the
    flattened [bs, locs_row_width] verify window and only the first
    commit_lens[b] columns of each row are written — the in-kernel
    replacement for masking the tail columns to -1 on the host.
    """
    T = kv.shape[0]
    if T == 0:
        return
    has_commit_lens = commit_lens is not None
    if has_commit_lens != (locs_row_width is not None):
        raise ValueError(
            "commit_lens and locs_row_width must be passed together, got "
            f"commit_lens={'set' if has_commit_lens else None}, "
            f"locs_row_width={locs_row_width}."
        )
    if has_commit_lens:
        if commit_lens.numel() * locs_row_width != locs.numel():
            raise ValueError(
                f"locs must be a flattened [{commit_lens.numel()}, "
                f"{locs_row_width}] window, got numel={locs.numel()}."
            )
        commit_lens_arg = commit_lens.contiguous()
    else:
        locs_row_width = 1
        commit_lens_arg = locs
    grid = (T, num_layers)
    _fused_kv_norm_rope_write_kernel[grid](
        kv,
        meta,
        k_norm_weights,
        cos_sin_cache,
        positions.to(torch.int64).contiguous(),
        locs.to(torch.int64).contiguous(),
        commit_lens_arg,
        locs_row_width,
        KV=kv_size,
        D=head_dim,
        NH=kv_size // head_dim,
        L=num_layers,
        EPS=eps,
        HAS_COMMIT_LENS=has_commit_lens,
    )
