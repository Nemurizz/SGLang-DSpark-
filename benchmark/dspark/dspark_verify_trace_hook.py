"""Opt-in eager boundary observer, loaded only through --forward-hooks.

No inference monkeypatches. Copies one committed-anchor row, never returns a
replacement output, and never measures performance. All graphs must be off.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


def prompt_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def request_rows(lengths, total_rows):
    if not lengths or any(int(n) <= 0 for n in lengths):
        raise ValueError("Trace requires positive query lengths")
    if sum(lengths) > total_rows:
        raise ValueError("Trace query lengths exceed tensor rows")
    starts, offset = [], 0
    for length in lengths:
        starts.append(offset)
        offset += length
    return starts


def anchor_rows(*, lengths, pools, positions, ids, pool_prompts, focus_hash,
                focus_position, anchor_token):
    if len(pools) != len(lengths) or len(positions) != len(ids):
        raise ValueError("Trace request/row geometry mismatch")
    starts = request_rows(lengths, len(ids))
    selected = []
    for slot, (pool, start, length) in enumerate(zip(pools, starts, lengths)):
        # Only anchor rows: exclude speculative future rows not yet committed.
        if (pool_prompts.get(pool, (None,))[0] == focus_hash
                and positions[start] == focus_position and ids[start] == anchor_token):
            if positions[start:start + length] != list(range(focus_position, focus_position + length)):
                raise ValueError("Trace query positions are not contiguous")
            selected.append((slot, start, pool))
    return selected


def make_hook(config):
    import torch
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend

    if not all(check_cuda_graph_backend(phase, Backend.DISABLED) for phase in (Phase.PREFILL, Phase.DECODE)):
        raise ValueError("Verify boundary trace requires all CUDA graphs disabled")
    folder = Path(config["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    focus_hash = config["prompt_sha256"]
    focus_length = int(config["prompt_length"])
    focus_position = int(config["position"])
    anchor_token = int(config["anchor_token"])
    limit = int(config.get("max_captures", 8))
    if focus_length < 1 or focus_position < focus_length or limit < 1:
        raise ValueError("Invalid trace focus or capture limit")
    pool_prompts, captured_epochs = {}, set()
    epoch = count = 0

    def cpu_list(value):
        if value is None:
            return None
        return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)

    def hook(module, inputs, output):
        nonlocal epoch, count
        if count >= limit or len(inputs) < 4:
            return None
        ids_tensor, hidden, _, fb = inputs[:4]
        if not hasattr(fb, "req_pool_indices") or fb.req_pool_indices is None:
            return None  # Not a real request ForwardBatch (e.g. warmup metadata).
        mode = fb.forward_mode
        pools = cpu_list(fb.req_pool_indices)
        if mode.is_extend() and not mode.is_target_verify():
            lengths = cpu_list(fb.extend_seq_lens_cpu)
            prefixes = cpu_list(fb.extend_prefix_lens_cpu)
            ids = cpu_list(ids_tensor)
            starts = request_rows(lengths, len(ids))
            if len(pools) != len(lengths) or prefixes is None or len(prefixes) != len(lengths):
                raise ValueError("Trace prefill geometry mismatch")
            for pool, start, length, prefix in zip(pools, starts, lengths, prefixes):
                epoch += 1
                pool_prompts.pop(pool, None)  # Never retain a stale reused-pool identity.
                if prefix == 0 and length == focus_length:
                    pool_prompts[pool] = (prompt_hash(ids[start:start + length]), epoch)
            return None
        if not (mode.is_target_verify() or mode.is_decode()):
            return None
        if not any(pool_prompts.get(pool, (None,))[0] == focus_hash for pool in pools):
            return None
        lengths = cpu_list(fb.extend_seq_lens_cpu) if mode.is_target_verify() else [1] * len(pools)
        ids, positions = cpu_list(ids_tensor), cpu_list(fb.positions)
        selected = anchor_rows(lengths=lengths, pools=pools, positions=positions, ids=ids,
                               pool_prompts=pool_prompts, focus_hash=focus_hash,
                               focus_position=focus_position, anchor_token=anchor_token)
        for slot, row, pool in selected:
            request_epoch = pool_prompts[pool][1]
            if request_epoch in captured_epochs:
                continue
            logits = output.next_token_logits
            if logits is None or hidden.shape[0] != len(ids) or logits.shape[0] != len(ids):
                raise ValueError("Trace cannot assume hidden/logit row identity for this forward")
            tensors = {"final_hidden": hidden[row].detach().cpu().clone(),
                       "raw_logits": logits[row].detach().cpu().clone()}
            base = folder / f"pid{os.getpid()}-capture{count:03d}"
            with base.with_suffix(".pt").open("xb") as stream:
                torch.save(tensors, stream)
            layout = getattr(getattr(fb, "spec_info", None), "ragged_verify_layout", None)
            metadata = {
                "timestamp_ns": time.time_ns(), "mode": str(mode), "pid": os.getpid(),
                "request_epoch": request_epoch, "prompt_sha256": focus_hash,
                "request_slot": slot, "pool_index": pool, "selected_row": row,
                "position": positions[row], "anchor_token": ids[row],
                "batch_size": len(pools), "query_lengths": lengths,
                "live_query_total_from_lengths": sum(lengths), "tensor_row_count": len(ids),
                "request_pool_indices": pools, "input_ids": ids, "positions": positions,
                "seq_lens": cpu_list(fb.seq_lens), "seq_lens_cpu": cpu_list(fb.seq_lens_cpu),
                "out_cache_loc": cpu_list(fb.out_cache_loc),
                "layout": {name: cpu_list(getattr(layout, name, None))
                           if name == "verify_lens_cpu" else getattr(layout, name, None)
                           for name in ("verify_lens_cpu", "total_verify_tokens", "graph_num_tokens")},
                "hidden_dtype": str(hidden.dtype), "logits_dtype": str(logits.dtype),
                "hidden_shape": list(hidden.shape), "logits_shape": list(logits.shape),
                "raw_argmax": int(tensors["raw_logits"].argmax()),
                "raw_top3": [(int(i), float(v)) for v, i in zip(*torch.topk(tensors["raw_logits"], 3))],
                "tensor_sha256": hashlib.sha256(base.with_suffix(".pt").read_bytes()).hexdigest(),
                "scope": "read-only eager boundary observation; not performance evidence",
            }
            with base.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
            captured_epochs.add(request_epoch)
            count += 1
        return None

    return hook
