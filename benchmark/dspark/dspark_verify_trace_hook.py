"""Opt-in eager boundary observer, loaded only through --forward-hooks.

No inference monkeypatches. Copies one committed-anchor row per selected stage
and optionally bounded past KV, never replaces output or measures performance.
All graphs must be off.
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


def verify_query_lengths(*, layout_lengths, declared_lengths, fixed_tokens_per_request,
                         pool_count, total_rows):
    if layout_lengths is not None:
        lengths, source = list(layout_lengths), "live_ragged_layout"
    elif declared_lengths is not None:
        lengths, source = list(declared_lengths), "declared_extend_lengths"
    elif fixed_tokens_per_request and fixed_tokens_per_request > 0:
        lengths = [int(fixed_tokens_per_request)] * pool_count
        source = "fixed_verify_input_checked_against_runtime"
        if sum(lengths) != total_rows:
            raise ValueError("Fixed verify input does not match actual runtime rows")
    else:
        raise ValueError("No authoritative verify query lengths available")
    if len(lengths) != pool_count:
        raise ValueError("Verify query lengths do not match request count")
    request_rows(lengths, total_rows)
    return lengths, source


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


def trace_stage(module):
    """The registry uses positional-only post-forward hooks, not kwargs."""
    if hasattr(module, "self_attn") and hasattr(module.self_attn, "attn"):
        return f"layer{module.self_attn.attn.layer_id}.post", "layer"
    if hasattr(module, "layer_id") and hasattr(module, "tp_q_head_num"):
        return f"layer{module.layer_id}.attention", "attention"
    return "logits", "logits"


def validate_past_locations(locations, *, position, buffer_rows, current_locations):
    # Check on CPU BEFORE launching index_select; this observer must not turn
    # invalid metadata into an illegal GPU access or read future draft slots.
    if len(locations) != position or len(set(locations)) != len(locations):
        raise ValueError("Trace past-KV mapping has wrong length or duplicate locations")
    if any(loc < 0 or loc >= buffer_rows for loc in locations):
        raise ValueError("Trace past-KV location outside physical buffer")
    if set(locations).intersection(current_locations):
        raise ValueError("Trace past-KV mapping overlaps current verify writes")


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
    kv_layer = config.get("kv_layer")
    max_kv_prefix = int(config.get("max_kv_prefix", 4096))
    if focus_length < 1 or focus_position < focus_length or limit < 1:
        raise ValueError("Invalid trace focus or capture limit")
    if kv_layer is not None and (int(kv_layer) < 0 or focus_position > max_kv_prefix):
        raise ValueError("Past-KV trace exceeds explicit layer/prefix bound")
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
        stage, kind = trace_stage(module)
        fb = inputs[2] if kind == "layer" else inputs[3]
        if not hasattr(fb, "req_pool_indices") or fb.req_pool_indices is None:
            return None  # Not a real request ForwardBatch (e.g. warmup metadata).
        ids_tensor = inputs[0] if kind == "logits" else fb.input_ids
        mode = fb.forward_mode
        pools = cpu_list(fb.req_pool_indices)
        if mode.is_extend() and not mode.is_target_verify():
            if kind != "logits":
                return None  # Update identity once, after the whole prefill.
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
        ids, positions = cpu_list(ids_tensor), cpu_list(fb.positions)
        spec = getattr(fb, "spec_info", None)
        layout = getattr(spec, "ragged_verify_layout", None)
        if mode.is_target_verify():
            layout_lengths = None
            if layout is not None:
                layout_lengths = cpu_list(layout.verify_lens_cpu)
                if layout_lengths is None:
                    layout_lengths = cpu_list(layout.verify_lens)
                if sum(layout_lengths) != layout.total_verify_tokens or layout.graph_num_tokens != len(ids):
                    raise ValueError("Trace layout/live/padded runtime geometry mismatch")
            lengths, length_source = verify_query_lengths(
                layout_lengths=layout_lengths, declared_lengths=cpu_list(fb.extend_seq_lens_cpu),
                fixed_tokens_per_request=getattr(spec, "draft_token_num", None),
                pool_count=len(pools), total_rows=len(ids))
        else:
            lengths, length_source = [1] * len(pools), "decode_one_row_per_request"
        selected = anchor_rows(lengths=lengths, pools=pools, positions=positions, ids=ids,
                               pool_prompts=pool_prompts, focus_hash=focus_hash,
                               focus_position=focus_position, anchor_token=anchor_token)
        for slot, row, pool in selected:
            request_epoch = pool_prompts[pool][1]
            capture_key = (request_epoch, stage)
            if capture_key in captured_epochs:
                continue

            def copy_row(value):
                if not isinstance(value, torch.Tensor) or value.shape[0] != len(ids):
                    raise ValueError("Trace cannot assume row identity for this tensor")
                return value[row].detach().cpu().clone()

            extra = {}
            if kind == "logits":
                hidden, logits = inputs[1], output.next_token_logits
                tensors = {"final_hidden": copy_row(hidden), "raw_logits": copy_row(logits)}
                extra.update(hidden_dtype=str(hidden.dtype), logits_dtype=str(logits.dtype),
                             hidden_shape=list(hidden.shape), logits_shape=list(logits.shape),
                             raw_argmax=int(tensors["raw_logits"].argmax()),
                             raw_top3=[(int(i), float(v)) for v, i in zip(*torch.topk(tensors["raw_logits"], 3))])
            elif kind == "layer":
                # Input tensors can alias fused residual/norm buffers by the
                # time a POST hook runs. Only claim to observe returned output.
                hidden, residual = output
                tensors = {"post_hidden": copy_row(hidden)}
                if residual is not None:
                    tensors["post_residual"] = copy_row(residual)
            else:
                tensors = {name: copy_row(value) for name, value in
                           zip(("query", "current_key", "current_value", "attention_output"),
                               (*inputs[:3], output))}
                if kv_layer is not None and module.layer_id == int(kv_layer):
                    from sglang.srt.model_executor.forward_context import get_attn_backend
                    backend = get_attn_backend()
                    kv_pool = backend.token_to_kv_pool
                    keys = kv_pool.get_key_buffer(module.layer_id)
                    values = kv_pool.get_value_buffer(module.layer_id)
                    if (keys.ndim != 3 or values.ndim != 3 or keys.shape[0] != values.shape[0]
                            or getattr(kv_pool, "use_hnd", False)
                            or getattr(kv_pool, "store_dtype", keys.dtype) != keys.dtype):
                        raise ValueError("Trace only supports explicit unquantized NHD KV rows")
                    locations = cpu_list(backend.req_to_token_pool.req_to_token[pool, :focus_position])
                    current_locations = cpu_list(fb.out_cache_loc)
                    validate_past_locations(locations, position=focus_position,
                                            buffer_rows=keys.shape[0], current_locations=current_locations)
                    index = torch.tensor(locations, dtype=torch.long, device=keys.device)
                    tensors["past_key"] = keys.index_select(0, index).detach().cpu().clone()
                    tensors["past_value"] = values.index_select(0, index).detach().cpu().clone()
                    current_loc = current_locations[row]
                    if current_loc < 0 or current_loc >= keys.shape[0]:
                        raise ValueError("Trace current KV location outside physical buffer")
                    tensors["written_key"] = keys[current_loc].detach().cpu().clone()
                    tensors["written_value"] = values[current_loc].detach().cpu().clone()
                    selected_backend = (backend._select_backend(mode)
                                        if hasattr(backend, "_select_backend") else backend)
                    wrappers = getattr(getattr(selected_backend, "forward_metadata", None),
                                       "prefill_wrappers", [])
                    extra.update(past_kv_locations=locations, past_kv_position_end_exclusive=focus_position,
                                 current_kv_location=current_loc,
                                 backend_class=type(selected_backend).__name__,
                                 prefill_split_tile_size=getattr(selected_backend, "prefill_split_tile_size", None),
                                 # FlashInfer 0.6.17 exposes a TVM-FFI Array,
                                 # not a JSON-native list. Do not change it.
                                 wrapper_plan_info=[None if getattr(wrapper, "_plan_info", None) is None
                                                    else [int(value) for value in wrapper._plan_info]
                                                    for wrapper in wrappers])
            base = folder / f"pid{os.getpid()}-capture{count:03d}"
            with base.with_suffix(".pt").open("xb") as stream:
                torch.save(tensors, stream)
            metadata = {
                "timestamp_ns": time.time_ns(), "mode": str(mode), "pid": os.getpid(),
                "stage": stage, "module_class": type(module).__name__,
                "request_epoch": request_epoch, "prompt_sha256": focus_hash,
                "prompt_length": focus_length,
                "request_slot": slot, "pool_index": pool, "selected_row": row,
                "position": positions[row], "anchor_token": ids[row],
                "batch_size": len(pools), "query_lengths": lengths,
                "query_length_source": length_source,
                "live_query_total_from_lengths": sum(lengths), "tensor_row_count": len(ids),
                "request_pool_indices": pools, "input_ids": ids, "positions": positions,
                "seq_lens": cpu_list(fb.seq_lens), "seq_lens_cpu": cpu_list(fb.seq_lens_cpu),
                "out_cache_loc": cpu_list(fb.out_cache_loc),
                "live_seq_lens_cpu_before_verify": cpu_list(getattr(spec, "live_seq_lens_cpu", None)),
                "layout": {name: cpu_list(getattr(layout, name, None))
                           if name.endswith("_cpu") or name.endswith("_host") else getattr(layout, name, None)
                           for name in ("verify_lens_cpu", "total_verify_tokens", "graph_num_tokens",
                                        "qo_indptr_host", "kv_indptr_host", "kv_lens_host")},
                "captured_tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                     for name, value in tensors.items()},
                **extra,
                "tensor_sha256": hashlib.sha256(base.with_suffix(".pt").read_bytes()).hexdigest(),
                "scope": "read-only eager boundary observation; not performance evidence",
            }
            with base.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
            captured_epochs.add(capture_key)
            count += 1
        return None

    return hook
