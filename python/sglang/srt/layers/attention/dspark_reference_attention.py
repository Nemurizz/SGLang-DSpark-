"""Default-off eager correctness candidate for the recorded Qwen3/FA2 geometry.

Preserve FlashInfer 0.6.17's ORIGINAL B=1 KV partition per request. This is not
a performance claim or a general attention backend, and never runs in graphs.
"""


def canonical_groups(query_lengths, kv_lengths, sm_count):
    if (not query_lengths or len(query_lengths) != len(kv_lengths)
            or any(not 1 <= value <= 8 for value in query_lengths)
            or any(value < query for value, query in zip(kv_lengths, query_lengths))):
        raise ValueError("Reference partition requires live positive Qwen3 verify lengths <= 8")
    if sm_count < 4:
        raise ValueError("Reference partition requires at least four SMs")
    # Checked installed FA2 scheduler: GQA=5, head_dim=128, q<=8 => packed
    # q<=40 => CTA_TILE_Q=16 for q<=3, otherwise 64; one B=1 q tile. Capacity
    # is 2*SM/8 KV heads. B=1 binary search has the closed form below, in pages
    # (page size MUST be 1). No batch size or stale/padded cap enters it.
    capacity = (2 * sm_count) // 8
    keys = [(max(128, (length + capacity - 1) // capacity), 16 if query <= 3 else 64)
            for query, length in zip(query_lengths, kv_lengths)]
    groups, start = [], 0
    for end in range(1, len(keys) + 1):
        if end == len(keys) or keys[end] != keys[start]:
            groups.append((start, end, *keys[start]))
            start = end
    return groups


class SerialReferencePlans:
    def __init__(self, backend):
        import flashinfer
        import torch
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend

        updater = backend.indices_updater_prefill
        if (flashinfer.__version__ != "0.6.17" or backend.prefill_backend != "fa2"
                or backend.page_size != 1 or backend.num_wrappers != 1
                or backend.prefill_uses_dequant_workspace or backend.is_multimodal
                or backend.enable_mis or backend.use_sliding_window_kv_pool
                or updater.num_qo_heads != 40 or updater.num_kv_heads != 8
                or updater.head_dim != 128 or updater.q_data_type != torch.bfloat16
                or updater.data_type != torch.bfloat16
                or not all(check_cuda_graph_backend(phase, Backend.DISABLED)
                           for phase in (Phase.PREFILL, Phase.DECODE))):
            raise ValueError("Serial-reference verify plans require checked FA2 0.6.17, BF16 Qwen3 NHD, TP1 and all graphs off")
        self.backend = backend
        properties = torch.cuda.get_device_properties(backend.workspace_buffer.device)
        if properties.major < 8:
            raise ValueError("Serial-reference CTA rule requires Ampere or newer")
        self.sm_count = properties.multi_processor_count
        self.wrappers = []

    def plan(self, original, forward_batch):
        import flashinfer

        if original.is_cuda_graph_enabled or original._custom_mask_buf is not None:
            raise ValueError("Serial-reference verify plans cannot process graphs or custom masks")
        qo = original._qo_indptr_buf.detach().cpu().tolist()
        kv = original._paged_kv_indptr_buf.detach().cpu().tolist()
        if qo[0] != 0 or kv[0] != 0 or len(qo) != len(kv):
            raise ValueError("Reference plan indptr geometry mismatch")
        queries = [b - a for a, b in zip(qo, qo[1:])]
        kv_lengths = [b - a for a, b in zip(kv, kv[1:])]
        if (qo[-1] != forward_batch.input_ids.numel()
                or len(queries) != forward_batch.req_pool_indices.numel()
                or kv[-1] > original._paged_kv_indices_buf.numel()):
            raise ValueError("Reference plan must use actual live, unpadded runtime rows")
        groups = canonical_groups(queries, kv_lengths, self.sm_count)
        if len(queries) == 1:
            return [(0, qo[-1], original)]  # Preserve original serial execution exactly.
        output_groups = []
        for index, (start, end, chunk, cta_tile_q) in enumerate(groups):
            if index == len(self.wrappers):
                self.wrappers.append(flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.backend.workspace_buffer, "NHD", backend="fa2"))
            wrapper = self.wrappers[index]
            wrapper.plan(original._qo_indptr_buf[start:end + 1] - qo[start],
                         original._paged_kv_indptr_buf[start:end + 1] - kv[start],
                         original._paged_kv_indices_buf[kv[start]:kv[end]],
                         original._paged_kv_last_page_len_buf[start:end],
                         40, 8, 128, 1, q_data_type=self.backend.indices_updater_prefill.q_data_type,
                         kv_data_type=self.backend.indices_updater_prefill.data_type,
                         fixed_split_size=chunk)
            info = [int(value) for value in wrapper._plan_info]
            if len(info) != 15 or info[3] != cta_tile_q or info[1] != qo[end] - qo[start] or info[13]:
                raise ValueError("Reference partition actual FA2 plan differs from checked geometry")
            output_groups.append((qo[start], qo[end], wrapper))
        return output_groups


def forward_groups(groups, q, kv_cache, **kwargs):
    import torch
    outputs = [wrapper.forward(q[start:end], kv_cache, **kwargs) for start, end, wrapper in groups]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
