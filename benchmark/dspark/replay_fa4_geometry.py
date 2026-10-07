"""Saved BF16 Q/K/V attention-only causality replay. Not a kernel benchmark.

Future rows are synthetic and causally masked from the first anchor. No model,
cache, acceptance oracle or inference sources are changed. Fixed tiles use the
existing private FA4 forward API solely to test a numerical hypothesis.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def difference(left, right):
    left, right = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Replay tensor shape/dtype mismatch")
    return {"bitwise_equal": bool(torch.equal(left, right)),
            "different_elements": int((left != right).sum()),
            "max_abs": float((left.float() - right.float()).abs().max())}


def load(folder):
    for path in sorted((folder / 'verify_trace').glob('*.json')):
        meta = json.loads(path.read_text())
        if meta['stage'] != 'layer0.attention':
            continue
        pt = path.with_suffix('.pt')
        if hashlib.sha256(pt.read_bytes()).hexdigest() != meta['tensor_sha256']:
            raise ValueError('Saved tensor SHA mismatch')
        return meta, torch.load(pt, map_location='cpu', weights_only=True)
    raise ValueError('Missing layer0 attention capture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import set_batch_invariant
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd
    from sglang.kernels.ops.attention.fa4_sm120.flash_fwd import FlashAttentionForwardSm120
    set_batch_invariant(True)
    loaded = {label: load(args.trace / label) for label in ('ordinary_target', 'dspark_full')}
    lm, lt = loaded['ordinary_target']
    rm, rt = loaded['dspark_full']
    for key in ('prompt_sha256', 'position', 'anchor_token'):
        if lm[key] != rm[key]:
            raise ValueError('Anchor identity mismatch')
    for name in ('query', 'current_key', 'current_value', 'past_key', 'past_value'):
        if not torch.equal(lt[name], rt[name]):
            raise ValueError('Replay requires identical observed attention inputs: ' + name)
    position, head_dim, qheads, kvheads = lm['position'], 128, 40, 8
    capacity = 131072  # Allocation bound only; visible KV lengths are checked below.
    q = lt['query'].reshape(qheads, head_dim).cuda()
    key = lt['current_key'].reshape(kvheads, head_dim).cuda()
    value = lt['current_value'].reshape(kvheads, head_dim).cuda()
    kcache = torch.zeros((capacity, 1, kvheads, head_dim), dtype=q.dtype, device='cuda')
    vcache = torch.zeros_like(kcache)
    kcache[:position, 0] = lt['past_key'].cuda()
    vcache[:position, 0] = lt['past_value'].cuda()
    kcache[position:position + 8, 0] = key
    vcache[position:position + 8, 0] = value
    report = {'scope': 'attention numerical causality only; not E2E or performance proof',
              'observed_decode_vs_verify': difference(lt['attention_output'], rt['attention_output']),
              'synthetic_future_rows': 'repeat committed Q/K/V, masked from first anchor',
              'profile_batch_invariant': True, 'num_splits': 1, 'page_size': 1, 'cases': {}}
    anchors = {}
    for length in (1, 8):
        queries = q.unsqueeze(0).repeat(length, 1, 1).contiguous()
        cu = torch.tensor([0, length], dtype=torch.int32, device='cuda')
        visible = torch.tensor([position + length], dtype=torch.int32, device='cuda')
        table = torch.arange(position + length, dtype=torch.int32, device='cuda').unsqueeze(0)
        expected = lt['attention_output'] if length == 1 else rt['attention_output']
        # Decode provides a host-visible K bound; target extend does not.
        auto_kw = {'max_seqlen_k': position + length} if length == 1 else {}
        def automatic():
            return flash_attn_with_kvcache(q=queries, k_cache=kcache, v_cache=vcache,
                page_table=table, cache_seqlens=visible, cu_seqlens_q=cu, max_seqlen_q=length,
                softmax_scale=head_dim ** -.5, causal=True, num_splits=1, **auto_kw)[0]
        with torch.inference_mode():
            auto = automatic()
            again = automatic()
        name = f'q{length}-automatic'
        anchors[name] = auto
        report['cases'][name] = {'repeat': difference(auto, again), 'observed': difference(auto, expected)}
        for tile in ((128, 64), (64, 64)):
            for kbound in (position + length, capacity):
                def fixed():
                    result = _flash_attn_fwd(q=queries, k=kcache, v=vcache, cu_seqlens_q=cu,
                        seqused_k=visible, page_table=table, max_seqlen_q=length, max_seqlen_k=kbound,
                        softmax_scale=head_dim ** -.5, causal=True, num_splits=1, tile_mn=tile)
                    return result[0][0]
                with torch.inference_mode():
                    first, second = fixed(), fixed()
                name = f'q{length}-tile{tile[0]}x{tile[1]}-kbound{kbound}'
                anchors[name] = first
                report['cases'][name] = {'repeat': difference(first, second),
                    'observed': difference(first, expected), 'vs_automatic': difference(first, auto)}
                print(name, report['cases'][name], flush=True)
    report['cross_query'] = {
        'automatic': difference(anchors['q1-automatic'], anchors['q8-automatic']),
        **{f'tile{m}x{n}-live_kbound': difference(anchors[f'q1-tile{m}x{n}-kbound{position+1}'],
                                                   anchors[f'q8-tile{m}x{n}-kbound{position+8}'])
           for m, n in ((128, 64), (64, 64))},
        **{f'tile{m}x{n}-capacity_kbound': difference(anchors[f'q1-tile{m}x{n}-kbound{capacity}'],
                                                       anchors[f'q8-tile{m}x{n}-kbound{capacity}'])
           for m, n in ((128, 64), (64, 64))}}
    print('CROSS_QUERY', report['cross_query'], flush=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    if not all(case['repeat']['bitwise_equal'] for case in report['cases'].values()):
        raise ValueError('Replay not repeat-stable')
    report_match = all(report['cases'][f'q{length}-automatic']['observed']['bitwise_equal'] for length in (1,8))
    print('automatic_matches_both_observed', report_match, flush=True)


if __name__ == '__main__':
    main()
