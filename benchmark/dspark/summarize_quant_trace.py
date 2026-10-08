"""Read-only bounded CUDA trace, with collector caveats retained."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def summarize(root):
    audit = json.loads((root / 'collector_audit.json').read_text())
    counts = json.loads((root / 'trace_counts.json').read_text())
    db = sqlite3.connect('file:' + str(root / 'trace.sqlite') + '?mode=ro', uri=True)
    spans = db.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
    busy, previous_end = 0, 0
    for start, end in spans:
        busy += max(0, end - max(start, previous_end))
        previous_end = max(end, previous_end)
    elapsed = spans[-1][1] - spans[0][0]
    kernels = list(csv.DictReader((root / 'stats_cuda_gpu_kern_sum.csv').open()))
    groups = {}
    for row in kernels:
        name = row['Name']
        category = ('native_cutlass_fp4' if 'MainloopSm120' in name else
                    'fa4_attention' if 'FlashAttentionForwardSm120' in name else
                    'fp4_activation_quantization' if 'quantize_with_block_size' in name else
                    'bf16_head_persistent' if name == 'matmul_kernel_persistent' else 'other')
        group = groups.setdefault(category, {'summed_kernel_ms': 0, 'launches': 0})
        group['summed_kernel_ms'] += int(row['Total Time (ns)']) / 1e6
        group['launches'] += int(row['Instances'])
    cpu = list(csv.DictReader((root / 'stats_nvtx_sum.csv').open()))
    ranges = [{'label': row['Range'], 'count': int(row['Instances']),
               'summed_cpu_wall_ms': int(row['Total Time (ns)']) / 1e6,
               'median_cpu_wall_us': float(row['Med (ns)']) / 1e3}
              for row in cpu if 'fp4_' in row['Range']]
    apis = list(csv.DictReader((root / 'stats_cuda_api_sum.csv').open()))
    requests = [json.loads(line) for line in (root / 'requests.jsonl').read_text().splitlines()]
    result = {'scope': 'one prefill + captured decode forwards, CUDA/NVTX instrumented, NOT steady A/B',
        'raw_collector_audit': audit, 'recorded_operator_counts': counts,
        'kernel_count': len(spans), 'kernel_span_ms': elapsed / 1e6,
        'observed_gpu_busy_union_ms': busy / 1e6, 'busy_union_fraction_of_kernel_span': busy / elapsed,
        'groups': groups, 'cpu_operator_ranges': ranges,
        'cuda_allocation_api_calls_observed': [row for row in apis if any(x in row['Name'] for x in ('Malloc', 'Free'))],
        'timing_rows_diagnostic_only': [{key: r[key] for key in ('phase','ttft_ms','tpot_ms','e2e_ms')} for r in requests],
        'workspace_conclusion': 'No device malloc/free API does not rule out PyTorch caching-allocator churn; no causal attribution yet.',
        'coverage_conclusion': 'Runtime-correlated kernel counts and explicit operator counts checked; retain not-all-events warnings, no blanket completeness claim.'}
    db.close()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    value = summarize(args.root)
    with (args.root / 'analysis.json').open('x') as out:
        json.dump(value, out, indent=2)
    print(json.dumps({k:v for k,v in value.items() if k != 'raw_collector_audit'}, indent=2), flush=True)
