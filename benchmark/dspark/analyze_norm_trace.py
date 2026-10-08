"""Read-only PID/launch-correlated attribution; never call this E2E savings."""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def analyze(folder):
    requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    requests = [row for row in requests if row['phase'] == 'diagnostic_profiled']
    identities = {row['meta_info']['id'] for row in requests}
    conn = sqlite3.connect('file:' + str(folder / 'trace.sqlite') + '?mode=ro', uri=True)
    strings = dict(conn.execute('select id,value from StringIds'))
    decode, norm = defaultdict(list), defaultdict(list)
    for start, end, tid, label, textid in conn.execute(
        'select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
        label = label or strings.get(textid, '')
        if label.startswith('DSpark._forward_decode '):
            item = json.loads(label.split(' ', 1)[1])
            if item['rid'] in identities:
                decode[tid].append((start, end, (item['rid'], item['step'])))
        elif label.startswith('DSparkResidualNorm '):
            norm[tid].append((start, end, json.loads(label.split(' ', 1)[1])))
    for groups in (decode, norm):
        for values in groups.values():
            values.sort(key=lambda row: row[0])
    dstarts = {tid: [row[0] for row in rows] for tid, rows in decode.items()}
    nstarts = {tid: [row[0] for row in rows] for tid, rows in norm.items()}
    def lookup(groups, starts, tid, timestamp):
        index = bisect.bisect_right(starts.get(tid, []), timestamp) - 1
        return groups[tid][index] if index >= 0 and timestamp < groups[tid][index][1] else None
    owners, norm_owners = {}, {}
    for start, tid, corr in conn.execute('select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        enclosing = lookup(decode, dstarts, tid, start)
        if enclosing:
            key = (tid & ~0xffffff, corr)
            owners[key] = enclosing[2]
            scope = lookup(norm, nstarts, tid, start)
            if scope:
                norm_owners[key] = scope[2]
    per_step, norm_step, norm_kernels = defaultdict(list), defaultdict(list), defaultdict(list)
    all_kernels = defaultdict(list)
    for start, end, pid, corr, nameid in conn.execute(
        'select start,end,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        key = (pid, corr)
        if key not in owners:
            continue
        name = strings[nameid]
        duration = (end - start) / 1000
        per_step[owners[key]].append((start, end, name, duration))
        all_kernels[name].append(duration)
        if key in norm_owners:
            norm_step[owners[key]].append(duration)
            norm_kernels[name].append(duration)
    conn.close()
    expected_steps = len(requests) * 40
    if len(per_step) != expected_steps or len(norm_step) != expected_steps:
        raise ValueError('Incomplete decode/norm launch coverage')
    norm_calls = Counter()
    for tid, rows in norm.items():
        for start, end, item in rows:
            if lookup(decode, dstarts, tid, start):
                norm_calls[json.dumps(item, sort_keys=True)] += 1
    def groups(values):
        return sorted([{'name': name, 'count': len(samples),
                        'gpu_us_per_actual_decode': sum(samples) / expected_steps}
                       for name, samples in values.items()],
                      key=lambda row: row['gpu_us_per_actual_decode'], reverse=True)
    return {
        'scope': 'profiled GPU attribution, NOT unprofiled E2E benefit',
        'requests': len(requests), 'actual_decode_steps': expected_steps,
        'norm_calls': dict(norm_calls),
        'kernels_per_decode': statistics.mean(map(len, per_step.values())),
        'norm_kernels_per_decode': statistics.mean(map(len, norm_step.values())),
        'norm_gpu_us_per_decode': statistics.mean(map(sum, norm_step.values())),
        'all_gpu_us_per_decode': statistics.mean(sum(row[3] for row in rows) for rows in per_step.values()),
        'gpu_span_us_per_decode': statistics.mean((max(row[1] for row in rows) - min(row[0] for row in rows)) / 1000
                                                 for rows in per_step.values()),
        'norm_kernel_groups': groups(norm_kernels), 'all_kernel_groups': groups(all_kernels),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.folder)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith('kernel_groups')}, indent=2))
