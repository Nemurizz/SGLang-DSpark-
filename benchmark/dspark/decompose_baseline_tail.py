"""Within-trace tail decomposition, NOT unprofiled performance attribution.

Kernel-uncovered intervals may include DMA, other GPU activity, host dispatch or
synchronization. They are NOT an automatically graph-removable latency budget.
"""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def union_ns(intervals):
    total, end = 0, None
    for a,b in sorted(intervals):
        if end is None or a > end:
            total += b-a
        elif b > end:
            total += b-end
        end = max(end or b, b)
    return total


def correlation(left, right):
    a, b = statistics.mean(left), statistics.mean(right)
    numerator = sum((x-a)*(y-b) for x,y in zip(left,right))
    divisor = (sum((x-a)**2 for x in left)*sum((y-b)**2 for y in right))**.5
    return numerator/divisor if divisor else None


def analyze(folder):
    base = json.loads((folder/'analysis.json').read_text())
    identities = set(base['gpu_execution_by_request'])
    c = sqlite3.connect('file:' + str(folder/'trace.sqlite') + '?mode=ro', uri=True)
    strings = dict(c.execute('select id,value from StringIds'))
    ranges = defaultdict(list)
    for a,b,tid,label,textid in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
        label = label or strings.get(textid,'')
        if label.startswith('DSpark._forward_decode '):
            details = json.loads(label.split(' ',1)[1])
            if details['rid'] in identities:
                ranges[tid].append({'start':a,'end':b,**details})
    for values in ranges.values():
        values.sort(key=lambda x:x['start'])
    starts = {tid:[x['start'] for x in values] for tid,values in ranges.items()}
    launches, apis = {}, defaultdict(lambda: defaultdict(list))
    for a,b,tid,corr,nameid in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        if tid not in ranges:
            continue
        i = bisect.bisect_right(starts[tid],a)-1
        if i >= 0 and a < ranges[tid][i]['end']:
            span = ranges[tid][i]
            launches[(tid & ~0xffffff,corr)] = (span['rid'],span['step'])
            apis[span['rid']][strings[nameid]].append((b-a)/1e6)
    kernels = defaultdict(list)
    for a,b,pid,corr in c.execute('select start,end,globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        key = launches.get((pid,corr))
        if key is not None:
            kernels[key].append((a,b))
    c.close()
    host = [json.loads(s) for s in (folder/'worker_trace.jsonl').read_text().splitlines()]
    rows = []
    for rid,original in base['gpu_execution_by_request'].items():
        values = [kernels[(rid,i)] for i in range(1,41)]
        assert all(len(v)==1636 for v in values)
        span = statistics.mean((max(b for a,b in v)-min(a for a,b in v))/1e6 for v in values)
        union = statistics.mean(union_ns(v)/1e6 for v in values)
        item = {'rid':rid, **original, 'mean_gpu_span_ms_per_decode':span,
            'mean_gpu_kernel_union_ms_per_decode':union,
            'mean_kernel_uncovered_span_ms_per_decode':span-union,
            'runtime_apis_ms_per_decode': {name:sum(v)/40 for name,v in apis[rid].items()},
            'host_methods_ms_per_decode':{}}
        methods = defaultdict(list)
        for record in host:
            if record.get('rid') == rid:
                methods[record['method']].append(record)
        for name,records in methods.items():
            item['host_methods_ms_per_decode'][name] = {
                'calls':len(records), 'wall':sum(r['wall_ns'] for r in records)/40e6,
                'thread_cpu':sum(r['thread_cpu_ns'] for r in records)/40e6}
        rows.append(item)
    rows.sort(key=lambda r:r['ordinal'])
    tpot = [r['tpot_ms'] for r in rows]
    names = ('mean_kernel_sum_ms_per_actual_decode','mean_gemm_sum_ms_per_actual_decode',
             'mean_gpu_span_ms_per_decode','mean_kernel_uncovered_span_ms_per_decode')
    result = {'scope':'within-profile full-only association; cannot attribute unprofiled h41 spikes',
        'notes':'Host spans overlap; API durations can nest. Kernel-uncovered spans are NOT graph-removable GPU idle.',
        'request_count':len(rows),'requests':rows,
        'pearson_with_profiled_tpot':{name:correlation(tpot,[r[name] for r in rows]) for name in names}}
    ordered = sorted(rows,key=lambda r:r['tpot_ms'])
    for group,selected in [('fastest_quartile',ordered[:8]),('slowest_quartile',ordered[-8:])]:
        result[group] = {name:statistics.mean(r[name] for r in selected) for name in ('tpot_ms',)+names}
        result[group]['runtime_apis_ms_per_decode'] = {name:statistics.mean(
            r['runtime_apis_ms_per_decode'].get(name,0) for r in selected) for name in
            sorted({name for r in selected for name in r['runtime_apis_ms_per_decode']})}
        result[group]['host_methods_ms_per_decode'] = {name:{metric:statistics.mean(
            r['host_methods_ms_per_decode'].get(name,{}).get(metric,0) for r in selected)
            for metric in ('wall','thread_cpu')} for name in
            sorted({name for r in selected for name in r['host_methods_ms_per_decode']})}
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    cli=p.parse_args()
    result=analyze(cli.folder)
    with cli.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2))
