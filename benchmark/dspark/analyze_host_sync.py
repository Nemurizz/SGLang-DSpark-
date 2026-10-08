"""Read-only attribution of existing trace APIs to innermost DSpark segment."""
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys


def run(path):
    db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    strings = dict(db.execute('select id,value from StringIds'))
    ranges = defaultdict(list)
    for a,b,tid,label,sid in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
        label = label or strings.get(sid,'')
        if label.startswith('DSpark.'):
            method, info = label.split(' ',1)
            if json.loads(info).get('step',0) > 0:
                ranges[tid].append((a,b,method))
    for rows in ranges.values():
        rows.sort()
    starts = {tid:[r[0] for r in rows] for tid,rows in ranges.items()}
    values = defaultdict(list)
    for a,b,tid,sid in db.execute('select start,end,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        name = strings[sid]
        if name not in ('cudaMemcpyAsync', 'cudaStreamSynchronize', 'cudaGraphLaunch') or tid not in starts:
            continue
        i = bisect.bisect_right(starts[tid],a)-1
        containing = [r for r in ranges[tid][max(0,i-8):i+1] if r[0]<=a<r[1]]
        if containing:
            inner = min(containing,key=lambda r:r[1]-r[0])
            values[(inner[2],name)].append((b-a)/1e6)
    result=[]
    for (method,api),rows in values.items():
        rows.sort()
        result.append({'method':method,'api':api,'count':len(rows),'sum_ms':sum(rows),
            'p50_ms':rows[len(rows)//2], 'p95_ms':rows[int((len(rows)-1)*.95)]})
    db.close()
    print(json.dumps(sorted(result,key=lambda r:r['sum_ms'],reverse=True),indent=2))


if __name__ == '__main__':
    run(Path(sys.argv[1]))
