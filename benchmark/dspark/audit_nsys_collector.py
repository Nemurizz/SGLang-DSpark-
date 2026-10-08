"""Read-only SQLite event coverage audit; retain ALL collector diagnostics."""
import argparse
import json
from pathlib import Path
import sqlite3


def audit(path, smoke=False):
    c = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    diagnostics = [{'severity': s, 'text': t} for s, t in
                   c.execute('select severity,text from DIAGNOSTIC_EVENT')]
    kernels = c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    # Materialize distinct launch keys so an unindexed export does not execute
    # a quadratic correlated subquery on multi-million-event traces.
    runtime_correlated = c.execute('''select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k
        join (select distinct correlationId, (globalTid & ~16777215) as globalPid
              from CUPTI_ACTIVITY_KIND_RUNTIME) r
        on r.correlationId=k.correlationId and r.globalPid=k.globalPid''').fetchone()[0]
    result = {'sqlite': str(path), 'kernel_count': kernels,
        'runtime_correlated_kernels': runtime_correlated, 'diagnostics': diagnostics,
        'loaded_cupti': [d['text'] for d in diagnostics if 'Loaded CUPTI library:' in d['text']],
        'trace_method': [d['text'] for d in diagnostics if 'trace was collected' in d['text']
                         or 'Hardware tracing used' in d['text']],
        'unsupported_driver_warning': any('not supported by this build' in d['text'] for d in diagnostics),
        'potential_missing_events_messages': [d for d in diagnostics if
            any(word in d['text'].lower() for word in ('not all', 'dropped', 'lost', 'missing'))],
        'scope': 'bounded capture coverage, NOT proof that events outside capture were collected'}
    if smoke:
        nvtx = c.execute('''select count(*) from NVTX_EVENTS n left join StringIds s on s.id=n.textId
            where coalesce(n.text,s.value) like 'nsys_smoke_mm_%' and n.end is not null''').fetchone()[0]
        result.update(expected_smoke_kernels=16, expected_smoke_nvtx=16, actual_smoke_nvtx=nvtx)
        result['bounded_smoke_coverage_passed'] = (kernels == runtime_correlated == nvtx == 16
            and not result['unsupported_driver_warning']
            and any('libcupti.so.13.2' in entry for entry in result['loaded_cupti']))
        if not result['bounded_smoke_coverage_passed']:
            c.close()
            raise ValueError('Smoke coverage/compatibility failed: ' + json.dumps(result))
    c.close()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    result = audit(cli.sqlite, cli.smoke)
    with cli.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
