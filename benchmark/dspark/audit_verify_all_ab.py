"""Read-only evidence audit. No serving, retries, oracle creation or promotion.

Summarizes every saved request (including excluded and interrupted rows), verifies
token/acceptance contracts, and provides descriptive serial block-bootstrap CIs.
The latter do NOT replace the prespecified p95/control guards or add deployment
confidence from only two counterbalanced pairs.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import fmean

import dspark_prefix_ab as ab
from stable_baseline_gate import sha


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def block_ratio_ci(control, candidate, seed=20261007, iterations=5000):
    if len(control) != 64 or len(candidate) != 64:
        return None
    a = [fmean(control[i:i + 8]) for i in range(0, 64, 8)]
    b = [fmean(candidate[i:i + 8]) for i in range(0, 64, 8)]
    rng = random.Random(seed)
    samples = [fmean(rng.choices(b, k=8)) / fmean(rng.choices(a, k=8))
               for _ in range(iterations)]
    return {'ratio': fmean(candidate) / fmean(control),
        'ci95_low': ab.percentile(samples, .025), 'ci95_high': ab.percentile(samples, .975),
        'resampling_unit': 'independent contiguous 8-request blocks within each serial arm',
        'iterations': iterations,
        'caveat': 'conditional within-window CI; does not establish cross-window/systematic-drift or deployment confidence'}


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    template = manifest['execution_integrity_template']
    oracle = Path(template['args']['oracle_jsonl'])
    dataset = json.loads((oracle.parent / 'dataset.json').read_text())
    refs = ab.load_frozen_references(oracle, dataset)
    result = {'status': json.loads((root / 'status.json').read_text())
                      if (root / 'status.json').exists() else 'incomplete',
        'manifest_sha256': sha(root / 'manifest.json'), 'frozen_oracle_sha256': sha(oracle),
        'arms': {}, 'request_count': 0, 'steady_count': 0,
        'correctness_failures': [], 'scope': 'audit only; does not alter experiment verdict'}
    all_rows = {}
    for folder in sorted(root.glob('[0-9][0-9]_*')):
        rows = read_lines(folder / 'requests.jsonl')
        all_rows[folder.name] = rows
        for ordinal, row in enumerate(rows):
            try:
                ab.check_reference(refs[(row['cell'], row['variant'])], row)
            except Exception as exc:
                result['correctness_failures'].append({'arm': folder.name, 'row': ordinal, 'error': str(exc)})
        steady = [r for r in rows if r['phase'] == 'steady']
        result['request_count'] += len(rows)
        result['steady_count'] += len(steady)
        item = {'requests': len(rows), 'steady': len(steady),
            'phase_counts': {p: sum(r['phase'] == p for r in rows) for p in sorted({r['phase'] for r in rows})},
            'cached_tokens': sorted({r['cached_tokens'] for r in rows}),
            'retractions': sorted({r['meta_info'].get('num_retractions') for r in rows}, key=str)}
        if (folder / 'device_telemetry.csv').exists():
            memory = []
            steady_memory = []
            for line in (folder / 'device_telemetry.csv').read_text().splitlines()[1:]:
                fields = [v.strip() for v in line.split(',')]
                if len(fields) != 10:
                    continue
                try:
                    value = int(fields[2].split()[0])
                    instant = datetime.strptime(fields[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                except (ValueError, TypeError):
                    continue
                memory.append(value)
                if any(r['began_wall_time'] <= instant <= r['finished_wall_time'] for r in steady):
                    steady_memory.append(value)
            item['device_memory_peak_mib_all_phases'] = max(memory) if memory else None
            item['device_memory_peak_mib_steady'] = max(steady_memory) if steady_memory else None
            item['memory_scope'] = 'device-wide NVML samples every200ms, not allocator peak'
        result['arms'][folder.name] = item
    for order, names in [('AB', ('01_adaptive', '02_fixed')), ('BA', ('04_adaptive', '03_fixed'))]:
        if any(name not in all_rows for name in names):
            continue
        left, right = ([r['e2e_ms'] for r in all_rows[name] if r['phase'] == 'steady'] for name in names)
        result.setdefault('e2e_block_bootstrap', {})[order] = block_ratio_ci(left, right)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, help='Create a new audit JSON; never replace existing evidence')
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
