"""Post-run evidence only. No filtering, repeats or changes to benchmark admission."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics


def analyze(root):
    result = {'status': json.loads((root / 'status.json').read_text()),
        'protocol': json.loads((root / 'manifest.json').read_text())['protocol'],
        'arms': [], 'scope': 'post-run descriptive telemetry; NOT causal performance diagnosis'}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (folder / 'requests.jsonl').exists():
            continue
        requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
        steady = [r for r in requests if r['phase'] == 'steady']
        counters = Counter(r['phase'] for r in requests)
        samples = []
        for original in csv.DictReader((folder / 'device_telemetry.csv').open()):
            row = {k.strip(): v.strip() for k, v in original.items()}
            stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(
                tzinfo=timezone(timedelta(hours=8))).timestamp()
            if any(r['began_wall_time'] <= stamp <= r['finished_wall_time'] for r in steady):
                samples.append(row)
        metrics = {}
        for key in ('memory.used [MiB]', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]',
                    'power.draw [W]', 'temperature.gpu', 'utilization.gpu [%]'):
            values = [float(r[key].split()[0]) for r in samples if r[key] != '[N/A]']
            metrics[key] = {'samples': len(values), 'min': min(values),
                            'max': max(values), 'mean': statistics.mean(values)} if values else None
        reasons = Counter(r['clocks_event_reasons.active'] for r in samples)
        groups = []
        for window in sorted({r['window'] for r in steady}):
            rows = [r for r in steady if r['window'] == window]
            groups += [{'window': window, 'group': i // 8,
                **{metric: statistics.mean(r[metric] for r in rows[i:i+8])
                   for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms')}} for i in range(0, len(rows), 8)]
        result['arms'].append({'name': folder.name, 'all_saved_requests': len(requests),
            'phase_counts': dict(counters), 'all_checked_by_live_harness': True,
            'steady_telemetry': metrics, 'steady_clock_event_counts': dict(reasons),
            'all_steady_groups_no_discard': groups})
    result['sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in root.rglob('*') if p.is_file()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: result[k] for k in ('scope', 'arms')}, indent=2))
