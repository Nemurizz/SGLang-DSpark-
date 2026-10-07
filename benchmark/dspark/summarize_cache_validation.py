"""Read-only JSON evidence reduction; raw token/acceptance records stay intact."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def summarize(root):
    root = Path(root)
    result = {'directory': str(root)}
    for name in ('status.json', 'recovery_summary.json', 'performance_summary.json'):
        if (root / name).exists():
            result[name] = json.loads((root / name).read_text())
    for folder in sorted(root.iterdir()):
        path = folder / 'requests.jsonl'
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        phases = collections.Counter(row['phase'] for row in rows)
        cells = collections.Counter(row['cell'] for row in rows)
        result.setdefault('arms', {})[folder.name] = {
            'saved_requests': len(rows), 'phases': dict(phases), 'cells': dict(cells),
            'cached_token_lengths': sorted({row['cached_tokens'] for row in rows}),
            'output_lengths': sorted({len(row['output_ids']) for row in rows}),
            'descriptive_e2e_median_ms_not_ab': statistics.median(row['e2e_ms'] for row in rows) if rows else None}
        if (folder / 'status.json').exists():
            result['arms'][folder.name]['status'] = json.loads((folder / 'status.json').read_text())
    if (root / 'summaries.jsonl').exists():
        result['timing_summaries'] = [json.loads(line) for line in (root / 'summaries.jsonl').read_text().splitlines()]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
