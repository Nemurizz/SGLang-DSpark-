"""Paired fixed-subset scores/uncertainty; never treat stability as quality."""
import argparse
import json
import math
from pathlib import Path
import random

from quality_data import sha


def paired_interval(deltas, draws=10000):
    rng = random.Random(20261009)
    n = len(deltas)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(draws))
    return [samples[int(.025 * draws)], samples[int(.975 * draws)]]


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - width, center + width]


def report(root, output):
    manifests, scores = {}, {}
    for arm in ('bf16-quality', 'quant-quality'):
        folder = root / arm
        if json.loads((folder / 'status.json').read_text())['status'] != 'passed':
            raise ValueError('Cannot score an incomplete/failed quality arm')
        manifests[arm] = json.loads((folder / 'manifest.json').read_text())
        rows = [json.loads(line) for line in (folder / 'scores.jsonl').read_text().splitlines()]
        scores[arm] = {r['id']: r for r in rows}
        if len(rows) != 384 or len(scores[arm]) != len(rows):
            raise ValueError('Quality requires all 384 distinct items')
    if manifests['bf16-quality']['dataset_sha256'] != manifests['quant-quality']['dataset_sha256']:
        raise ValueError('Different prompt/grade dataset')
    if set(scores['bf16-quality']) != set(scores['quant-quality']):
        raise ValueError('Different scored IDs')
    result = {'scope': 'Fixed heldout subsets, custom generative ARC and scalar MATH. NOT leaderboard/full-distribution noninferiority.',
        'dataset_sha256': manifests['bf16-quality']['dataset_sha256'], 'tasks': {},
        'performance_benefit': 'unproven', 'created_from': {arm: sha(root / arm / 'scores.jsonl') for arm in scores}}
    for task in sorted({r['task'] for r in scores['bf16-quality'].values()}):
        ids = [key for key, r in scores['bf16-quality'].items() if r['task'] == task]
        left = [scores['bf16-quality'][key]['grade']['correct'] for key in ids]
        right = [scores['quant-quality'][key]['grade']['correct'] for key in ids]
        deltas = [int(b) - int(a) for a, b in zip(left, right)]
        losses = [key for key in ids if scores['bf16-quality'][key]['grade']['correct']
                  and not scores['quant-quality'][key]['grade']['correct']]
        gains = [key for key in ids if scores['quant-quality'][key]['grade']['correct']
                 and not scores['bf16-quality'][key]['grade']['correct']]
        item = {'n': len(ids), 'bf16_correct': sum(left), 'nvfp4_correct': sum(right),
            'bf16_score': sum(left)/len(ids), 'nvfp4_score': sum(right)/len(ids),
            'score_delta': sum(deltas)/len(ids), 'paired_bootstrap_95_interval': paired_interval(deltas),
            'bf16_wilson_95_interval': wilson(sum(left), len(ids)),
            'nvfp4_wilson_95_interval': wilson(sum(right), len(ids)),
            'lost_correct_ids': losses, 'gained_correct_ids': gains,
            'truncated': {arm: sum(scores[arm][key]['grade']['truncated'] for key in ids) for arm in scores}}
        result['tasks'][task] = item
    result['no_observed_per_task_loss'] = all(r['score_delta'] >= 0 for r in result['tasks'].values())
    result['quality_gate'] = 'passed_for_fixed_subsets_only' if result['no_observed_per_task_loss'] else 'failed_observed_task_loss'
    if output.exists():
        raise ValueError('Preserve previous report; fresh output required')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report(args.root, args.output)
