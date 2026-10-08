"""Pinned, output-blind paired quality sample. Not a leaderboard protocol."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

SEED = 'dspark-quality-20261009-h64-v1'
SOURCES = {
    'gsm8k': ('https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl', 'gsm8k.jsonl'),
    'math500': ('https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl', 'math500.jsonl'),
    'arc': ('https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/test-00000-of-00001.parquet', 'arc.parquet'),
    'mbpp': ('https://raw.githubusercontent.com/google-research/google-research/72ffe35f5a24a312980e6bb811713b24f31e11f9/mbpp/sanitized-mbpp.json', 'mbpp.json'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(text):
    """Deliberately narrow exact rational grader; eligibility uses gold only."""
    text = str(text).strip().replace(',', '').replace('$', '').replace(' ', '')
    text = re.sub(r'\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', r'\1/\2', text)
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', text):
        raise ValueError('not a supported scalar rational: ' + text)
    return Fraction(text)


def final_answer(text):
    text = text.rsplit('</think>', 1)[-1]
    markers = list(re.finditer(r'(?:####|FINAL:|ANSWER:)\s*([^\n]+)', text, re.I))
    if markers:
        return markers[-1].group(1).strip().rstrip('.')
    boxes = list(re.finditer(r'\\boxed\{', text))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                return text[start:end].strip()
    return ''


def select(rows, task, n):
    return sorted(rows, key=lambda r: hashlib.sha256(
        (SEED + task + r['id']).encode()).hexdigest())[:n]


def prepare(root, tokenizer_path, staged_sources=None):
    import requests
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    if root.exists():
        raise ValueError('fresh dataset evidence folder required')
    root.mkdir(parents=True)
    sources = {}
    for key, (url, name) in SOURCES.items():
        if staged_sources:
            content = (staged_sources / name).read_bytes()
        else:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
        (root / name).write_bytes(content)
        sources[key] = {'url': url, 'sha256': sha(root / name), 'bytes': len(content),
                        'transport': 'desktop TLS download then SSH staging' if staged_sources else 'direct TLS download'}
    gsm = []
    for i, line in enumerate((root / 'gsm8k.jsonl').read_text().splitlines()):
        r = json.loads(line)
        gold = str(numeric(r['answer'].rsplit('####', 1)[-1]))
        gsm.append(dict(id=f'gsm8k-test-{i}', task='gsm8k', gold=gold,
            prompt='Solve the following problem. Explain briefly and end with exactly "FINAL: <number>".\n\n' + r['question'],
            max_tokens=1024, thinking=False))
    maths = []
    for line in (root / 'math500.jsonl').read_text().splitlines():
        r = json.loads(line)
        if '[asy]' in r['problem']:
            continue
        try:
            gold = str(numeric(r['answer']))
        except ValueError:
            continue
        maths.append(dict(id='math500-' + str(r['unique_id']), task='math500_scalar',
            gold=gold, subject=r['subject'], level=r['level'],
            prompt='Solve this mathematics problem. End your answer with exactly "FINAL: <integer, decimal, or fraction>".\n\n' + r['problem'],
            max_tokens=4096, thinking=True))
    arcs = []
    for r in pq.read_table(root / 'arc.parquet').to_pylist():
        choices = r['choices']
        labels = choices['label']
        letter_map = {label: chr(65 + i) for i, label in enumerate(labels)}
        prompt = 'Choose the correct answer. End with exactly "ANSWER: <letter>".\n\n' + r['question'] + '\n'
        prompt += '\n'.join(letter_map[label] + '. ' + text for label, text in zip(labels, choices['text']))
        arcs.append(dict(id='arc-' + str(r['id']), task='arc_generative',
                         gold=letter_map[r['answerKey']], prompt=prompt, max_tokens=256, thinking=False))
    mbpps = []
    for r in json.loads((root / 'mbpp.json').read_text()):
        if not 11 <= int(r['task_id']) <= 510:
            continue
        mbpps.append(dict(id=f"mbpp-{r['task_id']}", task='mbpp', gold_code=r['code'],
            tests=r['test_list'], imports=r.get('test_imports', []),
            prompt='Write Python 3 code solving the task and passing these tests. Return only a single Python code block, without explanation.\n\n' + r['prompt'] + '\n\n' + '\n'.join(r['test_list']),
            max_tokens=1024, thinking=False))
    rows = select(gsm, 'gsm8k', 128) + select(maths, 'math500', 64) + select(arcs, 'arc', 128) + select(mbpps, 'mbpp', 64)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    for r in rows:
        r['input_ids'] = tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=r['thinking'])
        if not isinstance(r['input_ids'], list) or not all(isinstance(t, int) for t in r['input_ids']):
            raise TypeError('Chat template must produce flat token IDs')
    dataset = {'protocol': SEED, 'rows': rows, 'sources': sources,
        'eligible_counts': {'gsm8k': len(gsm), 'math500_scalar': len(maths), 'arc_generative': len(arcs), 'mbpp': len(mbpps)},
        'tokenizer_path': tokenizer_path,
        'tokenizer_metadata_sha256': {p.name: sha(p) for p in Path(tokenizer_path).glob('*')
                                     if p.is_file() and ('tokenizer' in p.name or p.name == 'config.json')},
        'scope': 'Fixed output-blind heldout subsets; ARC generative, MATH scalar subset. NOT standard full benchmark scores.',
        'quality_acceptance': 'No observed per-task score loss vs BF16; report paired discordance and confidence intervals. Not proof of universal equivalence.',
        'truncation_policy': 'length-truncated outputs count incorrect; no retries, filtering, or post-output resampling'}
    (root / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'dataset_sha256': sha(root / 'dataset.json'), 'counts': {task: sum(r['task'] == task for r in rows) for task in set(r['task'] for r in rows)}, 'eligible': dataset['eligible_counts']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--staged-sources', type=Path)
    cli = p.parse_args()
    prepare(cli.output, cli.tokenizer, cli.staged_sources)
