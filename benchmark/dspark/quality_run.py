"""Real-task BF16/NVFP4 quality and variable-EOS exact execution gates.

No speed claims from these heterogeneous generations. Every arm is a new
owned service; model quality differences are NOT token correctness failures.
Only within-math AR/DSpark or repeat differences are safety failures.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import dspark_prefix_ab as ab
from code_jail import evaluate
from quality_data import numeric, final_answer, sha
from quant_feasibility_gate import (baseline_identity, launch_args_from_template,
    quant_command, validate_info)
from stable_baseline_gate import profile_command
from stable_profile_hook import validate_runtime
from nvfp4_profile_hook import validate_model

PROFILE = 'dspark-real-quality-fa4fixed64-h64-v1'


def grade(row, response, jail):
    text = response['text'].rsplit('</think>', 1)[-1]
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    truncated = response['meta_info']['finish_reason'].get('type') == 'length'
    if truncated:
        return {'correct': False, 'truncated': True, 'reason': 'length'}
    if row['task'] == 'mbpp':
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
        code = blocks[0] if blocks else text.strip()
        result = evaluate(jail, code, row['tests'], row['imports'])
        return {'correct': result['passed'], 'truncated': False, 'code': code, 'sandbox': result}
    answer = final_answer(text)
    try:
        correct = (answer.strip().upper() == row['gold'] if row['task'] == 'arc_generative'
                   else numeric(answer) == numeric(row['gold']))
    except ValueError:
        correct = False
    return {'correct': correct, 'truncated': False, 'extracted_answer': answer}


async def generate(session, base, row, tokenizer, seed):
    started = time.perf_counter()
    tracker = ab.StreamTracker(started)
    body = {'input_ids': row['input_ids'], 'stream': True,
            'sampling_params': {'temperature': 0, 'top_p': 1, 'top_k': -1,
                'max_new_tokens': row['max_tokens'], 'ignore_eos': False, 'sampling_seed': seed}}
    async with session.post(base + '/generate', json=body) as response:
        response.raise_for_status()
        async for frame in ab.sse_frames(response.content):
            if frame == '[DONE]':
                break
            tracker.add(json.loads(frame), time.perf_counter())
    result = tracker.result(time.perf_counter())
    if result['prompt_tokens'] != len(row['input_ids']) or result['cached_tokens']:
        raise ab.SafetyFailure('Variable prompt length/cache contract violated')
    if result['meta_info'].get('num_retractions', 0) or result['meta_info'].get('retraction_count', 0):
        raise ab.SafetyFailure('Unexpected retraction')
    result['text'] = tokenizer.decode(result['output_ids'], skip_special_tokens=False)
    result.update(id=row['id'], task=row['task'], cell=row['id'], variant=0)
    return result


class QualityServer(ab.ManagedServer):
    def command(self):
        cmd, env = super().command()
        if self.args.quant:
            cmd, env = quant_command(cmd, env, self.args,
                ordinary=self.args.ordinary, folder=self.folder)
        else:
            cmd, env = profile_command(cmd, env, ordinary=self.args.ordinary,
                profile_id=PROFILE, runtime_folder=self.folder / 'runtime',
                hook_folder=Path(__file__).resolve().parent, fixed_tile=[64, 64])
            if '--disable-radix-cache' not in cmd:
                cmd.append('--disable-radix-cache')
        return cmd, env


def validate(folder, quant, ordinary):
    info = json.loads((folder / 'server_info.json').read_text())
    if quant:
        validate_info(info, ordinary)
        actual = json.loads((folder / 'runtime/target-backbone.json').read_text())
        actual.update(json.loads((folder / 'runtime/target-head.json').read_text()))
        validate_model(actual)
        if not ordinary:
            validate_model(json.loads((folder / 'runtime/draft-weights.json').read_text()))
    else:
        expected = {'enable_deterministic_inference': True, 'disable_cuda_graph': True,
            'disable_radix_cache': True, 'attention_backend': 'fa4',
            'prefill_attention_backend': 'fa4', 'decode_attention_backend': 'fa4',
            'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16', 'bf16_gemm_backend': 'torch',
            'tp_size': 1, 'page_size': 1, 'quantization': None}
        if not ordinary:
            expected['speculative_algorithm'] = 'DSPARK'
        for key, value in expected.items():
            if info.get(key) != value:
                raise ab.SafetyFailure(f'BF16 profile {key} differs: {info.get(key)!r}')
    runtime = list((folder / 'runtime').glob('runtime-*.json'))
    if not runtime:
        raise ab.SafetyFailure('Actual attention geometry attestation missing')
    for path in runtime:
        validate_runtime(json.loads(path.read_text()), fixed_tile=[64, 64])


async def execute(cli):
    import aiohttp
    from transformers import AutoTokenizer
    if cli.output.exists():
        raise ValueError('New evidence folder required; do not overwrite/retry')
    args = launch_args_from_template(json.loads(cli.template.read_text()))
    args.tokenizer = args.model
    args.quant = cli.arm.startswith('quant')
    args.ordinary = cli.arm.endswith('ar')
    args.startup_timeout, args.request_timeout = 900, 240
    args.enable_standard_graphs = False
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    if args.quant:
        args.model = str(cli.checkpoint / 'model')
    dataset = json.loads(cli.dataset.read_text())
    rows = dataset['rows']
    if cli.sentinels:
        rows = [r for task in ('gsm8k', 'math500_scalar', 'arc_generative', 'mbpp')
                for r in [r for r in rows if r['task'] == task][:2]]
    refs = {}
    if cli.references:
        reference_status = json.loads((cli.references.parent / 'status.json').read_text())
        reference_manifest = json.loads((cli.references.parent / 'manifest.json').read_text())
        if reference_status['status'] != 'passed' or reference_manifest['dataset_sha256'] != sha(cli.dataset):
            raise ValueError('AR references must be complete and same dataset')
        if reference_manifest['quant'] != args.quant or not reference_manifest['ordinary']:
            raise ValueError('Cannot compare different arithmetic contracts')
        refs = {r['id']: r for r in map(json.loads, cli.references.read_text().splitlines())
                if r['phase'] == 'serial_reference'}
        if set(refs) != {r['id'] for r in rows}:
            raise ValueError('References do not match exact requested rows')
    if not args.ordinary and cli.sentinels and not refs:
        raise ValueError('DSpark sentinel gate requires its independent AR references')
    before = baseline_identity(args.repo)
    cli.output.mkdir(parents=True)
    hashes = {name: sha(Path(__file__).with_name(name)) for name in (
        'quality_run.py', 'quality_data.py', 'code_jail.py', 'quant_feasibility_gate.py',
        'nvfp4_profile_hook.py', 'fixed_fa4_policy_hook.py', 'stable_profile_hook.py',
        'stable_baseline_gate.py', 'dspark_prefix_ab.py')}
    ab.dump(cli.output / 'manifest.json', {'arm': cli.arm, 'quant': args.quant, 'ordinary': args.ordinary,
        'args': vars(args), 'dataset_sha256': sha(cli.dataset), 'tools_sha256': hashes,
        'sentinels_only': cli.sentinels, 'concurrency': cli.concurrency,
        'within_contract_oracle_sha256': sha(cli.references) if cli.references else None,
        'old_bf16_oracle_sha256': sha(args.oracle_jsonl), 'baseline_before': before,
        'performance_benefit': 'NOT measured by quality run', 'created_at': time.time()})
    status = {'status': 'running', 'completed_requests': 0, 'scored_requests': 0, 'arm': cli.arm}
    ab.dump(cli.output / 'status.json', status)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    server = QualityServer(args, 'full', cli.output)
    sampler = ab.GPUSampler(cli.output / 'gpu.jsonl', 200)
    sentinel_refs = {}
    try:
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise ab.SafetyFailure('GPU occupied: ' + occupied)
        sampler.start()
        server.start()
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp),
                timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as session:
            base = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, '/model_info')
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() >= deadline:
                        raise ab.SafetyFailure('Startup timeout')
                    await asyncio.sleep(.5)
            ab.dump(cli.output / 'server_info.json', await ab.api(session, base, '/server_info'))

            async def checked(row, phase, reference=None, acceptance=False):
                task = asyncio.create_task(generate(session, base, row, tokenizer, args.seed))
                try:
                    while not task.done():
                        await asyncio.wait([task], timeout=.5)
                        server.check()
                    response = await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                response.update(phase=phase, arm=cli.arm)
                ab.append_jsonl(cli.output / 'requests.jsonl', response)
                if not args.ordinary:
                    ab.check_reference(response, response)
                if reference:
                    ab.check_reference(reference, response, check_acceptance=acceptance)
                if phase == 'quality':
                    response['grade'] = grade(row, response, cli.jail)
                    ab.append_jsonl(cli.output / 'scores.jsonl', response)
                    status['scored_requests'] += 1
                status['completed_requests'] += 1
                ab.dump(cli.output / 'status.json', status)
                print(json.dumps({'id': row['id'], 'phase': phase, 'tokens': len(response['output_ids']),
                    'grade': response.get('grade', {}).get('correct'), 'finished': response['meta_info']['finish_reason']}), flush=True)
                return response

            if cli.sentinels:
                for row in rows:
                    response = await checked(row, 'serial_reference', refs.get(row['id']))
                    sentinel_refs[row['id']] = response
                validate(cli.output, args.quant, args.ordinary)
                for concurrency in (1, 4, 8):
                    for repeat in range(2):
                        for start in range(0, len(rows), concurrency):
                            await asyncio.gather(*(checked(row, f'c{concurrency}_r{repeat}',
                                sentinel_refs[row['id']], not args.ordinary) for row in rows[start:start + concurrency]))
            else:
                for start in range(0, len(rows), cli.concurrency):
                    await asyncio.gather(*(checked(row, 'quality') for row in rows[start:start + cli.concurrency]))
                    validate(cli.output, args.quant, args.ordinary)
                scores = [json.loads(line) for line in (cli.output / 'scores.jsonl').read_text().splitlines()]
                ab.dump(cli.output / 'summary.json', {task: {'n': len(items),
                    'correct': sum(r['grade']['correct'] for r in items),
                    'truncated': sum(r['grade']['truncated'] for r in items)}
                    for task in sorted({r['task'] for r in rows})
                    for items in [[r for r in scores if r['task'] == task]]})
            status['status'] = 'passed'
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        server.stop()
        sampler.stop()
        status.update(baseline_after=baseline_identity(args.repo), finished_at=time.time())
        if status['baseline_after'] != before:
            status.update(status='safety_stopped', error='Serving checkout mutated')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'passed' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=['bf16-ar', 'bf16-dspark', 'quant-ar', 'quant-dspark'])
    p.add_argument('--template', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--jail', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--references', type=Path)
    p.add_argument('--sentinels', action='store_true')
    p.add_argument('--concurrency', type=int, default=8, choices=[1, 4, 8])
    raise SystemExit(asyncio.run(execute(p.parse_args())))
