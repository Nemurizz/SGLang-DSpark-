"""Bounded NVFP4 target/BF16 draft feasibility and exact within-profile gate.

Ordinary quantized target supplies the NEW token oracle; old BF16 oracle is
immutable and is not an equality target. Never start performance A/B on failure.
Quality/effectiveness on real tasks remains a separate prerequisite.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from stable_baseline_gate import profile_command, SOURCE_PATHS
from stable_profile_hook import validate_runtime
from nvfp4_profile_hook import validate_model
from prepare_nvfp4 import REVISION

BASELINE = '0577109f4e659b8ef2ef3f979f1fd7f079b25821'
MAIN = '0c1f1eb64af939379ce47045c56aa8d2508c54df'
UNTRACKED = '9077a67cf4e576a78b7750734c1e956defe85114739b9bb6eed68bc828708b7e'
PROFILE = 'dspark-qwen3-14b-nvfp4-target-bf16-draft-fa4fixed64-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_identity(repo):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    identity = {'head': git('rev-parse', 'HEAD').decode().strip(),
                'main': git('rev-parse', 'refs/heads/main').decode().strip(),
                'tracked_clean': not git('status', '--porcelain', '--untracked-files=no'),
                'untracked_paths_sha256': hashlib.sha256(git('ls-files', '--others', '--exclude-standard', '-z')).hexdigest()}
    if identity != dict(head=BASELINE, main=MAIN, tracked_clean=True, untracked_paths_sha256=UNTRACKED):
        raise ab.SafetyFailure('Baseline checkout identity changed: ' + repr(identity))
    paths = SOURCE_PATHS + (
        'python/sglang/srt/layers/quantization/modelopt_quant.py',
        'python/sglang/srt/layers/quantization/fp4_utils.py',
        'python/sglang/srt/model_executor/hook_manager.py',
        'python/sglang/srt/models/dflash.py',
        'python/sglang/srt/layers/logits_processor.py',
    )
    identity['serving_source_sha256'] = {name: digest(Path(repo) / name) for name in paths}
    return identity


def quant_command(cmd, env, args, *, ordinary, folder):
    cmd, env = profile_command(cmd, env, ordinary=ordinary, profile_id=PROFILE,
        runtime_folder=folder / 'runtime', hook_folder=Path(__file__).resolve().parent,
        fixed_tile=[64, 64])
    cmd[cmd.index('--tokenizer-path') + 1] = args.tokenizer
    cmd += ['--fp4-gemm-backend', 'flashinfer_cutlass', '--disable-radix-cache']
    if not ordinary:
        cmd += ['--speculative-draft-model-quantization', 'unquant',
                '--speculative-draft-kv-cache-dtype', 'bfloat16']
    # No explicit --quantization: detect the pinned offline checkpoint format.
    hooks = json.loads(cmd[cmd.index('--forward-hooks') + 1])
    hooks[0]['hook_factory'] = 'nvfp4_profile_hook:make_hook'
    hooks.append({'name': 'quant-model-storage-attestation', 'target_modules': ['model'],
                  'hook_factory': 'nvfp4_profile_hook:make_model_hook',
                  'config': {'folder': str(folder / 'runtime')}})
    if not ordinary:
        hooks.append({'name': 'bf16-draft-storage-attestation',
                      'target_modules': [f'layers.{i}' for i in range(5)],
                      'hook_factory': 'nvfp4_profile_hook:make_draft_hook',
                      'config': {'folder': str(folder / 'runtime')}})
    cmd[cmd.index('--forward-hooks') + 1] = json.dumps(hooks)
    for key in ('SGLANG_DSPARK_PREFIX_CACHE', 'SGLANG_DSPARK_PREFIX_CACHE_SHADOW',
                'SGLANG_DSPARK_RAGGED_VERIFY_GRAPH', 'SGLANG_DSPARK_COMPACT_SCAN_CUDA',
                'SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH'):
        env[key] = '0'
    return cmd, env


class QuantServer(ab.ManagedServer):
    def command(self):
        return quant_command(*super().command(), self.args,
                             ordinary=self.args.ordinary, folder=self.folder)


def validate_info(info, ordinary):
    expected = dict(enable_deterministic_inference=True, disable_cuda_graph=True,
                    disable_radix_cache=True, dtype='bfloat16', kv_cache_dtype='bfloat16',
                    attention_backend='fa4', prefill_attention_backend='fa4',
                    decode_attention_backend='fa4', bf16_gemm_backend='torch',
                    tp_size=1, page_size=1, quantization='modelopt_fp4',
                    fp4_gemm_runner_backend='flashinfer_cutlass')
    if not ordinary:
        expected.update(speculative_algorithm='DSPARK',
                        speculative_draft_model_quantization=None,
                        speculative_draft_kv_cache_dtype='bfloat16')
    for key, value in expected.items():
        if info.get(key) != value:
            raise ab.SafetyFailure(f'Unqualified quant profile {key}: {info.get(key)!r} != {value!r}')


async def execute(args, folder, prompts, target_refs, status):
    import aiohttp
    folder.mkdir()
    server = QuantServer(args, 'full', folder)
    base = f'http://127.0.0.1:{args.port}'
    sampler = ab.GPUSampler(folder / 'gpu.jsonl', 200)
    async def request(session, index, phase, reference=None, acceptance=False):
        row = await ab.generate(session, base, prompts[index], args.output_tokens, args.seed)
        row.update(cell=f'prompt_{index}', variant=index, phase=phase,
                   model_revision=REVISION, ordinary=args.ordinary, profile_id=PROFILE,
                   prompt_sha256=hashlib.sha256(json.dumps(prompts[index]).encode()).hexdigest())
        ab.append_jsonl(folder / 'requests.jsonl', row)
        if row['cached_tokens'] or row['meta_info'].get('retraction_count', 0):
            raise ab.SafetyFailure('Cache/retraction must remain zero')
        if not args.ordinary:
            ab.check_reference(row, row)
        if reference is not None:
            ab.check_reference(reference, row, check_acceptance=acceptance)
        server.check()
        status['completed_requests'] += 1
        return row
    try:
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise ab.SafetyFailure('GPU occupied; do not replace external work: ' + occupied)
        sampler.start()
        server.start()
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp),
                timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as session:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, '/model_info')
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() > deadline:
                        raise ab.SafetyFailure('Quantized server startup timeout')
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, '/server_info')
            ab.dump(folder / 'server_info.json', info)
            validate_info(info, args.ordinary)
            refs = {}
            for index in range(len(prompts)):
                refs[index] = await request(session, index, 'new_quant_reference',
                                            None if args.ordinary else target_refs[index])
                ab.append_jsonl(folder / 'references.jsonl', refs[index])
            runtime_files = list((folder / 'runtime').glob('runtime-*.json'))
            if not runtime_files:
                raise ab.SafetyFailure('Missing actual attention attestation')
            for path in runtime_files:
                validate_runtime(json.loads(path.read_text()), fixed_tile=[64, 64])
            actual = json.loads((folder / 'runtime' / 'target-backbone.json').read_text())
            actual.update(json.loads((folder / 'runtime' / 'target-head.json').read_text()))
            validate_model(actual)
            ab.dump(folder / 'runtime' / 'target-weights.json', actual)
            if not args.ordinary:
                validate_model(json.loads((folder / 'runtime' / 'draft-weights.json').read_text()))
            for concurrency in (1, 4, 8):
                for repeat in range(2):
                    for start in range(0, len(prompts), concurrency):
                        tasks = [asyncio.create_task(request(session, index,
                            f'c{concurrency}_repeat{repeat}', refs[index], not args.ordinary))
                            for index in range(start, min(start + concurrency, len(prompts)))]
                        try:
                            await asyncio.gather(*tasks)
                        finally:
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
            status['arms_completed'].append(folder.name)
            print('PASSED ' + folder.name, flush=True)
            return refs
    finally:
        server.stop()
        sampler.stop()


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', required=True, type=Path)
    p.add_argument('--checkpoint-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=['target', 'dspark'], required=True)
    p.add_argument('--target-references', type=Path)
    cli = p.parse_args()
    if cli.output.exists():
        raise ValueError('Preserve evidence; use a fresh output folder')
    cli.output.mkdir(parents=True)
    template = json.loads(cli.template.read_text())
    # Reuse launch scalars only; old quant-unaware integrity/profile checks are
    # intentionally NOT treated as certificates for this new math contract.
    args = SimpleNamespace(**template['args'])
    args.tokenizer = args.model
    args.model = str(cli.checkpoint_root / 'model')
    args.ordinary = cli.phase == 'target'
    args.startup_timeout = 900
    args.request_timeout = 45
    args.port = 30117
    args.enable_standard_graphs = False
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    args.output_tokens = 128
    # The long128 fixed dataset is reused only as an execution/correctness
    # workload; it is NOT a representative quality benchmark.
    old_data = Path(args.oracle_jsonl).parent / 'dataset.json'
    prompts = json.loads(old_data.read_text())['p512_c8']['prompts']
    before = baseline_identity(args.repo)
    frozen = {str(old_data): digest(old_data), args.oracle_jsonl: digest(args.oracle_jsonl)}
    model = json.loads((cli.checkpoint_root / 'manifest.json').read_text())
    if model['status'] != 'verified' or model['revision'] != REVISION:
        raise ab.SafetyFailure('Full model manifest is not verified')
    for f in model['files']:
        if (Path(args.model) / f['name']).stat().st_size != f['bytes']:
            raise ab.SafetyFailure('Checkpoint file changed after verification')
    refs = None
    if not args.ordinary:
        if cli.target_references is None:
            raise ValueError('DSpark requires the independent quantized AR target references')
        target_status = json.loads((cli.target_references.parent.parent / 'status.json').read_text())
        target_manifest = json.loads((cli.target_references.parent.parent / 'manifest.json').read_text())
        if target_status['status'] != 'passed_for_bounded_workload_only' or target_status['phase'] != 'target':
            raise ValueError('Do not use references from an incomplete/failed target gate')
        if target_manifest['checkpoint_manifest'] != model or target_manifest['old_reference_sha256'] != frozen:
            raise ValueError('Target gate belongs to different checkpoint/input metadata')
        rows = [json.loads(line) for line in cli.target_references.read_text().splitlines()]
        if len(rows) != 8 or any(not r['ordinary'] or r['model_revision'] != REVISION
                                or r.get('profile_id') != PROFILE for r in rows):
            raise ValueError('Unqualified target references')
        refs = {r['variant']: r for r in rows}
        if set(refs) != set(range(8)):
            raise ValueError('Incomplete target oracle')
        if any(refs[i].get('prompt_sha256') != hashlib.sha256(json.dumps(prompts[i]).encode()).hexdigest()
               for i in range(8)):
            raise ValueError('Target oracle prompt identity differs')
    status = {'status': 'running', 'phase': cli.phase, 'completed_requests': 0,
              'arms_completed': [], 'started_at': time.time(),
              'performance_benefit': 'unproven', 'quality': 'not_evaluated',
              'baseline_identity_before': before}
    ab.dump(cli.output / 'manifest.json', {'args': vars(args), 'baseline': before,
             'checkpoint_manifest': model, 'old_reference_sha256': frozen,
             'workload': 'p512_c8 fixed inputs, 128 output; serial/c4/c8 two repeats',
             'quality_gate': 'NOT covered by synthetic execution workload',
             'stop_on_any_token_or_within_dspark_acceptance_mismatch': True})
    try:
        await execute(args, cli.output / cli.phase, prompts, refs, status)
        status['status'] = 'passed_for_bounded_workload_only'
    except Exception as exc:
        status.update(status='safety_stopped', error_type=type(exc).__name__, error=str(exc))
    finally:
        status['baseline_identity_after'] = baseline_identity(args.repo)
        status['old_references_unchanged'] = all(digest(k) == v for k, v in frozen.items())
        status['finished_at'] = time.time()
        if not status['old_references_unchanged'] or status['baseline_identity_after'] != before:
            status.update(status='safety_stopped', error='Frozen BF16 input/source mutated')
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'passed_for_bounded_workload_only' else 2


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
