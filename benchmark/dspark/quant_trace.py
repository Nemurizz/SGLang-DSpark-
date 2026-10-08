"""Owned short native FP4 model trace; NOT performance or quality evidence."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import time

import dspark_prefix_ab as ab
from quant_feasibility_gate import baseline_identity, launch_args_from_template
from quality_run import QualityServer, validate

NSYS = '/root/autodl-fs/dspark-nsys-2025.6.3.541-20261008-h40/package/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys'


class TracedServer(QualityServer):
    def command(self):
        cmd, env = super().command()
        hooks = json.loads(cmd[cmd.index('--forward-hooks') + 1])
        hooks[0]['hook_factory'] = 'quant_trace_hook:make_hook'
        hooks[0]['config']['trace_folder'] = str(self.folder)
        cmd[cmd.index('--forward-hooks') + 1] = json.dumps(hooks)
        prefix = [NSYS, 'profile', '--trace=cuda-sw,nvtx', '--sample=none', '--cpuctxsw=none',
            '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--kill=none',
            '--wait=all', '--flush-on-cudaprofilerstop=true', '--cuda-event-trace=false',
            '--output=' + str(self.folder / 'trace')]
        return prefix + cmd, env


async def run(cli):
    import aiohttp
    from audit_nsys_collector import audit
    if cli.output.exists():
        raise ValueError('Fresh evidence required')
    args = launch_args_from_template(json.loads(cli.template.read_text()))
    args.tokenizer = args.model
    args.model = str(cli.checkpoint / 'model')
    args.quant, args.ordinary, args.fp4_backend = True, True, cli.backend
    args.startup_timeout, args.request_timeout = 900, 45
    args.enable_standard_graphs = False
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    before = baseline_identity(args.repo)
    old_data = Path(args.oracle_jsonl).parent / 'dataset.json'
    prompt = json.loads(old_data.read_text())['p2048_c1']['prompts'][0]
    cli.output.mkdir()
    server = TracedServer(args, 'full', cli.output)
    status = {'status': 'running', 'benefit': 'NOT measured', 'quality': 'NOT evaluated'}
    try:
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise ValueError('GPU occupied')
        server.start()
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp), timeout=aiohttp.ClientTimeout(total=45)) as session:
            base = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + 900
            while True:
                server.check()
                try:
                    await ab.api(session, base, '/model_info')
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Startup deadline')
                    await asyncio.sleep(.5)
            ab.dump(cli.output / 'server_info.json', await ab.api(session, base, '/server_info'))
            async def checked(phase, reference=None):
                row = await ab.generate(session, base, prompt, 64, args.seed)
                row.update(phase=phase)
                ab.append_jsonl(cli.output / 'requests.jsonl', row)
                if reference:
                    ab.check_reference(reference, row, check_acceptance=False)
                validate(cli.output, True, True, args.fp4_backend)
                server.check()
                return row
            ref = await checked('prime_excluded')
            await checked('warmup_excluded', ref)
            (cli.output / 'trace_start').touch(exist_ok=False)
            await checked('diagnostic_capture', ref)
            (cli.output / 'trace_stop').touch(exist_ok=False)
            await checked('capture_close_excluded', ref)
            deadline = time.monotonic() + 60
            report = cli.output / 'trace.nsys-rep'
            while not report.exists():
                server.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError('Collector report missing')
                await asyncio.sleep(.5)
        status['status'] = 'captured_exact_64_token_repeats'
    except Exception as exc:
        status.update(status='safety_stopped', error=f'{type(exc).__name__}: {exc}')
    finally:
        server.stop()
        status['baseline_unchanged'] = baseline_identity(args.repo) == before
        ab.dump(cli.output / 'status.json', status)
    if status['status'].startswith('captured'):
        subprocess.run([NSYS, 'export', '--type=sqlite', '--output=' + str(cli.output / 'trace.sqlite'),
            str(cli.output / 'trace.nsys-rep')], check=True)
        ab.dump(cli.output / 'collector_audit.json', audit(cli.output / 'trace.sqlite'))
        subprocess.run([NSYS, 'stats', '--report=cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum',
            '--format=csv', '--output=' + str(cli.output / 'stats'), str(cli.output / 'trace.nsys-rep')], check=True)
    print(json.dumps(status), flush=True)
    return 0 if status['status'].startswith('captured') else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', default='flashinfer_cutlass', choices=['flashinfer_cutlass', 'flashinfer_cudnn'])
    raise SystemExit(asyncio.run(run(p.parse_args())))
