"""Two-request nsys attribution, fixed oracle and process-local candidate only."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import time

import aiohttp
import dspark_prefix_ab as ab
from stable_baseline_gate import validate_row, sha
from stable_cache_validation import QualifiedServer, start
from steady_tpot import integrity, DeviceTelemetry, drain_listener


class NormServer(QualifiedServer):
    def __init__(self, args, mode, folder, *, diagnostic=False, candidate=False, nsys=None):
        super().__init__(args, 'full', folder)
        self.diagnostic, self.candidate, self.nsys = diagnostic, candidate, nsys

    def command(self):
        cmd, env = super().command()
        env['SGLANG_DSPARK_RESIDUAL_RMSNORM_FUSION'] = '1' if self.candidate else '0'
        env['SGLANG_DSPARK_TPOT_TRACE'] = '1' if self.diagnostic else '0'
        env['PYTHONPATH'] = str(Path(__file__).parent) + ':' + env.get('PYTHONPATH', '')
        if self.diagnostic or self.candidate:
            i = cmd.index('--forward-hooks') + 1
            hooks = json.loads(cmd[i])
            hooks[0]['hook_factory'] = 'residual_rmsnorm_hook:make_hook'
            hooks[0]['config'].update(norm_diagnostic=self.diagnostic,
                norm_policy_folder=str(self.folder / 'norm_policy'),
                diagnostic_folder=str(self.folder), nsys=bool(self.nsys))
            cmd[i] = json.dumps(hooks)
        if self.nsys:
            cmd = [self.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none',
                   '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
                   '--capture-range-end=stop', '--kill=none', '--wait=all',
                   '--flush-on-cudaprofilerstop=true', '--cuda-event-trace=false',
                   '--output=' + str(self.folder / 'trace'), *cmd]
        return cmd, env


async def run(cli):
    source = json.loads((cli.inherit / 'manifest.json').read_text())
    if not source.get('execution_integrity_template'):
        raise ValueError('Requires audited current-instance h47 identity')
    template = source['execution_integrity_template']
    args = SimpleNamespace(**template['args'])
    args.observe_work = False
    args.startup_timeout = min(args.startup_timeout, 600)
    dataset = json.loads((Path(args.oracle_jsonl).parent / 'dataset.json').read_text())
    refs = ab.load_frozen_references(Path(args.oracle_jsonl), dataset)
    integrity(template, args)
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / 'manifest.json', {'scope': 'profiled kernel attribution ONLY',
        'candidate': cli.candidate, 'created_at': time.time(), 'source': str(cli.inherit),
        'source_sha': sha(cli.inherit / 'manifest.json'), 'execution_integrity_template': template,
        'tool_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'experiment_worktree_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent, text=True).strip(),
        'nsys': cli.nsys, 'requests': 2, 'settle': 24, 'performance': 'UNPROVEN'})
    folder = cli.output / '00_full'
    folder.mkdir()
    args.output = str(folder)
    server = NormServer(args, 'full', folder, diagnostic=True, candidate=cli.candidate, nsys=cli.nsys)
    telemetry = DeviceTelemetry(folder)
    status = {'status': 'running'}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
            connector=ab.http_connector(aiohttp)) as session:
            telemetry.begin()
            base = await start(server, session)
            async def checked(phase, ordinal):
                row = await ab.generate(session, base, dataset['p2048_c1']['prompts'][0], 128, args.seed)
                row.update(cell='p2048_c1', variant=0, phase=phase, ordinal=ordinal, mode='full')
                ab.append_jsonl(folder / 'requests.jsonl', row)
                validate_row(row, dataset['p2048_c1'], 128, ordinary=False, require_cold=True)
                ab.check_reference(refs[('p2048_c1', 0)], row)
                server.check()
            await checked('prime_excluded', 0)
            for i in range(24):
                await checked('settle_excluded', i)
            ab.dump(folder / 'trace_start', {'time': time.time()})
            for i in range(2):
                await checked('diagnostic_profiled', i)
            ab.dump(folder / 'trace_stop', {'time': time.time()})
            await checked('drain_excluded', 0)
        status['status'] = 'diagnostic_complete_NOT_performance_proof'
    except asyncio.CancelledError:
        status.update(status='safety_stopped', error_type='CancelledError', error='Wall budget/supervisor stop')
    except Exception as error:
        status.update(status='safety_stopped', error_type=type(error).__name__, error=str(error))
    finally:
        server.stop()
        telemetry.stop()
        await drain_listener(args.port)
        integrity(template, args)
        ab.dump(cli.output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if status['status'].startswith('diagnostic_complete') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nsys', required=True)
    parser.add_argument('--candidate', action='store_true')
    async def bounded(cli):
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await asyncio.wait_for(run(cli), timeout=1200)
    raise SystemExit(asyncio.run(bounded(parser.parse_args())))
