#!/usr/bin/env python3
"""Fail-fast batch-invariance isolation; not a cache performance benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time

import dspark_prefix_ab as ab


class DiagnosticServer(ab.ManagedServer):
    def command(self):
        cmd, env = super().command()
        a = self.args
        if a.profile.startswith("target-"):
            cleaned = []
            i = 0
            while i < len(cmd):
                if cmd[i].startswith("--speculative-"):
                    i += 2
                else:
                    cleaned.append(cmd[i])
                    i += 1
            cmd = cleaned
        backend = "fa4" if a.profile.endswith("fa4") else "flashinfer"
        cmd[cmd.index("--attention-backend") + 1] = backend
        cmd += ["--disable-radix-cache"]
        return cmd, env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--sps-table")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cell", default="p2048_c8")
    parser.add_argument("--profile", choices=("target-fi", "target-fa4", "dspark-fi", "dspark-fa4"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=30117)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--mem-fraction", type=float, default=.55)
    parser.add_argument("--cache-budget-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.output_tokens < 1:
        parser.error("repeats and output-tokens must be positive")
    args.enable_standard_graphs = False
    args.deterministic_control_only = True
    return args


async def run(args):
    import aiohttp

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    raw_dataset = Path(args.dataset).read_bytes()
    spec = json.loads(raw_dataset)[args.cell]
    ab.dump(root / "dataset.json", {args.cell: spec})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    ab.dump(root / "manifest.json", {"args": vars(args), "revision": revision,
            "source_dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
            "purpose": "batch invariance isolation, no performance claim"})
    server = DiagnosticServer(args, "full", root)
    base = f"http://127.0.0.1:{args.port}"
    status = {"status": "running", "profile": args.profile, "cell": args.cell,
              "matched_serial_repeats": 0, "matched_concurrent_repeats": 0}
    references = {}
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            server.start()
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, "/model_info")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() > deadline:
                        raise ab.SafetyFailure("Server readiness timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(root / "server_info.json", info)
            for flag in ("enable_deterministic_inference", "disable_cuda_graph", "disable_radix_cache"):
                if not info.get(flag):
                    raise ab.SafetyFailure(f"Required resolved flag is false: {flag}")
            print(f"READY profile={args.profile}", flush=True)

            async def request(variant, phase, repeat):
                server.check()
                row = await ab.generate(session, base, spec["prompts"][variant], args.output_tokens, args.seed)
                row.update(cell=args.cell, variant=variant, phase=phase, repeat=repeat, profile=args.profile)
                ab.append_jsonl(root / "requests.jsonl", row)
                if row["cached_tokens"] != 0:
                    raise ab.SafetyFailure("Cache-off diagnostic unexpectedly reused prefix tokens")
                if phase == "reference":
                    references[variant] = row
                else:
                    ab.check_reference(references[variant], row,
                                       check_acceptance=args.profile.startswith("dspark-"))
                return row

            # Separate warmup from evidence, then independent serial cold references.
            await ab.generate(session, base, spec["prompts"][0], args.output_tokens, args.seed)
            for variant in range(spec["concurrency"]):
                await request(variant, "reference", -1)
            for repeat in range(args.repeats):
                for variant in range(spec["concurrency"]):
                    await request(variant, "serial_repeat", repeat)
                status["matched_serial_repeats"] += 1
                tasks = [asyncio.create_task(request(v, "concurrent", repeat))
                         for v in range(spec["concurrency"])]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                status["matched_concurrent_repeats"] += 1
                print(f"PASS cell={args.cell} repeat={repeat} serial/concurrent tokens match", flush=True)
            server.check()
        status["status"] = "correctness_passed_for_this_case_only"
    except Exception as exc:
        status.update(status="safety_stopped", error=str(exc), error_type=type(exc).__name__)
        print(f"SAFETY STOP: {exc}", flush=True)
    finally:
        server.stop()
        ab.dump(root / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "correctness_passed_for_this_case_only" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
