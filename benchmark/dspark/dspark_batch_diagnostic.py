#!/usr/bin/env python3
"""Fail-fast batch-invariance isolation; not a cache performance benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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
        backend = ("fa4" if a.profile.endswith("fa4") else
                   "triton" if a.profile.endswith("triton") else "flashinfer")
        cmd[cmd.index("--attention-backend") + 1] = backend
        cmd += ["--disable-radix-cache"]
        if getattr(a, "verify_trace_output_index", None) is not None:
            from dspark_verify_trace_hook import prompt_hash
            spec = json.loads(Path(a.dataset).read_text(encoding="utf-8"))[a.cell]
            refs = load_oracle(a.oracle_jsonl, a.cell, spec["concurrency"])
            prompt = spec["prompts"][a.verify_trace_variant]
            index = a.verify_trace_output_index
            ref_ids = refs[a.verify_trace_variant]["output_ids"]
            if index >= len(ref_ids):
                raise ValueError("Trace output index exceeds independent reference")
            config = {"folder": str(self.folder / "verify_trace"),
                      "prompt_sha256": prompt_hash(prompt), "prompt_length": len(prompt),
                      "position": len(prompt) + index - 1, "anchor_token": ref_ids[index - 1],
                      "max_captures": a.verify_trace_max_captures}
            modules = ["logits_processor"]
            if getattr(a, "verify_trace_upstream", False):
                modules += ["model.layers.0", "model.layers.19", "model.layers.39",
                            "model.layers.0.self_attn.attn"]
                config.update(kv_layer=0, max_kv_prefix=4096)
            cmd += ["--forward-hooks", json.dumps([{
                "name": "verify-boundary-read-only", "target_modules": modules,
                "hook_factory": "dspark_verify_trace_hook:make_hook", "config": config}])]
            env["PYTHONPATH"] += os.pathsep + str(Path(__file__).resolve().parent)
        return cmd, env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--sps-table")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle-jsonl", help="Optional independently saved ordinary-target references")
    parser.add_argument("--cell", default="p2048_c8")
    parser.add_argument("--profile", choices=("target-fi", "target-fa4", "target-triton",
                                             "dspark-fi", "dspark-fa4", "dspark-triton"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=30117)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--logprob-topk", type=int, default=0,
                        help="Optional diagnostics only; changes returned observability, not gates")
    parser.add_argument("--verify-trace-output-index", type=int,
                        help="Opt-in read-only eager final-hidden/raw-logit trace; zero-based index >= 1")
    parser.add_argument("--verify-trace-variant", type=int, default=3)
    parser.add_argument("--verify-trace-max-captures", type=int, default=8)
    parser.add_argument("--verify-trace-upstream", action="store_true",
                        help="Diagnostic only: layer 0 attention/past KV and layers 0/19/39 post output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--mem-fraction", type=float, default=.55)
    parser.add_argument("--cache-budget-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.output_tokens < 1 or args.logprob_topk < 0:
        parser.error("repeats/output-tokens must be positive; logprob-topk cannot be negative")
    if args.verify_trace_output_index is not None and (
            args.verify_trace_output_index < 1 or args.verify_trace_variant < 0
            or args.verify_trace_max_captures < 1 or not args.oracle_jsonl):
        parser.error("Trace requires independent oracle, output index >= 1, variant >= 0 and positive capture limit")
    if args.verify_trace_upstream and (args.verify_trace_output_index is None or args.verify_trace_max_captures < 15):
        parser.error("Upstream trace requires focused trace and >= 15 captures for three five-stage passes")
    args.enable_standard_graphs = False
    args.deterministic_control_only = True
    return args


def diagnostic_body(prompt, args):
    body = ab.request_body(prompt, args.output_tokens, args.seed)
    if args.logprob_topk:
        body.update(return_logprob=True, top_logprobs_num=args.logprob_topk,
                    return_text_in_logprobs=False)
    return body


def load_oracle(path, cell, concurrency):
    if not path:
        return {}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    refs = {}
    for row in rows:
        if row.get("cell") == cell and row.get("phase") == "reference":
            variant = row["variant"]
            if variant in refs:
                raise ValueError(f"Duplicate oracle variant: {variant}")
            refs[variant] = row
    if set(refs) != set(range(concurrency)):
        raise ValueError("Oracle must have exactly one reference for every variant in the selected cell")
    return refs


async def generate(session, base, prompt, args):
    start = time.perf_counter()
    tracker = ab.StreamTracker(start)
    async with session.post(base + "/generate", json=diagnostic_body(prompt, args)) as response:
        response.raise_for_status()
        async for frame in ab.sse_frames(response.content):
            if frame == "[DONE]":
                break
            tracker.add(json.loads(frame), time.perf_counter())
    row = tracker.result(time.perf_counter())
    if row["prompt_tokens"] != len(prompt) or row["completion_tokens"] != args.output_tokens:
        raise ab.SafetyFailure("Unexpected prompt or completion token count")
    if args.logprob_topk and len(row["meta_info"].get("output_top_logprobs", [])) != args.output_tokens:
        raise ab.SafetyFailure("Requested per-token top logprobs unavailable or incomplete")
    return row


async def run(args):
    import aiohttp

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    raw_dataset = Path(args.dataset).read_bytes()
    spec = json.loads(raw_dataset)[args.cell]
    if args.oracle_jsonl:
        oracle_dataset = Path(args.oracle_jsonl).parent / "dataset.json"
        if not oracle_dataset.exists() or json.loads(oracle_dataset.read_text(encoding="utf-8")).get(args.cell) != spec:
            raise ValueError("Independent oracle must have an adjacent dataset.json with identical prompts")
    oracle = load_oracle(args.oracle_jsonl, args.cell, spec["concurrency"])
    ab.dump(root / "dataset.json", {args.cell: spec})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo,
                              capture_output=True, text=True, check=True).stdout.strip()
    ab.dump(root / "manifest.json", {"args": vars(args), "revision": revision,
            "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_harness_source_sha256": hashlib.sha256(Path(ab.__file__).read_bytes()).hexdigest(),
            "source_dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
            "oracle_sha256": hashlib.sha256(Path(args.oracle_jsonl).read_bytes()).hexdigest() if oracle else None,
            "trace_hook_sha256": hashlib.sha256(Path(__file__).with_name("dspark_verify_trace_hook.py").read_bytes()).hexdigest()
                if args.verify_trace_output_index is not None else None,
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
                row = await generate(session, base, spec["prompts"][variant], args)
                row.update(cell=args.cell, variant=variant, phase=phase, repeat=repeat, profile=args.profile)
                ab.append_jsonl(root / "requests.jsonl", row)
                if row["cached_tokens"] != 0:
                    raise ab.SafetyFailure("Cache-off diagnostic unexpectedly reused prefix tokens")
                if oracle:
                    ab.check_reference(oracle[variant], row, check_acceptance=False)
                if phase == "reference":
                    references[variant] = row
                else:
                    ab.check_reference(references[variant], row,
                                       check_acceptance=args.profile.startswith("dspark-"))
                return row

            # Separate warmup from evidence, then independent serial cold references.
            await generate(session, base, spec["prompts"][0], args)
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
        if args.verify_trace_output_index is not None:
            status["trace_captures"] = len(list((root / "verify_trace").glob("*.json")))
            if not status["trace_captures"] and status["status"] != "safety_stopped":
                status.update(status="safety_stopped", error="No focused anchor trace captured",
                              error_type="TraceUnavailable")
        ab.dump(root / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "correctness_passed_for_this_case_only" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
