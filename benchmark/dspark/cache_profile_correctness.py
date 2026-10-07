"""Original cache-compatible baseline gate, before any paired-cache A/B.

New independent original references are explicitly authorized. The existing
deterministic reference file remains read-only. Publish a complete oracle only
after every cell passes cold serial references, serial repeats, and concurrency.
No inference candidate, hooks, cache reuse, graph, or performance conclusion.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab


ORIGINAL = "a61e0bb61f285a404f527d98190a9fe2610c553a"


def validate_profile(args, info):
    if args.deterministic_control_only or args.reference_verify_plans or args.enable_standard_graphs:
        raise ValueError("Only original non-deterministic graph-off full control is allowed")
    if info.get("enable_deterministic_inference") is not False or info.get("disable_cuda_graph") is not True:
        raise ab.SafetyFailure("Resolved original baseline configuration mismatch")
    if info.get("disable_radix_cache") is not False:
        raise ab.SafetyFailure("Original cache-compatible radix structure configuration changed")


def validate_row(row, spec, output_tokens):
    if row["cached_tokens"] != 0:
        raise ab.SafetyFailure("Full control unexpectedly reused prefix cache")
    if (row["prompt_tokens"] != spec["prompt_length"] or row["completion_tokens"] != output_tokens
            or len(row["output_ids"]) != output_tokens):
        raise ab.SafetyFailure("Requested/observed prompt or output length differs")
    ab.check_reference(row, row)  # Every required acceptance field must exist.


def publish_complete(root, dataset, references, completed_cells):
    expected = {(cell, v) for cell, spec in dataset.items() for v in range(spec["concurrency"])}
    if completed_cells != len(dataset) or references.keys() != expected:
        raise ValueError("Incomplete/unstable matrix cannot publish a usable oracle")
    path = root / "references.jsonl"
    if path.exists():
        raise ValueError("Refusing to overwrite an oracle")
    for cell, spec in dataset.items():
        for variant in range(spec["concurrency"]):
            ab.append_jsonl(path, references[cell, variant])
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main():
    import aiohttp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    cli = parser.parse_args()
    if cli.repeats < 3:
        raise ValueError("Baseline gate requires at least three serial/concurrent rounds")
    previous = json.loads((cli.control_run / "manifest.json").read_text())
    args = SimpleNamespace(**previous["args"])
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    args.enable_standard_graphs = False
    args.output = str(cli.output)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=args.repo, text=True).strip()
    if head != ORIGINAL or dirty:
        raise ValueError("Baseline must run original a61e0bb with clean tracked files")
    deterministic_oracle = Path(args.oracle_jsonl)
    oracle_before = hashlib.sha256(deterministic_oracle.read_bytes()).hexdigest()
    dataset = json.loads(deterministic_oracle.with_name("dataset.json").read_text())
    ab.load_frozen_references(str(deterministic_oracle), dataset)  # Complete immutable provenance.
    control = [json.loads(line) for line in (cli.control_run / "requests.jsonl").read_text().splitlines()]
    if len(control) != 1 or control[0]["cell"] != "p128_c1" or control[0]["variant"] != 0:
        raise ValueError("Requires the preserved single original h19 profile control")
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / "dataset.json", dataset)
    manifest = {"args": vars(args), "head": head, "repeats": cli.repeats,
        "reference_policy": "human-authorized independent original cache-compatible control; deterministic oracle preserved",
        "deterministic_oracle_read_only": str(deterministic_oracle), "deterministic_oracle_sha256": oracle_before,
        "saved_profile_control_sha256": hashlib.sha256((cli.control_run / "requests.jsonl").read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(ab.__file__))},
        "http_connection_policy": "fresh connection per request, no generation retries",
        "cache_reuse": False, "inference_candidate": False, "performance_claim": "unproven",
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "flashinfer-python", "aiohttp", "transformers")},
        "started_at": time.time()}
    ab.dump(cli.output / "manifest.json", manifest)
    server = ab.ManagedServer(args, "full", cli.output)
    status = {"status": "running", "completed_cells": 0, "original_references": 0,
              "performance_claim": "unproven", "paired_started": False}
    references = {}
    base = f"http://127.0.0.1:{args.port}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                        connector=ab.http_connector(aiohttp)) as session:
            server.start()
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, "/model_info")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() > deadline:
                        raise ab.SafetyFailure("Original baseline startup timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(cli.output / "server_info.json", info)
            validate_profile(args, info)
            print("READY original cache-compatible full control; cache reuse OFF", flush=True)
            for _ in range(args.warmup_requests):
                warmup = await ab.generate(session, base, dataset["p512_c1"]["prompts"][0], args.output_tokens, args.seed)
                validate_row(warmup, dataset["p512_c1"], args.output_tokens)
                server.check()

            async def request(cell, spec, variant, phase, repeat):
                server.check()
                row = await ab.generate(session, base, spec["prompts"][variant], args.output_tokens, args.seed)
                row.update(cell=cell, variant=variant, phase=phase, repeat=repeat, mode="full",
                           concurrency=spec["concurrency"], original_revision=head)
                ab.append_jsonl(cli.output / "requests.jsonl", row)
                validate_row(row, spec, args.output_tokens)
                server.check()
                if phase == "reference":
                    if (cell, variant) == ("p128_c1", 0):
                        ab.check_reference(control[0], row)
                    references[cell, variant] = row
                    status["original_references"] += 1
                else:
                    ab.check_reference(references[cell, variant], row)

            for cell, spec in dataset.items():
                status["active_cell"] = cell
                for variant in range(spec["concurrency"]):
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    await request(cell, spec, variant, "reference", -1)
                for repeat in range(cli.repeats):
                    status.update(active_repeat=repeat, active_phase="serial_repeat")
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    for variant in range(spec["concurrency"]):
                        await request(cell, spec, variant, "serial_repeat", repeat)
                    status["active_phase"] = "concurrent"
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    tasks = [asyncio.create_task(request(cell, spec, v, "concurrent", repeat))
                             for v in range(spec["concurrency"])]
                    try:
                        await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    print(f"PASS {cell} round={repeat} tokens and acceptance", flush=True)
                status["completed_cells"] += 1
                ab.append_jsonl(cli.output / "cells.jsonl", {"cell": cell, "repeats": cli.repeats,
                    "status": "original_profile_correctness_passed_for_this_cell", "output_tokens": args.output_tokens})
            status["published_oracle_sha256"] = publish_complete(cli.output, dataset, references, status["completed_cells"])
            status["status"] = "original_cache_profile_baseline_gate_passed"
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
        print("SAFETY STOP: " + str(exc), flush=True)
    finally:
        server.stop()
        status.update(managed_service_stopped=True, finished_at=time.time(),
            deterministic_oracle_unchanged=hashlib.sha256(deterministic_oracle.read_bytes()).hexdigest() == oracle_before)
        ab.dump(cli.output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "original_cache_profile_baseline_gate_passed" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
