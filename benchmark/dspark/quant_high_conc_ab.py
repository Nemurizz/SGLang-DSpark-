"""Counterbalanced high-concurrency BF16 vs native W8A8 FP8 DSpark A/B.

This is an independent timing harness. It deliberately keeps graphs, radix
cache, compact scan, adaptive draft length and KV quantization disabled, so the
result isolates the target arithmetic path under queueing and batch pressure.
Outputs are compared only within the same arithmetic recipe; BF16 and FP8 are
not required to emit identical tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from quality_run import QualityServer, validate
from quant_feasibility_gate import launch_args_from_template


CAPACITY = 32768
OUTPUT_TOKENS = 128
REQUESTS = 32
WARMUP_BATCHES = 2
MAX_CONCURRENCY = 16
ORDER = ("bf16_a", "fp8_a", "fp8_b", "bf16_b")
CELLS = (
    (512, 4), (512, 8), (512, 16),
    (1024, 4), (1024, 8), (1024, 16),
    (2048, 4), (2048, 8),
)


class HighConcServer(QualityServer):
    def __init__(self, args, mode, folder, max_running=MAX_CONCURRENCY):
        super().__init__(args, mode, folder)
        self.max_running = max_running

    def command(self):
        cmd, env = super().command()
        marker = cmd.index("--max-running-requests")
        cmd[marker + 1] = str(self.max_running)
        cmd += ["--max-total-tokens", str(CAPACITY)]
        return cmd, env


def make_cell(tokenizer, length: int, concurrency: int, seed: int) -> dict:
    passages = (
        "A speculative decoder proposes tokens with a small draft model and verifies them with the target. "
        "Prefix reuse is valid only when both models have compatible cached key and value tensors. "
        "Readiness metadata must survive node splits, but must be invalidated after weight changes. "
        "A cache budget may evict draft readiness without evicting the corresponding target prefix. "
        "In that case, the missing suffix must be recomputed from target hidden states. "
        "Measure first-token latency separately from decode latency and report the slowest requests. "
        "Concurrency affects queueing, batch geometry, acceptance, and memory consumption. "
        "A faster isolated kernel does not establish a faster end-to-end serving workload.\n"
    )
    body = tokenizer.encode(passages * 1000, add_special_tokens=False)
    cell = f"p{length}_c{concurrency}"
    head = tokenizer.encode(
        f"Benchmark case {seed}-{cell}. Read this technical document:\n",
        add_special_tokens=False,
    )
    topics = (
        "cache validity", "partial recovery", "latency measurement", "batch scheduling",
        "memory budget", "token verification", "safe fallback", "experimental evidence",
        "queue pressure", "capacity boundaries", "batch invariance", "deployment safety",
        "acceptance stability", "tail latency", "arithmetic precision", "service isolation",
    )
    suffixes = [tokenizer.encode(
        f"\nTask {v}: summarize the document and explain {topics[v]}. "
        "Use clear technical sentences.\nAnswer:", add_special_tokens=False)
        for v in range(concurrency)
    ]
    tail_len = max(map(len, suffixes))
    padding_id = tokenizer.encode(" ", add_special_tokens=False)
    if len(padding_id) != 1:
        raise ValueError("Tokenizer must encode a space as one token")
    prefix_len = length - tail_len
    if prefix_len < len(head):
        raise ValueError(f"Prompt too short for {cell}")
    prefix = head + body[:prefix_len - len(head)]
    prompts = [prefix + tail + padding_id * (tail_len - len(tail)) for tail in suffixes]
    if not all(len(p) == length for p in prompts):
        raise ValueError(f"Prompt length mismatch for {cell}")
    if len({tuple(p) for p in prompts}) != concurrency:
        raise ValueError(f"Tokenizer collapsed suffixes for {cell}")
    return {"cell": cell, "prompt_length": length, "concurrency": concurrency,
            "common_prefix_tokens": prefix_len, "prompts": prompts}


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path: Path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


async def drain_listener(port: int):
    for _ in range(30):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        await asyncio.sleep(.5)
    raise ab.SafetyFailure(f"Owned listener {port} did not drain")


async def gather_checked(tasks, server):
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=.5)
            server.check()
            for task in done:
                task.result()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def run_arm(args, root: Path, arm: str, dataset: dict,
                  references: dict, summaries: list[dict]):
    import aiohttp

    folder = root / arm
    folder.mkdir(parents=True)
    server = HighConcServer(args, "full", folder)
    sampler = ab.GPUSampler(folder / "gpu.jsonl", 100)
    base = f"http://127.0.0.1:{args.port}"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    local_refs = {}
    quant = arm.startswith("fp8")
    recipe = "w8a8_fp8_native" if quant else "bf16"
    try:
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True).strip()
        if occupied:
            raise ab.SafetyFailure("GPU occupied before arm: " + occupied)
        sampler.start()
        server.start()
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp), timeout=timeout) as session:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await ab.api(session, base, "/model_info")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() >= deadline:
                        raise ab.SafetyFailure(f"Startup timeout for {arm}")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(folder / "server_info.json", info)
            if info.get("max_total_tokens") != CAPACITY:
                raise ab.SafetyFailure(f"max_total_tokens not resolved: {info.get('max_total_tokens')}")
            if info.get("max_running_requests") not in (None, MAX_CONCURRENCY):
                raise ab.SafetyFailure(f"max_running_requests not resolved: {info.get('max_running_requests')}")

            async def one(cell, variant, phase, index):
                spec = dataset[cell]
                task = asyncio.create_task(ab.generate(session, base, spec["prompts"][variant], OUTPUT_TOKENS, args.seed))
                try:
                    while not task.done():
                        await asyncio.wait([task], timeout=.5)
                        server.check()
                    row = await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                row.update(arm=arm, recipe=recipe, quant=quant, cell=cell, variant=variant,
                           phase=phase, request_index=index, concurrency=spec["concurrency"])
                append(folder / "requests.jsonl", row)
                if row["cached_tokens"] or row["meta_info"].get("num_retractions", 0):
                    raise ab.SafetyFailure(f"cache/retraction in {arm} {cell}")
                key = (cell, variant)
                if phase == "reference":
                    if key in references and references[key]["recipe"] == recipe:
                        ab.check_reference(references[key], row, check_acceptance=True)
                    local_refs[key] = row
                else:
                    reference = references.get(key) or local_refs.get(key)
                    if reference is None or reference["recipe"] != recipe:
                        raise ab.SafetyFailure(f"missing same-recipe oracle for {arm} {cell} v{variant}")
                    ab.check_reference(reference, row, check_acceptance=True)
                return row

            # A cold reference per variant establishes the independent arithmetic oracle.
            for cell, spec in dataset.items():
                for variant in range(spec["concurrency"]):
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    await one(cell, variant, "reference", variant)
            # The profile hooks emit runtime geometry/weight attestations on the
            # first real forward, not during /server_info. Validate only after
            # the cold reference requests have materialized those files.
            validate(folder, quant, False, "sgl_kernel_cutlass_fp8" if quant else "flashinfer_cutlass", recipe)

            for cell, spec in dataset.items():
                c = spec["concurrency"]
                for warm in range(WARMUP_BATCHES):
                    tasks = [asyncio.create_task(one(cell, i % c, "warmup", warm * c + i)) for i in range(c)]
                    await gather_checked(tasks, server)
                sample_start = sampler.begin(f"{cell}_steady")
                before = ab.info_stats(await ab.api(session, base, "/server_info"), required=False)
                began = time.perf_counter()
                tasks = [asyncio.create_task(one(cell, i % c, "steady", i)) for i in range(REQUESTS)]
                await gather_checked(tasks, server)
                elapsed = time.perf_counter() - began
                rows = [task.result() for task in tasks]
                after = ab.info_stats(await ab.api(session, base, "/server_info"), required=False)
                summary = ab.aggregate("full", cell, 0, rows, before, after,
                                       sampler.summary(sample_start), elapsed)
                summary.update(arm=arm, recipe=recipe, quant=quant,
                               prompt_length=spec["prompt_length"], concurrency=c,
                               capacity=CAPACITY, requests=REQUESTS,
                               throughput_tokens_per_s=summary["client_observed_output_tokens_per_s"])
                append(root / "cells.jsonl", summary)
                summaries.append(summary)
                print(json.dumps({"arm": arm, "cell": cell, "e2e_p50_ms": summary["e2e_ms"]["p50"],
                                  "e2e_p95_ms": summary["e2e_ms"]["p95"],
                                  "ttft_p95_ms": summary["ttft_ms"]["p95"],
                                  "tpot_p95_ms": summary["tpot_ms"]["p95"],
                                  "peak_mib": summary["gpu"]["peak_device_memory_used_mib"],
                                  "accept": summary["acceptance"]["draft_accept_rate"]}, sort_keys=True), flush=True)
    finally:
        try:
            server.stop()
        finally:
            sampler.stop()
            await drain_listener(args.port)


def comparison_rows(summaries):
    by = {(s["arm"], s["cell"]): s for s in summaries}
    output = []
    for cell in sorted({s["cell"] for s in summaries}):
        for order, control_arm, cand_arm in (("AB", "bf16_a", "fp8_a"), ("BA", "bf16_b", "fp8_b")):
            control, candidate = by.get((control_arm, cell)), by.get((cand_arm, cell))
            if not control or not candidate:
                continue
            ratios = {metric: {stat: candidate[metric][stat] / control[metric][stat]
                               for stat in ("mean", "p95")}
                      for metric in ("ttft_ms", "tpot_ms", "e2e_ms")}
            row = {"cell": cell, "order": order, "prompt_length": control["prompt_length"],
                   "concurrency": control["concurrency"], "recipe": "w8a8_fp8_native",
                   "e2e_gain_mean": 1 - ratios["e2e_ms"]["mean"],
                   "e2e_gain_p95": 1 - ratios["e2e_ms"]["p95"],
                   "p95_regression": any(ratios[k]["p95"] > 1 for k in ratios),
                   "ratios": ratios,
                   "bf16_peak_mib": control["gpu"]["peak_device_memory_used_mib"],
                   "fp8_peak_mib": candidate["gpu"]["peak_device_memory_used_mib"],
                   "bf16_acceptance": control["acceptance"]["draft_accept_rate"],
                   "fp8_acceptance": candidate["acceptance"]["draft_accept_rate"],
                   "bf16_throughput_tokens_per_s": control["throughput_tokens_per_s"],
                   "fp8_throughput_tokens_per_s": candidate["throughput_tokens_per_s"]}
            row["passed_1pct_and_p95"] = row["e2e_gain_mean"] > .01 and not row["p95_regression"]
            output.append(row)
    return output


def write_report(root, summaries, comparisons, status):
    dump(root / "comparisons.json", comparisons)
    dump(root / "status.json", status)
    lines = ["# Native W8A8 FP8 high-concurrency DSpark A/B", "",
             f"Status: {status['status']}", "",
             "Protocol: graphs/radix cache/compact scan/adaptive draft/KV quantization OFF; max_total_tokens=32768; output=128; 32 measured requests per cell; BF16→FP8→FP8→BF16.",
             "Correctness is exact within each recipe (tokens plus five DSpark acceptance fields). BF16/FP8 cross-recipe token equality is not required.", "",
             "| Cell | Order | E2E gain mean | E2E gain p95 | TTFT p95 ratio | TPOT p95 ratio | BF16/FP8 peak MiB | BF16/FP8 acceptance | Pass |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in comparisons:
        ratio = r["ratios"]
        lines.append(f"| {r['cell']} | {r['order']} | {r['e2e_gain_mean']*100:.2f}% | {r['e2e_gain_p95']*100:.2f}% | {ratio['ttft_ms']['p95']:.3f} | {ratio['tpot_ms']['p95']:.3f} | {r['bf16_peak_mib']} / {r['fp8_peak_mib']} | {r['bf16_acceptance']:.3f} / {r['fp8_acceptance']:.3f} | {'PASS' if r['passed_1pct_and_p95'] else 'FAIL'} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(cli):
    root = cli.output.resolve()
    if root.exists():
        raise ValueError(f"Fresh output folder required: {root}")
    root.mkdir(parents=True)
    template = json.loads(cli.template.read_text())
    args = launch_args_from_template(template)
    args.tokenizer = args.model
    args.output_tokens = OUTPUT_TOKENS
    args.seed = cli.seed
    args.port = cli.port
    args.mem_fraction = cli.mem_fraction
    args.cache_budget_fraction = .25
    args.startup_timeout = 900
    args.request_timeout = 300
    args.enable_standard_graphs = False
    args.deterministic_control_only = False
    args.reference_verify_plans = False
    args.ordinary = False
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    dataset = {f"p{length}_c{c}": make_cell(tokenizer, length, c, cli.seed) for length, c in CELLS}
    dump(root / "dataset.json", dataset)
    dump(root / "protocol.json", {"capacity": CAPACITY, "output_tokens": OUTPUT_TOKENS,
                                   "requests": REQUESTS, "warmup_batches": WARMUP_BATCHES,
                                   "order": ORDER, "cells": list(dataset), "graphs_cache_compact_adaptive_kv_off": True})
    status = {"status": "running", "started_at": time.time(), "completed_arms": []}
    dump(root / "status.json", status)
    summaries = []
    refs = {}
    try:
        for ordinal, arm in enumerate(ORDER):
            args.quant = arm.startswith("fp8")
            args.recipe = "w8a8_fp8_native" if args.quant else "bf16"
            await run_arm(args, root, arm, dataset, refs, summaries)
            status["completed_arms"].append(arm)
            dump(root / "status.json", status)
        comps = comparison_rows(summaries)
        passed_cells = {r["cell"] for r in comps if r["passed_1pct_and_p95"]}
        status.update(status="completed", completed_at=time.time(),
                      comparison_count=len(comps), passed_orders=sum(r["passed_1pct_and_p95"] for r in comps),
                      passed_cells=sorted(passed_cells),
                      performance_claim="descriptive high-concurrency matrix; per-cell pass requires >1% E2E mean gain in both orders and no p95 regression")
    except Exception as exc:
        comps = comparison_rows(summaries)
        status.update(status="safety_stopped", error=f"{type(exc).__name__}: {exc}", stopped_at=time.time())
    finally:
        comps = comparison_rows(summaries)
        write_report(root, summaries, comps, status)
    return 0 if status["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--port", type=int, default=30119)
    parser.add_argument("--mem-fraction", type=float, default=.55)
    cli = parser.parse_args()
    raise SystemExit(asyncio.run(run(cli)))


if __name__ == "__main__":
    main()
