#!/usr/bin/env python3
"""Measure paired-prefix-cache benefit as the request reuse rate changes.

This is intentionally a small, fixed p512/c4 experiment.  It runs the same
prompt sequence with BF16 and graph-off in two server modes: paired radix cache
and full-prefill (radix disabled).  Only the fraction of measured requests that
reuse one of four cold reference prompts changes (0/25/50/100%).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import aiohttp

import dspark_prefix_ab as ab
from quant_high_conc_ab import make_cell


REPO = Path("/root/autodl-fs/dspark-quant-fp8-tools-20261009-h66")
TEMPLATE = Path("/root/autodl-tmp/dspark-quant-fp8-paired-high-conc-20261009-h74/bf16_a/launch.json")
MODEL = "/root/autodl-tmp/models/Qwen3-14B"
PORT = 30227
SEED = 20261007
PROMPT_LENGTH = 512
CONCURRENCY = 4
OUTPUT_TOKENS = 128
REQUESTS = 16
WARMUP_BATCHES = 2
RATES = (0, 25, 50, 100)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_launch(mode: str, folder: Path):
    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cmd = list(data["command"])
    env = dict(data["environment"])
    cmd[cmd.index("--port") + 1] = str(PORT)
    cmd[cmd.index("--max-running-requests") + 1] = str(CONCURRENCY)
    # The template is the already-attested deterministic graph-off paired
    # profile.  Repoint its runtime attestation folder to this run.
    hooks = cmd[cmd.index("--forward-hooks") + 1]
    hooks = hooks.replace("/root/autodl-tmp/dspark-quant-fp8-paired-high-conc-20261009-h74/bf16_a/runtime",
                         str(folder / "runtime"))
    cmd[cmd.index("--forward-hooks") + 1] = hooks
    has_disable = "--disable-radix-cache" in cmd
    if mode == "full" and not has_disable:
        cmd += ["--disable-radix-cache"]
    if mode == "paired" and has_disable:
        cmd = [x for x in cmd if x != "--disable-radix-cache"]
    env.update({
        # The runnable SGLang checkout is the shared source tree; this branch
        # contributes the benchmark/profile modules through the second entry.
        "PYTHONPATH": f"/root/autodl-tmp/src/sglang-main/python:{REPO}/python:{REPO}/benchmark/dspark",
        "SGLANG_DSPARK_PREFIX_CACHE": "1" if mode == "paired" else "0",
        "SGLANG_DSPARK_PREFIX_CACHE_SHADOW": "0",
        "SGLANG_DSPARK_DISABLE_RADIX_CACHE": "0" if mode == "paired" else "1",
        "SGLANG_DSPARK_RAGGED_VERIFY_GRAPH": "0",
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
        "SGLANG_DSPARK_COMPACT_SCAN_CUDA": "0",
        "SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH": "0",
        "SGLANG_RADIX_FORCE_MISS": "0",
    })
    return cmd, env


class OwnedServer:
    def __init__(self, mode: str, folder: Path):
        self.mode = mode
        self.folder = folder
        self.process = None
        self.log = None
        self.scanned = 0

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        cmd, env = build_launch(self.mode, self.folder)
        dump(self.folder / "launch.json", {"command": cmd, "environment": env})
        self.log = (self.folder / "server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def check(self):
        if self.process.poll() is not None:
            raise ab.SafetyFailure(f"server exited with {self.process.returncode}")
        if self.log:
            self.log.flush()
        text = (self.folder / "server.log").read_text(encoding="utf-8", errors="replace")
        text = text[self.scanned:]
        self.scanned += len(text)
        for marker in ab.ERROR_MARKERS:
            if marker.lower() in text.lower():
                raise ab.SafetyFailure(f"server log contains {marker}")

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
            except ProcessLookupError:
                pass
        if self.log:
            self.log.close()


async def wait_ready(server: OwnedServer, session: aiohttp.ClientSession, base: str):
    deadline = time.monotonic() + 900
    while True:
        server.check()
        try:
            await ab.api(session, base, "/model_info")
            info = await ab.api(session, base, "/server_info")
            return info
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if time.monotonic() >= deadline:
                raise ab.SafetyFailure("startup timeout")
            await asyncio.sleep(.5)


async def run_one(session, base: str, prompt, seed: int, phase: str, index: int):
    row = await ab.generate(session, base, prompt, OUTPUT_TOKENS, seed)
    row.update(phase=phase, request_index=index, concurrency=CONCURRENCY,
               prompt_sha256=__import__("hashlib").sha256(
                   json.dumps(prompt, separators=(",", ":")).encode()).hexdigest())
    return row


def hit_indices(rate: int, count: int):
    target = round(count * rate / 100)
    if target == 0:
        return set()
    if target == count:
        return set(range(count))
    # Evenly spread hits over the measured window, avoiding a front-loaded
    # cache advantage that would distort queueing.
    return {min(count - 1, (i * count) // target) for i in range(target)}


def make_prompt_sets(tokenizer):
    base = make_cell(tokenizer, PROMPT_LENGTH, CONCURRENCY, SEED)
    misses = []
    # Each mode executes four rates; at 0% every measured and warmup request is
    # unique.  Keep a generous deterministic pool so no request is accidentally
    # recycled into a cache hit.
    for i in range(len(RATES) * (REQUESTS + WARMUP_BATCHES * CONCURRENCY) + 32):
        spec = make_cell(tokenizer, PROMPT_LENGTH, CONCURRENCY, SEED + 1000 + i)
        misses.append(spec["prompts"][i % CONCURRENCY])
    return base["prompts"], misses


async def run_mode(mode: str, root: Path, tokenizer, base_prompts, miss_prompts):
    folder = root / mode
    server = OwnedServer(mode, folder)
    sampler = ab.GPUSampler(folder / "gpu.jsonl", 100)
    base_url = f"http://127.0.0.1:{PORT}"
    timeout = aiohttp.ClientTimeout(total=300)
    output = []
    server.start()
    sampler.start()
    try:
        async with aiohttp.ClientSession(connector=ab.http_connector(aiohttp), timeout=timeout) as session:
            info = await wait_ready(server, session, base_url)
            dump(folder / "server_info.json", info)
            expected_disabled = mode == "full"
            if bool(info.get("disable_radix_cache")) != expected_disabled:
                raise ab.SafetyFailure(f"disable_radix_cache mismatch: {info.get('disable_radix_cache')}")
            if mode == "paired" and ab.info_stats(info, required=True) is None:
                raise ab.SafetyFailure("paired sidecar stats unavailable")

            miss_cursor = 0
            for rate in RATES:
                await ab.api(session, base_url, "/flush_cache?timeout=30", "POST")
                # Cold references establish four exact prefixes for the hit side.
                for variant, prompt in enumerate(base_prompts):
                    await run_one(session, base_url, prompt, SEED, "reference", variant)

                hit = hit_indices(rate, REQUESTS)
                warm_rows = []
                for warm in range(WARMUP_BATCHES):
                    tasks = []
                    for i in range(CONCURRENCY):
                        if i == 0 and rate:
                            prompt = base_prompts[(warm + i) % CONCURRENCY]
                        else:
                            prompt = miss_prompts[miss_cursor]
                            miss_cursor += 1
                        tasks.append(asyncio.create_task(
                            run_one(session, base_url, prompt, SEED, "warmup", warm * CONCURRENCY + i)))
                    warm_rows.extend(await asyncio.gather(*tasks))

                before = ab.info_stats(await ab.api(session, base_url, "/server_info"), required=(mode == "paired"))
                sample_start = sampler.begin(f"reuse_{rate}")
                started = time.perf_counter()
                tasks = []
                measured_prompts = []
                for i in range(REQUESTS):
                    if i in hit:
                        prompt = base_prompts[i % CONCURRENCY]
                    else:
                        prompt = miss_prompts[miss_cursor]
                        miss_cursor += 1
                    measured_prompts.append(prompt)
                    tasks.append(asyncio.create_task(
                        run_one(session, base_url, prompt, SEED, "steady", i)))
                rows = await asyncio.gather(*tasks)
                elapsed = time.perf_counter() - started
                after = ab.info_stats(await ab.api(session, base_url, "/server_info"), required=(mode == "paired"))
                summary = ab.aggregate(mode, f"p{PROMPT_LENGTH}_c{CONCURRENCY}_reuse{rate}", 0,
                                       rows, before, after, sampler.summary(sample_start), elapsed)
                summary.update(reuse_rate=rate, mode=mode, warmup_requests=len(warm_rows),
                               hit_request_count=len(hit), miss_request_count=REQUESTS-len(hit))
                output.append(summary)
                dump(folder / f"reuse_{rate}.json", {"summary": summary, "rows": rows})
                print(json.dumps({"mode": mode, "reuse_rate": rate,
                                  "e2e_mean_ms": summary["e2e_ms"]["mean"],
                                  "e2e_p95_ms": summary["e2e_ms"]["p95"],
                                  "ttft_p95_ms": summary["ttft_ms"]["p95"],
                                  "tpot_p95_ms": summary["tpot_ms"]["p95"],
                                  "cache": summary["cache"],
                                  "acceptance": summary["acceptance"]["draft_accept_rate"]},
                                 sort_keys=True), flush=True)
    finally:
        server.stop()
        sampler.stop()
    return output


async def main_async(root: Path):
    root.mkdir(parents=True, exist_ok=False)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, local_files_only=True)
    base_prompts, miss_prompts = make_prompt_sets(tokenizer)
    dump(root / "protocol.json", {"prompt_length": PROMPT_LENGTH, "concurrency": CONCURRENCY,
                                  "requests": REQUESTS, "warmup_batches": WARMUP_BATCHES,
                                  "reuse_rates": RATES, "graph": "off", "dtype": "bfloat16"})
    result = {}
    # Full-prefill first is the conservative control; paired cache then uses a
    # fresh process, preventing allocator/cache state from crossing arms.
    for mode in ("full", "paired"):
        result[mode] = await run_mode(mode, root, tokenizer, base_prompts, miss_prompts)
    dump(root / "result.json", result)
    lines = ["# Paired-cache prefix reuse-rate sweep (BF16, graph off)", "",
             "Same p512/c4 prompts and request sequence in full-prefill and paired-cache modes.", "",
             "| Reuse | Full E2E mean ms | Paired E2E mean ms | Mean reduction | Full p95 ms | Paired p95 ms | Paired target hit | Paired paired hit | Partial recovery | Acceptance |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    full = {x["reuse_rate"]: x for x in result["full"]}
    paired = {x["reuse_rate"]: x for x in result["paired"]}
    for rate in RATES:
        f, p = full[rate], paired[rate]
        reduction = 1 - p["e2e_ms"]["mean"] / f["e2e_ms"]["mean"]
        cache = p["cache"]
        lines.append(f"| {rate}% | {f['e2e_ms']['mean']:.2f} | {p['e2e_ms']['mean']:.2f} | {reduction*100:.2f}% | "
                     f"{f['e2e_ms']['p95']:.2f} | {p['e2e_ms']['p95']:.2f} | "
                     f"{cache.get('observed_target_token_hit_rate')} | {cache.get('observed_paired_token_hit_rate')} | "
                     f"{cache['counter_delta'].get('partial_recovery_calls')} | {p['acceptance']['draft_accept_rate']:.4f} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.output.resolve()))


if __name__ == "__main__":
    main()
