#!/usr/bin/env python3
"""Fixed DSpark cache A/B protocol. No inference-path mutations or shell jobs.

Run from the repository root with the same Python environment as the server.
Optional target_only_shadow intentionally exercises unsafe target-only reuse.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

LENGTHS = (128, 512, 2048, 8192)
CONCURRENCIES = (1, 4, 8)
COUNTERS = ("match_calls", "target_hit_tokens", "paired_hit_tokens",
            "metadata_evictions", "partial_recovery_calls", "partial_recovery_tokens",
            "full_recovery_calls", "full_recovery_tokens")
ACCEPTANCE_KEYS = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts",
                   "spec_accept_histogram", "spec_correct_drafts_histogram")
ERROR_MARKERS = ("Scheduler hit an exception", "illegal memory access",
                 "CUDA error:", "Traceback (most recent call last)", "Watchdog timeout")


class SafetyFailure(RuntimeError):
    pass


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values: list[float]) -> dict:
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "p99": percentile(values, .99)}


def counter_delta(before: dict, after: dict) -> dict:
    delta = {}
    for key in COUNTERS:
        if key not in before or key not in after:
            delta[key] = None
        else:
            delta[key] = after[key] - before[key]
            if delta[key] < 0:
                raise SafetyFailure(f"Cache counter reset during measured window: {key}")
    return delta


def cache_metrics(delta: dict, rows: list[dict], active: bool) -> dict:
    matches = delta.get("match_calls")
    eligible = sum(max(0, r["prompt_tokens"] - 1) for r in rows)
    # Scheduler retries can cause multiple observations of the same request.
    valid_token_denominator = active and matches == len(rows) and eligible > 0
    target, paired = delta.get("target_hit_tokens"), delta.get("paired_hit_tokens")
    return {
        "request_target_hit_rate": sum(r["cached_tokens"] > 0 for r in rows) / len(rows),
        "actual_cached_token_rate": sum(r["cached_tokens"] for r in rows) / eligible,
        "observed_target_token_hit_rate": target / eligible if valid_token_denominator and target is not None else None,
        "observed_paired_token_hit_rate": paired / eligible if valid_token_denominator and paired is not None else None,
        "paired_over_target_token_coverage": paired / target if active and target and paired is not None else None,
        "paired_hit_length_mean_per_match": paired / matches if active and matches and paired is not None else None,
        "match_invocations_per_request": matches / len(rows) if active and matches is not None else None,
        "token_rate_denominator_valid": valid_token_denominator,
        "counter_delta": delta,
        "recovery_counts_are_plans_not_worker_fallbacks": True,
    }


def check_reference(reference: dict, row: dict, *, check_acceptance: bool = True) -> None:
    expected, actual = reference["output_ids"], row["output_ids"]
    if expected != actual:
        index = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                     min(len(expected), len(actual)))
        raise SafetyFailure(f"token mismatch cell={row['cell']} variant={row['variant']} "
                            f"phase={row['phase']} at token {index}; "
                            f"reference={expected[index:index+5]} actual={actual[index:index+5]}")
    if check_acceptance:
        for key in ACCEPTANCE_KEYS:
            left, right = reference["meta_info"].get(key), row["meta_info"].get(key)
            if left is None or right is None:
                raise SafetyFailure(f"Required acceptance metadata unavailable: {key}")
            if left != right:
                raise SafetyFailure(f"acceptance changed cell={row['cell']} variant={row['variant']} "
                                    f"phase={row['phase']} {key}: {left} != {right}")


def check_p95(control: dict, candidate: dict, min_samples: int, tolerance: float) -> str:
    if min(control["request_count"], candidate["request_count"]) < min_samples:
        return "insufficient_samples"
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        left, right = control[metric]["p95"], candidate[metric]["p95"]
        if left is not None and right is not None and right > left * (1 + tolerance):
            raise SafetyFailure(f"p95 regression {candidate['cell']} {metric}: full={left:.3f} paired={right:.3f}")
    return "passed"


class StreamTracker:
    """Cumulative output IDs, not text length; burst tokens are not fake ITLs."""
    def __init__(self, started: float):
        self.started = started
        self.ids: list[int] = []
        self.events: list[dict] = []
        self.meta: dict = {}
        self.first: float | None = None
        self.last: float | None = None
        self.finished = False

    def add(self, payload: dict, received: float) -> None:
        if "error" in payload:
            raise SafetyFailure(f"SSE error: {payload['error']}")
        if "output_ids" not in payload:
            raise SafetyFailure("Streaming output_ids unavailable; cannot validate tokens")
        ids = payload["output_ids"]
        if ids:
            if len(ids) < len(self.ids) or ids[:len(self.ids)] != self.ids:
                raise SafetyFailure("SSE is not cumulative or emitted output tokens changed")
            new = len(ids) - len(self.ids)
            if new:
                self.first = received if self.first is None else self.first
                self.last = received
                self.events.append({"elapsed_ms": (received - self.started) * 1000,
                                    "cumulative_tokens": len(ids), "new_tokens": new})
                self.ids = list(ids)
        self.meta.update(payload.get("meta_info", {}))
        self.finished = bool(self.meta.get("finish_reason"))

    def result(self, ended: float) -> dict:
        if not self.finished or self.first is None or not self.ids:
            raise SafetyFailure("SSE ended without a completed token-bearing response")
        n = len(self.ids)
        if self.meta.get("completion_tokens") != n:
            raise SafetyFailure("output_ids length != final completion_tokens")
        return {"output_ids": self.ids, "output_sha256": hashlib.sha256(
                    json.dumps(self.ids, separators=(",", ":")).encode()).hexdigest(),
                "meta_info": self.meta, "stream_events": self.events,
                "ttft_ms": (self.first - self.started) * 1000,
                "tpot_ms": (self.last - self.first) * 1000 / (n - 1) if n > 1 else None,
                "e2e_ms": (ended - self.started) * 1000,
                "prompt_tokens": self.meta["prompt_tokens"], "completion_tokens": n,
                "cached_tokens": self.meta.get("cached_tokens", 0)}


async def sse_frames(content):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    data = []
    async for chunk in content.iter_any():
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
    buffer += decoder.decode(b"", final=True)
    if buffer.startswith("data:"):
        data.append(buffer[5:].strip())
    if data:
        yield "\n".join(data)


class GPUSampler:
    """Device-wide nvidia-smi memory/utilization. Not process allocator peak."""
    def __init__(self, path: Path, interval_ms: int):
        self.path, self.interval_ms = path, interval_ms
        self.rows: list[dict] = []
        self.scope = "startup"
        self.process = None
        self.thread = None
        self.error = None
        self.lock = threading.Lock()

    def start(self):
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.error = str(exc)

    def _read(self):
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                for line in self.process.stdout:
                    values = [x.strip() for x in line.split(",")]
                    if len(values) != 5:
                        continue
                    with self.lock:
                        row = {"monotonic_s": time.monotonic(), "wall_time_s": time.time(),
                               "scope": self.scope, "gpu": int(values[0]),
                               "memory_used_mib": int(values[1]), "memory_total_mib": int(values[2]),
                               "utilization_pct": int(values[3]), "temperature_c": int(values[4])}
                        self.rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
        except Exception as exc:
            self.error = str(exc)

    def begin(self, scope: str) -> int:
        with self.lock:
            self.scope = scope
            return len(self.rows)

    def summary(self, start: int) -> dict:
        with self.lock:
            rows = self.rows[start:].copy()
            self.scope = "between_cells"
        return {"sample_count": len(rows), "sample_interval_ms": self.interval_ms,
                "peak_device_memory_used_mib": max((r["memory_used_mib"] for r in rows), default=None),
                "mean_device_utilization_pct": sum(r["utilization_pct"] for r in rows) / len(rows) if rows else None,
                "peak_temperature_c": max((r["temperature_c"] for r in rows), default=None),
                "unavailable_reason": self.error if not rows else None}

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.thread:
                self.thread.join(timeout=5)


class ManagedServer:
    def __init__(self, args, mode: str, folder: Path):
        self.args, self.mode, self.folder = args, mode, folder
        self.process = None
        self.log_file = None
        self.scanned = 0

    def command(self) -> tuple[list[str], dict]:
        a = self.args
        env = os.environ.copy()
        # Do not inherit unrelated experimental switches from an interactive shell.
        for key in list(env):
            if key.startswith("SGLANG_DSPARK_") or key.startswith("SGLANG_RAGGED_"):
                del env[key]
        env.update({"PYTHONPATH": str(Path(a.repo) / "python"),
                    "SGLANG_DSPARK_PREFIX_CACHE": "1" if self.mode == "paired" else "0",
                    "SGLANG_DSPARK_PREFIX_CACHE_SHADOW": "1" if self.mode == "target_only_shadow" else "0",
                    "SGLANG_DSPARK_DISABLE_RADIX_CACHE": "1",
                    "SGLANG_DSPARK_PREFIX_CACHE_BUDGET_FRACTION": str(a.cache_budget_fraction),
                    "SGLANG_DSPARK_RAGGED_VERIFY_GRAPH": "0", "SGLANG_RAGGED_VERIFY_MODE": "compact",
                    "SGLANG_DSPARK_COMPACT_SCAN_CUDA": "0", "SGLANG_DSPARK_DYNAMIC_DRAFT_LENGTH": "0",
                    "SGLANG_DSPARK_VERIFY_SERIAL_REFERENCE": "1" if getattr(a, "reference_verify_plans", False) else "0",
                    "SGLANG_RADIX_FORCE_MISS": "0"})
        # JIT helpers (e.g. ninja) must use the same environment as the server.
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", a.model,
               "--tokenizer-path", a.model, "--trust-remote-code", "--host", "127.0.0.1",
               "--port", str(a.port), "--mem-fraction-static", str(a.mem_fraction),
               "--max-running-requests", "8", "--random-seed", str(a.seed),
               "--attention-backend", "flashinfer", "--decode-attention-backend", "fa4",
               "--sampling-backend", "pytorch", "--stream-interval", "1",
               "--speculative-algorithm", "DSPARK", "--speculative-draft-model-path", a.draft,
               "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
               "--speculative-num-draft-tokens", "8", "--speculative-dspark-block-size", "7",
               "--speculative-draft-attention-backend", "flashinfer",
               "--speculative-dsa-topk-backend", "sgl-kernel"]
        if a.sps_table:
            cmd += ["--speculative-dspark-sps-table-path", a.sps_table]
        if not a.enable_standard_graphs:
            cmd += ["--disable-cuda-graph"]
        if a.deterministic_control_only:
            if self.mode != "full":
                raise ValueError("Deterministic FlashInfer control cannot measure prefix reuse")
            cmd += ["--enable-deterministic-inference"]
        return cmd, env

    def start(self):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise SafetyFailure("Refusing to replace an existing service on the benchmark port")
        cmd, env = self.command()
        dump(self.folder / "launch.json", {"command": cmd,
             "environment": {k: v for k, v in env.items() if k.startswith("SGLANG_") or k in ("PYTHONPATH", "PATH")},
             "unsafe_target_only": self.mode == "target_only_shadow"})
        self.log_file = (self.folder / "server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, env=env, cwd=self.args.repo,
                                        stdout=self.log_file, stderr=subprocess.STDOUT,
                                        start_new_session=True)

    def check(self):
        if self.process.poll() is not None:
            raise SafetyFailure(f"Server exited with {self.process.returncode}")
        with (self.folder / "server.log").open(encoding="utf-8", errors="replace") as stream:
            stream.seek(self.scanned)
            text = stream.read()
            self.scanned = stream.tell()
        for marker in ERROR_MARKERS:
            if marker.lower() in text.lower():
                raise SafetyFailure(f"Server log contains {marker}; inspect {self.folder / 'server.log'}")

    def stop(self):
        if self.process:
            # Only signal the session created by this harness, never pkill by name.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
            except ProcessLookupError:
                pass
            dump(self.folder / "shutdown.json", {"pid": self.process.pid,
                 "exit_code": self.process.returncode, "stopped_at": time.time()})
        if self.log_file:
            self.log_file.close()


def make_dataset(tokenizer, seed: int) -> dict:
    passages = (
        "A speculative decoder proposes tokens with a small draft model and verifies them with the target. "
        "Prefix reuse is valid only when both models have compatible cached key and value tensors. "
        "Readiness metadata must survive node splits, but must be invalidated after weight changes. "
        "A cache budget may evict draft readiness without evicting the corresponding target prefix. "
        "In that case, the missing suffix must be recomputed from target hidden states. "
        "Measure first-token latency separately from decode latency and report the slowest requests. "
        "Concurrency affects queueing, batch geometry, acceptance, and memory consumption. "
        "A faster isolated kernel does not establish a faster end-to-end serving workload.\n")
    body = tokenizer.encode(passages * 1000, add_special_tokens=False)
    data = {}
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            cell = f"p{length}_c{concurrency}"
            head = tokenizer.encode(f"Benchmark case {seed}-{cell}. Read this technical document:\n",
                                    add_special_tokens=False)
            suffixes = [tokenizer.encode(
                f"\nTask {v}: summarize the document and explain "
                f"{('cache validity', 'partial recovery', 'latency measurement', 'batch scheduling', 'memory budget', 'token verification', 'safe fallback', 'experimental evidence')[v]}"
                ". Use clear technical sentences.\nAnswer:", add_special_tokens=False)
                for v in range(concurrency)]
            tail_len = max(map(len, suffixes))
            padding_id = tokenizer.encode(" ", add_special_tokens=False)
            if len(padding_id) != 1:
                raise ValueError("Tokenizer must encode a space as one token for fixed suffix padding")
            prefix_len = length - tail_len
            if prefix_len < len(head):
                raise ValueError("Prompt too short for protocol")
            prefix = head + body[:prefix_len - len(head)]
            prompts = [prefix + tail + padding_id * (tail_len - len(tail)) for tail in suffixes]
            assert all(len(p) == length for p in prompts)
            if len({tuple(p) for p in prompts}) != concurrency:
                raise ValueError("Tokenizer collapsed distinct task suffixes into identical prompts")
            data[cell] = {"prompt_length": length, "concurrency": concurrency,
                          "common_prefix_tokens": prefix_len, "prompts": prompts}
    return data


async def api(session, base: str, path: str, method: str = "GET") -> Any:
    async with session.request(method, base + path) as response:
        response.raise_for_status()
        if path.startswith("/flush_cache"):
            return await response.text()
        return await response.json()


def request_body(prompt: list[int], output_tokens: int, seed: int) -> dict:
    return {"input_ids": prompt, "stream": True,
            "sampling_params": {"temperature": 0, "top_p": 1, "top_k": -1,
                                "max_new_tokens": output_tokens, "ignore_eos": True, "sampling_seed": seed}}


async def generate(session, base: str, prompt: list[int], output_tokens: int, seed: int) -> dict:
    body = request_body(prompt, output_tokens, seed)
    start = time.perf_counter()
    tracker = StreamTracker(start)
    async with session.post(base + "/generate", json=body) as response:
        response.raise_for_status()
        async for frame in sse_frames(response.content):
            now = time.perf_counter()
            if frame == "[DONE]":
                break
            tracker.add(json.loads(frame), now)
    row = tracker.result(time.perf_counter())
    if row["prompt_tokens"] != len(prompt):
        raise SafetyFailure("Server prompt length does not match fixed input_ids length")
    if row["completion_tokens"] != output_tokens:
        raise SafetyFailure("Response did not produce the configured fixed output length")
    return row


def aggregate(mode: str, cell: str, repeat: int, rows: list[dict], before: dict,
              after: dict, gpu: dict, elapsed: float) -> dict:
    proposed = sum(r["meta_info"]["spec_num_proposed_drafts"] for r in rows)
    correct = sum(r["meta_info"]["spec_num_correct_drafts"] for r in rows)
    verifies = sum(r["meta_info"]["spec_verify_ct"] for r in rows)
    return {"mode": mode, "cell": cell, "repeat": repeat, "request_count": len(rows),
            "prompt_length": rows[0]["prompt_tokens"], "concurrency": rows[0]["concurrency"],
            "output_tokens": rows[0]["completion_tokens"],
            "ttft_ms": distribution([r["ttft_ms"] for r in rows]),
            "tpot_ms": distribution([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
            "e2e_ms": distribution([r["e2e_ms"] for r in rows]),
            "client_observed_output_tokens_per_s": sum(r["completion_tokens"] for r in rows) / elapsed,
            "acceptance": {"correct_drafts": correct, "proposed_drafts": proposed,
                           "verify_count": verifies, "draft_accept_rate": correct / proposed if proposed else None,
                           "completion_tokens_per_verify": sum(r["completion_tokens"] for r in rows) / verifies if verifies else None},
            "cache": cache_metrics(counter_delta(before, after), rows, active=mode != "full"),
            "cache_gauges_before": {k: before.get(k) for k in ("ready_tokens", "tracked_tokens", "max_tokens")},
            "cache_gauges_after": {k: after.get(k) for k in ("ready_tokens", "tracked_tokens", "max_tokens")},
            "gpu": gpu, "correctness": "token_and_acceptance_match",
            "p95_claim": "descriptive_only"}


def info_stats(info: dict, *, required: bool = True) -> dict:
    states = info.get("internal_states", [])
    if len(states) != 1:
        raise SafetyFailure("This protocol requires exactly one scheduler/DP rank")
    if states[0].get("incremental_streaming_output"):
        raise SafetyFailure("Cumulative streaming output is required")
    stats = states[0].get("dspark_prefix_cache")
    if stats is None:
        if required:
            raise SafetyFailure("Server does not expose dspark_prefix_cache statistics")
        return {}  # Disabled radix implementations may have no sidecar: unavailable, not zero.
    return stats


def load_frozen_references(path, dataset):
    """Never let a new candidate silently replace the original serial oracle."""
    if not path:
        return {}
    source = Path(path)
    adjacent = source.parent / "dataset.json"
    if not adjacent.exists() or json.loads(adjacent.read_text(encoding="utf-8")) != dataset:
        raise ValueError("Frozen oracle must have an adjacent identical full dataset.json")
    expected = {(cell, variant) for cell, spec in dataset.items() for variant in range(spec["concurrency"])}
    references = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("phase") != "reference":
            continue
        key = row["cell"], row["variant"]
        if key in references:
            raise ValueError("Duplicate frozen oracle reference")
        references[key] = row
    if set(references) != expected:
        raise ValueError("Frozen oracle must cover every variant of every selected cell exactly once")
    return references


async def run_mode(args, mode: str, dataset: dict, references: dict, summaries: list[dict]):
    import aiohttp
    folder = Path(args.output) / mode
    folder.mkdir()
    server = ManagedServer(args, mode, folder)
    gpu = GPUSampler(folder / "gpu_samples.jsonl", args.gpu_sample_ms)
    base = f"http://127.0.0.1:{args.port}"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout,
                                    connector=aiohttp.TCPConnector(limit=16)) as session:
        try:
            gpu.start()
            server.start()
            deadline = time.monotonic() + args.startup_timeout
            while True:
                server.check()
                try:
                    await api(session, base, "/model_info")
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if time.monotonic() > deadline:
                        raise SafetyFailure("Server readiness timeout")
                    await asyncio.sleep(.5)
            info = await api(session, base, "/server_info")
            dump(folder / "server_info.json", info)
            info_stats(info, required=mode != "full")
            if bool(info.get("enable_deterministic_inference")) != args.deterministic_control_only:
                raise SafetyFailure("Resolved deterministic flag differs from protocol")
            if info.get("enable_deterministic_inference") and mode != "full":
                raise SafetyFailure("FlashInfer deterministic mode disables radix caching")
            if not args.enable_standard_graphs and not info.get("disable_cuda_graph"):
                raise SafetyFailure("Requested graph-off configuration was not applied")
            for _ in range(args.warmup_requests):
                await generate(session, base, dataset["p512_c1"]["prompts"][0], args.output_tokens, args.seed)
            print(f"READY mode={mode} gpu_samples={len(gpu.rows)}", flush=True)

            async def request(cell, spec, variant, phase, repeat, index):
                server.check()
                row = await generate(session, base, spec["prompts"][variant], args.output_tokens, args.seed)
                row.update(mode=mode, cell=cell, variant=variant, phase=phase, repeat=repeat,
                           request_index=index, concurrency=spec["concurrency"])
                append_jsonl(folder / "requests.jsonl", row)  # Preserve the mismatching row before raising.
                if mode == "full" and row["cached_tokens"] != 0:
                    raise SafetyFailure("Full-prefill control unexpectedly reused cached tokens")
                reference_key = (cell, variant)
                if mode == "full" and phase == "reference":
                    if reference_key in references:
                        check_reference(references[reference_key], row)
                    references[reference_key] = row
                else:
                    check_reference(references[reference_key], row)
                return row

            for cell, spec in dataset.items():
                # Every reference is actually cold, not accidentally warmed by a sibling.
                for variant in range(spec["concurrency"]):
                    await api(session, base, "/flush_cache?timeout=10", "POST")
                    await request(cell, spec, variant, "reference", -1, variant)
                for repeat in range(args.repeats):
                    await api(session, base, "/flush_cache?timeout=10", "POST")
                    for variant in range(spec["concurrency"]):
                        await request(cell, spec, variant, "prime", repeat, variant)
                    before = info_stats(await api(session, base, "/server_info"), required=mode != "full")
                    gpu_start = gpu.begin(f"{cell}_r{repeat}_steady")
                    count = args.requests
                    queue = asyncio.Queue()
                    for index in range(count):
                        queue.put_nowait(index)
                    rows = []

                    async def worker():
                        while not queue.empty():
                            index = queue.get_nowait()
                            rows.append(await request(cell, spec, index % spec["concurrency"],
                                                      "steady", repeat, index))

                    started = time.perf_counter()
                    tasks = [asyncio.create_task(worker()) for _ in range(spec["concurrency"])]
                    try:
                        await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    elapsed = time.perf_counter() - started
                    server.check()
                    after = info_stats(await api(session, base, "/server_info"), required=mode != "full")
                    summary = aggregate(mode, cell, repeat, rows, before, after, gpu.summary(gpu_start), elapsed)
                    if summary["gpu"]["sample_count"] == 0:
                        raise SafetyFailure("No measured GPU samples; cannot report memory")
                    append_jsonl(Path(args.output) / "cells.jsonl", summary)
                    summaries.append(summary)
                    print(json.dumps({"mode": mode, "cell": cell, "repeat": repeat,
                          "n": len(rows), "ttft_p50_ms": summary["ttft_ms"]["p50"],
                          "e2e_p95_ms": summary["e2e_ms"]["p95"],
                          "paired_hit_tokens": summary["cache"]["counter_delta"]["paired_hit_tokens"]}), flush=True)
                    # Full and paired are only comparable after adequate sample counts.
                    if mode == "paired":
                        controls = [s for s in summaries if s["mode"] == "full" and s["cell"] == cell and s["repeat"] == repeat]
                        if controls:
                            check_p95(controls[0], summary, args.min_p95_samples, args.max_p95_regression)
        finally:
            try:
                server.stop()
            finally:
                gpu.stop()


def write_report(args, summaries: list[dict], status: dict):
    root = Path(args.output)
    flat = []
    for s in summaries:
        delta = s["cache"]["counter_delta"]
        flat.append({"mode": s["mode"], "cell": s["cell"], "repeat": s["repeat"],
                     "n": s["request_count"], "ttft_p50_ms": s["ttft_ms"]["p50"],
                     "ttft_p95_ms": s["ttft_ms"]["p95"], "tpot_p50_ms": s["tpot_ms"]["p50"],
                     "tpot_p95_ms": s["tpot_ms"]["p95"], "e2e_p95_ms": s["e2e_ms"]["p95"],
                     "device_memory_peak_mib": s["gpu"]["peak_device_memory_used_mib"],
                     "accept_rate": s["acceptance"]["draft_accept_rate"],
                     "accept_length": s["acceptance"]["completion_tokens_per_verify"],
                     "target_token_hit_rate": s["cache"]["observed_target_token_hit_rate"],
                     "paired_token_hit_rate": s["cache"]["observed_paired_token_hit_rate"],
                     "paired_hit_length_mean": s["cache"]["paired_hit_length_mean_per_match"],
                     "shadow": s["mode"] == "target_only_shadow", **delta})
    if flat:
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    dump(root / "status.json", status)
    lines = ["# DSpark fixed cache A/B", "", f"Status: {status['status']}", "",
             "Performance conclusion: UNPROVEN (a smoke run or one fixed order is not a stable-state A/B).", "",
             "TTFT: request send to first token-bearing SSE frame. TPOT: (last token receipt - first token receipt)/(N-1).",
             "Speculative burst tokens share a frame; no synthetic per-token arrival times are invented.", "",
             "GPU memory: device-wide nvidia-smi samples, not PyTorch allocator peak.", "",
             "Recovery/fallback counters describe sidecar plans, not independently instrumented draft worker fallbacks.", "",
             "Cold references and priming requests are excluded from steady-state latency percentiles.", ""]
    if status.get("error"):
        lines += [f"Safety stop: {status['error']}", ""]
    if flat:
        lines += ["| Mode | Cell | n | TTFT p50 / p95 ms | TPOT p50 ms | E2E p95 ms | GPU MiB |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in flat:
            lines.append(f"| {r['mode']} | {r['cell']} | {r['n']} | {r['ttft_p50_ms']:.2f} / {r['ttft_p95_ms']:.2f} | {r['tpot_p50_ms']:.2f} | {r['e2e_p95_ms']:.2f} | {r['device_memory_peak_mib']} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path.cwd()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--sps-table")
    parser.add_argument("--oracle-jsonl", help="Complete frozen original references with adjacent identical dataset.json")
    parser.add_argument("--output", required=True, help="New, non-existing directory; never overwrites a run")
    parser.add_argument("--preset", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--include-target-only-shadow", action="store_true", help="Explicit unsafe, isolated target-only control")
    parser.add_argument("--deterministic-control-only", action="store_true",
                        help="Diagnose serial/concurrent correctness with full prefill only; NOT a cache A/B")
    parser.add_argument("--reference-verify-plans", action="store_true",
                        help="Default-off eager DSpark serial-reference attention correctness candidate")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--port", type=int, default=30117)
    parser.add_argument("--mem-fraction", type=float, default=.55)
    parser.add_argument("--cache-budget-fraction", type=float, default=.25)
    parser.add_argument("--enable-standard-graphs", action="store_true", help="Never enables ragged target-verify graph")
    parser.add_argument("--gpu-sample-ms", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--min-p95-samples", type=int, default=64)
    parser.add_argument("--max-p95-regression", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.requests = args.requests if args.requests is not None else (8 if args.preset == "smoke" else 64)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.preset == "smoke" else 3)
    args.output_tokens = args.output_tokens if args.output_tokens is not None else (32 if args.preset == "smoke" else 128)
    if args.requests < 8 or args.requests % 8 or args.repeats < 1 or args.output_tokens < 2:
        parser.error("requests must be a positive multiple of 8; repeats>=1; output-tokens>=2")
    if args.gpu_sample_ms < 20 or args.min_p95_samples < 8 or args.max_p95_regression < 0:
        parser.error("Invalid GPU interval or p95 guard parameters")
    if not (0 < args.mem_fraction < 1 and 0 < args.cache_budget_fraction <= 1):
        parser.error("Invalid memory/cache fraction")
    if args.deterministic_control_only and (args.include_target_only_shadow or args.enable_standard_graphs):
        parser.error("Deterministic control requires full prefill and all graphs off")
    if args.reference_verify_plans and args.enable_standard_graphs:
        parser.error("Serial-reference verify plans require all graphs off")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.repo = str(Path(args.repo).resolve())
    args.output = str(Path(args.output).resolve())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    modes = ["full"] if args.deterministic_control_only else ["full", "paired"] + (["target_only_shadow"] if args.include_target_only_shadow else [])
    manifest = {"protocol_version": 2, "args": vars(args), "modes": modes,
                "fixed_lengths": LENGTHS, "fixed_concurrencies": CONCURRENCIES,
                "started_at": time.time(), "python": sys.version,
                "deterministic_inference": args.deterministic_control_only, "sampling": "greedy, pytorch, fixed seed",
                "control_only_not_cache_ab": args.deterministic_control_only,
                "reference_verify_candidate_enabled": args.reference_verify_plans,
                "frozen_oracle_sha256": hashlib.sha256(Path(args.oracle_jsonl).read_bytes()).hexdigest()
                    if args.oracle_jsonl else None,
                "unsafe_mode_acknowledged": args.include_target_only_shadow,
                "performance_claim": "unproven", "p95_scope": "per cell/repeat; stop on regression only with adequate n"}
    manifest["harness_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("aiohttp", "torch", "transformers", "flashinfer-python", "sglang"):
        try:
            manifest.setdefault("packages", {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest.setdefault("packages", {})[name] = None
    for name, cmd in {"git_head": ["git", "rev-parse", "HEAD"], "git_branch": ["git", "branch", "--show-current"],
                      "git_tracked_diff": ["git", "diff", "--name-only"],
                      "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]}.items():
        manifest[name] = subprocess.run(cmd, cwd=args.repo, capture_output=True, text=True).stdout.strip()
    dump(root / "manifest.json", manifest)
    summaries: list[dict] = []
    references: dict = {}
    status = {"status": "running", "completed_cells": 0,
              "inference_code_modified": args.reference_verify_plans}
    dump(root / "status.json", status)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        dataset = make_dataset(tokenizer, args.seed)
        dump(root / "dataset.json", dataset)
        references.update(load_frozen_references(args.oracle_jsonl, dataset))
        for mode in modes:
            asyncio.run(run_mode(args, mode, dataset, references, summaries))
        status["status"] = "completed_smoke" if args.preset == "smoke" else "completed_no_gain_claim"
        return_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="safety_stopped", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(status), file=sys.stderr, flush=True)
        return_code = 2
    finally:
        status.update(completed_cells=len(summaries), finished_at=time.time(),
                      performance_claim="unproven", managed_services_stopped=True)
        write_report(args, summaries, status)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
