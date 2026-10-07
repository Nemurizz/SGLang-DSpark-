"""Independent stable numerical profile correctness gate, NOT performance A/B.

Human authorized a new arithmetic contract. Old references are immutable. First
collect/check ordinary target greedy controls, then DSpark cold/repeated/full
concurrent tokens and all five acceptance fields. Cache cannot start before that.
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
from stable_profile_hook import validate_runtime

PROFILE = {"name": "dspark-stable-fa4-v1", "deterministic": True,
    "target_prefill": "fa4", "target_verify_decode": "fa4", "draft_attention": "fa4",
    "bf16_gemm_backend": "torch", "dtype": "bfloat16", "kv_dtype": "bfloat16", "tp_size": 1,
    "page_size": 1, "graphs": "OFF", "sampling": "greedy-pytorch",
    "mm_deepgemm": False, "mm_batch_variant_fallback": False,
    "compact_scan_cuda": False, "dynamic_draft_length": False,
    "sps_table": "preserved, fixed", "old_bf16_reduction_probe": False}
SOURCE_PATHS = (
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py",
    "python/sglang/srt/layers/attention/flashattention_backend.py",
    "python/sglang/srt/models/qwen3.py", "python/sglang/srt/models/dspark.py",
    "python/sglang/srt/layers/layernorm.py", "python/sglang/srt/layers/quantization/unquant.py",
    "python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py",
    "python/sglang/kernels/ops/attention/fa4_sm120/dispatch.py",
    "python/sglang/kernels/ops/attention/fa4_sm120/runtime.py",
    "python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd.py",
    "python/sglang/kernels/ops/attention/fa4_sm120/flash_fwd_decode.py",
    "python/sglang/kernels/ops/attention/flash_attn/cute/batch_invariance.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_command(cmd, env, *, ordinary, profile_id, runtime_folder, hook_folder):
    cmd, env = list(cmd), dict(env)
    for flag in ("--attention-backend", "--decode-attention-backend", "--speculative-draft-attention-backend"):
        cmd[cmd.index(flag) + 1] = "fa4"
    if ordinary:
        filtered, index = [], 0
        while index < len(cmd):
            if cmd[index].startswith("--speculative-"):
                index += 2  # Shared harness emits these as option/value pairs.
            else:
                filtered.append(cmd[index])
                index += 1
        cmd = filtered + ["--disable-radix-cache"]
    cmd += ["--enable-deterministic-inference", "--prefill-attention-backend", "fa4",
            "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--bf16-gemm-backend", "torch",
            "--page-size", "1", "--forward-hooks", json.dumps([{"name": "stable-profile-attestation",
                "target_modules": ["logits_processor"], "hook_factory": "stable_profile_hook:make_hook",
                "config": {"profile_id": profile_id, "folder": str(runtime_folder)}}])]
    # Set import-time choices before worker launch; no hidden batch-variant fallback.
    env.update({"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "0",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST": "0",
                "SGLANG_DSPARK_BF16_FULL_REDUCTION": "0", "SGLANG_INKLING_FA4_USE_PIP": "0",
                "SGLANG_DSPARK_PREFIX_CACHE_FINGERPRINT": profile_id,
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"})
    env["PYTHONPATH"] += ":" + str(hook_folder)
    return cmd, env


def validate_profile(info, ordinary):
    expected = {"enable_deterministic_inference": True, "disable_cuda_graph": True,
                "disable_radix_cache": ordinary, "attention_backend": "fa4",
                "prefill_attention_backend": "fa4", "decode_attention_backend": "fa4",
                "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "bf16_gemm_backend": "torch",
                "tp_size": 1, "page_size": 1, "quantization": None}
    if not ordinary:
        expected.update(speculative_algorithm="DSPARK", speculative_draft_attention_backend="fa4")
    for key, value in expected.items():
        if info.get(key) != value:
            raise ab.SafetyFailure(f"Resolved stable profile mismatch {key}: {info.get(key)!r} != {value!r}")


def validate_row(row, spec, output_tokens, *, ordinary, require_cold):
    if require_cold and row["cached_tokens"] != 0:
        raise ab.SafetyFailure("Cold/full control unexpectedly reused cache")
    if row["prompt_tokens"] != spec["prompt_length"] or len(row["output_ids"]) != output_tokens:
        raise ab.SafetyFailure("Stable gate prompt/output length mismatch")
    if not ordinary:
        ab.check_reference(row, row)


async def run_arm(args, root, dataset, references, target_refs, profile_id, *, ordinary, paired, repeats):
    import aiohttp
    mode = "paired" if paired else "full"
    label = "ordinary_target" if ordinary else ("dspark_paired" if paired else "dspark_full")
    folder = root / label
    folder.mkdir()

    class StableServer(ab.ManagedServer):
        def command(self):
            cmd, env = super().command()
            return profile_command(cmd, env, ordinary=ordinary, profile_id=profile_id,
                runtime_folder=folder / "runtime", hook_folder=Path(__file__).resolve().parent)

    server = StableServer(args, mode, folder)
    base = f"http://127.0.0.1:{args.port}"
    status = {"label": label, "status": "running", "completed_cells": 0, "performance_claim": "unproven"}
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
                        raise ab.SafetyFailure(label + " startup timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(folder / "server_info.json", info)
            validate_profile(info, ordinary)
            print("READY " + label, flush=True)
            for _ in range(args.warmup_requests):
                warm = await ab.generate(session, base, args.warmup_prompt, args.output_tokens, args.seed)
                validate_row(warm, {"prompt_length": len(args.warmup_prompt)}, args.output_tokens,
                             ordinary=ordinary, require_cold=not paired)
                server.check()
            runtime_records = list((folder / "runtime").glob("*.json"))
            if not runtime_records:
                raise ab.SafetyFailure("Missing worker runtime attestation")
            for path in runtime_records:
                validate_runtime(json.loads(path.read_text()))

            async def request(cell, v, phase, repeat):
                row = await ab.generate(session, base, dataset[cell]["prompts"][v], args.output_tokens, args.seed)
                row.update(cell=cell, variant=v, phase=phase, repeat=repeat, profile_id=profile_id, mode=label)
                ab.append_jsonl(folder / "requests.jsonl", row)
                validate_row(row, dataset[cell], args.output_tokens, ordinary=ordinary,
                             require_cold=not paired or phase == "cold_reference")
                key = (cell, v)
                if ordinary:
                    if phase == "cold_reference":
                        target_refs[key] = row
                    else:
                        ab.check_reference(target_refs[key], row, check_acceptance=False)
                else:
                    ab.check_reference(target_refs[key], row, check_acceptance=False)
                    if phase == "cold_reference" and not paired:
                        references[key] = row
                    else:
                        ab.check_reference(references[key], row)
                server.check()

            async def concurrent(cell, phase, repeat):
                tasks = [asyncio.create_task(request(cell, v, phase, repeat)) for v in range(dataset[cell]["concurrency"])]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            for cell, spec in dataset.items():
                status["active_cell"] = cell
                for v in range(spec["concurrency"]):
                    status.update(active_phase="cold_reference", active_variant=v)
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    await request(cell, v, "cold_reference", -1)
                    if not paired:
                        await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                        await request(cell, v, "cold_repeat", -1)
                if paired:
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    before = ab.info_stats(await ab.api(session, base, "/server_info"), required=True)
                    # Priming variant 0 then DISTINCT suffix variants also tests shared-prefix reuse.
                    for v in range(spec["concurrency"]):
                        await request(cell, v, "prime_distinct_suffix", -1)
                for repeat in range(repeats):
                    status.update(active_phase="serial_repeat", active_repeat=repeat)
                    if not paired:
                        await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    for v in range(spec["concurrency"]):
                        await request(cell, v, "serial_repeat", repeat)
                    status["active_phase"] = "concurrent"
                    if not paired:
                        await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    await concurrent(cell, "concurrent", repeat)
                    print(f"PASS {label} {cell} round={repeat}", flush=True)
                item = {"cell": cell, "repeats": repeats, "output_tokens": args.output_tokens, "status": "passed"}
                if paired:
                    after = ab.info_stats(await ab.api(session, base, "/server_info"), required=True)
                    item.update(cache_stats_before=before, cache_stats_after=after)
                    # A correctness check with zero real reuse does not qualify as a cache gate.
                    rows = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
                    hot = [row for row in rows if row["cell"] == cell and row["phase"] in ("serial_repeat", "concurrent")]
                    if not hot or any(row["cached_tokens"] <= 0 for row in hot):
                        raise ab.SafetyFailure("Warm paired correctness did not exercise actual reuse")
                ab.append_jsonl(folder / "cells.jsonl", item)
                status["completed_cells"] += 1
            status["status"] = "passed_for_requested_cells_only"
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
        print("SAFETY STOP " + label + ": " + str(exc), flush=True)
        raise
    finally:
        server.stop()
        status.update(managed_service_stopped=True, finished_at=time.time())
        ab.dump(folder / "status.json", status)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--cells", default="p128_c1,p128_c4,p128_c8")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--with-paired", action="store_true")
    cli = parser.parse_args()
    previous = json.loads((cli.template / "manifest.json").read_text())
    args = SimpleNamespace(**previous["args"])
    args.deterministic_control_only = args.reference_verify_plans = args.enable_standard_graphs = False
    args.output_tokens, args.output = cli.output_tokens, str(cli.output)
    if args.output_tokens < 32:
        raise ValueError("Correctness smoke requires at least 32 output tokens")
    repo = Path(args.repo)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != cli.expected_head or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        raise ValueError("Requires the explicitly pinned clean independent profile commit")
    old_oracle = Path(args.oracle_jsonl)
    frozen_paths = (old_oracle, cli.template / "requests.jsonl", cli.template / "dataset.json")
    frozen_before = {str(path): sha(path) for path in frozen_paths}
    all_data = json.loads((cli.template / "dataset.json").read_text())
    cells = list(all_data) if cli.cells == "all" else cli.cells.split(",")
    if not cells or len(set(cells)) != len(cells) or any(cell not in all_data for cell in cells):
        raise ValueError("Unknown or duplicate matrix cells")
    dataset = {cell: all_data[cell] for cell in cells}
    args.warmup_prompt = all_data["p512_c1"]["prompts"][0]
    sources = {name: sha(repo / name) for name in SOURCE_PATHS}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "aiohttp", "transformers")}
    numerical = {"profile": PROFILE, "source_sha256": sources, "packages": versions}
    profile_id = PROFILE["name"] + "-" + hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()[:24]
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / "dataset.json", dataset)
    ab.dump(cli.output / "manifest.json", {"args": vars(args), "head": head, "profile_id": profile_id,
        **numerical, "old_references_read_only": frozen_before, "cells": cells, "repeats": 3,
        "human_authorization": "independent stable numerical baseline; old BF16 rounding equality not required",
        "new_contract": "independent target greedy + exact within-profile tokens and all five acceptance fields",
        "performance_claim": "unproven", "partial_recovery_coverage": "not guaranteed; record observations, not zero",
        "started_at": time.time()})
    target_refs, references = {}, {}
    status = {"status": "running", "profile_id": profile_id, "paired_started": False, "performance_claim": "unproven"}
    try:
        await run_arm(args, cli.output, dataset, references, target_refs, profile_id, ordinary=True, paired=False, repeats=3)
        await run_arm(args, cli.output, dataset, references, target_refs, profile_id, ordinary=False, paired=False, repeats=3)
        status["full_correctness_passed"] = True
        if cli.with_paired:
            status["paired_started"] = True
            await run_arm(args, cli.output, dataset, references, target_refs, profile_id, ordinary=False, paired=True, repeats=3)
            status["paired_correctness_passed"] = True
        status["status"] = "passed_for_requested_cells_only"
        # Never publish a complete reusable matrix oracle from a focused subset.
        if cells == list(all_data):
            for row in references.values():
                ab.append_jsonl(cli.output / "baseline_references.jsonl", row)
            status["baseline_reference_sha256"] = sha(cli.output / "baseline_references.jsonl")
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        status.update(finished_at=time.time(), old_references_unchanged=all(sha(path) == frozen_before[str(path)] for path in frozen_paths))
        ab.dump(cli.output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "passed_for_requested_cells_only" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
