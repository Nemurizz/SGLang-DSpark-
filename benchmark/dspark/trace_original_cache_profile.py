"""Original-checkpoint, cache-off profile isolation; never creates a new oracle.

Both arms use original a61e0bb. Expected non-deterministic output is the saved
h19 diagnostic control, NOT an acceptance oracle for cache benchmarks. The
existing read-only hook captures the first committed verify anchor twice.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from dspark_verify_trace_hook import prompt_hash


ORIGINAL = "a61e0bb61f285a404f527d98190a9fe2610c553a"
STAGES = {"layer0.attention", "layer0.post", "layer19.post", "layer39.post", "logits"}


def trace_config(folder, prompt, frozen):
    if not prompt or not frozen["output_ids"]:
        raise ValueError("Missing prompt or original reference anchor")
    return {"folder": str(folder / "verify_trace"), "prompt_sha256": prompt_hash(prompt),
            "prompt_length": len(prompt), "position": len(prompt),
            "anchor_token": frozen["output_ids"][0], "max_captures": 10,
            "kv_layer": 0, "max_kv_prefix": 4096}


def validate_server_info(info, deterministic):
    if bool(info.get("enable_deterministic_inference")) != deterministic:
        raise ab.SafetyFailure("Server deterministic profile differs from requested arm")
    if not info.get("disable_cuda_graph"):
        raise ab.SafetyFailure("Profile isolation requires graphs disabled")
    # Preserve h19: radix structures exist in the non-deterministic profile,
    # but DSpark's key_limit=0 guard prevents reuse (all requests also checked).
    if bool(info.get("disable_radix_cache")) != deterministic:
        raise ab.SafetyFailure("Radix structure configuration differs from original profile")


class TraceServer(ab.ManagedServer):
    def __init__(self, args, folder, config):
        super().__init__(args, "full", folder)
        self.config = config

    def command(self):
        cmd, env = super().command()
        cmd += ["--forward-hooks", json.dumps([{
            "name": "original-profile-read-only", "target_modules": [
                "model.layers.0.self_attn.attn", "model.layers.0", "model.layers.19",
                "model.layers.39", "logits_processor"],
            "hook_factory": "dspark_verify_trace_hook:make_hook", "config": self.config}])]
        env["PYTHONPATH"] += ":" + str(Path(__file__).resolve().parent)
        return cmd, env


async def main():
    import aiohttp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--arm", choices=("deterministic", "cache-profile"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    manifest = json.loads((cli.control_run / "manifest.json").read_text())
    args = SimpleNamespace(**manifest["args"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=args.repo, text=True).strip()
    if head != ORIGINAL or dirty:
        raise ValueError("Original clean tracked checkpoint a61e0bb required")
    args.deterministic_control_only = cli.arm == "deterministic"
    args.reference_verify_plans = False
    args.output = str(cli.output)
    cli.output.mkdir(parents=True, exist_ok=False)
    oracle = Path(args.oracle_jsonl)
    dataset = json.loads(oracle.with_name("dataset.json").read_text())
    frozen = ab.load_frozen_references(str(oracle), dataset)["p128_c1", 0]
    saved = [json.loads(line) for line in (cli.control_run / "requests.jsonl").read_text().splitlines()]
    if len(saved) != 1 or saved[0]["cached_tokens"] != 0:
        raise ValueError("Requires exactly one saved cache-off h19 diagnostic control")
    expected = frozen if args.deterministic_control_only else saved[0]
    config = trace_config(cli.output, dataset["p128_c1"]["prompts"][0], frozen)
    server = TraceServer(args, cli.output, config)
    status = {"status": "running", "arm": cli.arm, "scope": "profile diagnosis, not cache A/B"}
    ab.dump(cli.output / "manifest.json", {"args": vars(args), "head": head, "arm": cli.arm,
        "oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        "saved_control_sha256": hashlib.sha256((cli.control_run / "requests.jsonl").read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(ab.__file__), Path(__file__).with_name("dspark_verify_trace_hook.py"))},
        "no_new_oracle": True, "no_inference_candidate": True, "trace_config": config})
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
                        raise ab.SafetyFailure("Profile trace startup timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(cli.output / "server_info.json", info)
            validate_server_info(info, args.deterministic_control_only)
            for _ in range(args.warmup_requests):
                warmup = await ab.generate(session, base, dataset["p512_c1"]["prompts"][0], args.output_tokens, args.seed)
                if warmup["cached_tokens"] != 0:
                    raise ab.SafetyFailure("Warmup unexpectedly reused prefix cache")
                server.check()
            for repeat in range(2):
                row = await ab.generate(session, base, dataset["p128_c1"]["prompts"][0], args.output_tokens, args.seed)
                row.update(cell="p128_c1", variant=0, phase="diagnostic_repeat", mode="full", repeat=repeat)
                ab.append_jsonl(cli.output / "requests.jsonl", row)
                server.check()
                if row["cached_tokens"] != 0:
                    raise ab.SafetyFailure("Isolation arm unexpectedly reused prefix cache")
                ab.check_reference(expected, row)
            captures = [json.loads(p.read_text()) for p in (cli.output / "verify_trace").glob("*.json")]
            if Counter(m["stage"] for m in captures) != Counter({stage: 2 for stage in STAGES}):
                raise ab.SafetyFailure("Missing committed-anchor captures; cannot claim stage comparison")
            status.update(status="original_profile_trace_complete", requests=2, stage_captures=len(captures))
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        server.stop()
        ab.dump(cli.output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "original_profile_trace_complete" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
