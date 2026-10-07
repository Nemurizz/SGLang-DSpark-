"""Read-only same-profile target decode vs DSpark verify diagnosis.

Uses preserved h26 controls only to ensure observers did not change each arm.
This is not a new oracle, a correctness pass, or a performance measurement.
"""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from dspark_verify_trace_hook import prompt_hash
from stable_baseline_gate import SOURCE_PATHS, profile_command, sha, validate_profile, validate_row
from stable_profile_hook import validate_runtime

ORIGINAL = "a61e0bb61f285a404f527d98190a9fe2610c553a"
STAGES = {"layer0.attention", "layer0.post", "layer19.post", "layer39.post", "logits"}


def config(folder, prompt, expected):
    return {"folder": str(folder / "verify_trace"), "prompt_sha256": prompt_hash(prompt),
            "prompt_length": len(prompt), "position": len(prompt),
            "anchor_token": expected["output_ids"][0], "max_captures": 10,
            "kv_layer": 0, "max_kv_prefix": 4096, "observe_layer0_projection": True,
            "max_projection_rows": 1024}


def load_expected(folder):
    rows = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
    rows = [row for row in rows if (row["cell"], row["variant"], row["phase"]) ==
            ("p128_c1", 0, "cold_reference")]
    if len(rows) != 1 or rows[0]["cached_tokens"] != 0:
        raise ValueError("Requires one preserved cache-off h26 cold control per arm")
    return rows[0]


async def run_arm(args, root, label, prompt, expected, profile_id):
    import aiohttp
    ordinary = label == "ordinary_target"
    folder = root / label
    folder.mkdir()
    cfg = config(folder, prompt, expected)

    class TraceServer(ab.ManagedServer):
        def command(self):
            cmd, env = profile_command(*super().command(), ordinary=ordinary, profile_id=profile_id,
                runtime_folder=folder / "runtime", hook_folder=Path(__file__).resolve().parent)
            index = cmd.index("--forward-hooks") + 1
            hooks = json.loads(cmd[index])
            hooks.append({"name": "target-verify-read-only", "target_modules": [
                "model.layers.0.input_layernorm", "model.layers.0.self_attn.qkv_proj",
                "model.layers.0.self_attn.attn", "model.layers.0", "model.layers.19",
                "model.layers.39", "logits_processor"],
                "hook_factory": "dspark_verify_trace_hook:make_hook", "config": cfg})
            cmd[index] = json.dumps(hooks)
            return cmd, env

    server = TraceServer(args, "full", folder)
    base = f"http://127.0.0.1:{args.port}"
    status = {"status": "running", "label": label, "scope": "diagnosis only; failed target gate remains failed"}
    ab.dump(folder / "trace_config.json", cfg)
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
                        raise ab.SafetyFailure("Trace startup timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(folder / "server_info.json", info)
            validate_profile(info, ordinary)
            print("READY " + label, flush=True)
            for _ in range(args.warmup_requests):
                row = await ab.generate(session, base, args.warmup_prompt, args.output_tokens, args.seed)
                validate_row(row, {"prompt_length": len(args.warmup_prompt)}, args.output_tokens,
                             ordinary=ordinary, require_cold=True)
                server.check()
            for repeat in range(2):
                await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                row = await ab.generate(session, base, prompt, args.output_tokens, args.seed)
                row.update(cell="p128_c1", variant=0, phase="diagnostic_repeat", mode=label, repeat=repeat)
                ab.append_jsonl(folder / "requests.jsonl", row)
                validate_row(row, {"prompt_length": len(prompt)}, args.output_tokens,
                             ordinary=ordinary, require_cold=True)
                ab.check_reference(expected, row, check_acceptance=not ordinary)
                server.check()
                print(f"MATCH preserved {label} repeat={repeat}", flush=True)
            captures = [json.loads(p.read_text()) for p in (folder / "verify_trace").glob("*.json")]
            if Counter(item["stage"] for item in captures) != Counter({stage: 2 for stage in STAGES}):
                raise ab.SafetyFailure("Incomplete committed-anchor stage captures")
            records = list((folder / "runtime").glob("*.json"))
            if not records:
                raise ab.SafetyFailure("Missing worker runtime attestation")
            for path in records:
                validate_runtime(json.loads(path.read_text()))
            status.update(status="target_verify_trace_complete", requests=2, stage_captures=len(captures))
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        server.stop()
        status.update(managed_service_stopped=True)
        ab.dump(folder / "status.json", status)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    manifest = json.loads((cli.control / "manifest.json").read_text())
    args = SimpleNamespace(**manifest["args"])
    args.deterministic_control_only = args.reference_verify_plans = args.enable_standard_graphs = False
    args.output = str(cli.output)
    repo = Path(args.repo)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != ORIGINAL or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        raise ValueError("Clean original checkout required; no inference changes")
    sources = {path: sha(repo / path) for path in SOURCE_PATHS}
    if sources != manifest["source_sha256"]:
        raise ValueError("Inference source hashes differ from preserved profile")
    guards = [cli.control / "manifest.json", cli.control / "dataset.json"]
    for label in ("ordinary_target", "dspark_full"):
        guards.append(cli.control / label / "requests.jsonl")
    guards += [Path(path) for path in manifest["old_references_read_only"]]
    before = {str(path): sha(path) for path in guards}
    all_data = json.loads((cli.template / "dataset.json").read_text())
    args.warmup_prompt = all_data["p512_c1"]["prompts"][0]
    prompt = all_data["p128_c1"]["prompts"][0]
    expected = {label: load_expected(cli.control / label) for label in ("ordinary_target", "dspark_full")}
    if expected["ordinary_target"]["output_ids"][0] != expected["dspark_full"]["output_ids"][0]:
        raise ValueError("Arms have no common first committed anchor")
    cli.output.mkdir(parents=True, exist_ok=False)
    ab.dump(cli.output / "manifest.json", {"args": vars(args), "head": head,
        "profile_id": manifest["profile_id"], "source_sha256": sources,
        "read_only_controls": before, "tool_sha256": {p.name: sha(p) for p in (
            Path(__file__), Path(ab.__file__), Path(__file__).with_name("dspark_verify_trace_hook.py"),
            Path(__file__).with_name("stable_baseline_gate.py"), Path(__file__).with_name("stable_profile_hook.py"))},
        "no_new_oracle": True, "no_inference_candidate": True, "performance_claim": "unproven"})
    status = {"status": "running", "baseline_target_gate": "failed", "performance_claim": "unproven"}
    try:
        for label, control in expected.items():
            if {path: sha(repo / path) for path in SOURCE_PATHS} != sources:
                raise ab.SafetyFailure("Inference sources changed before arm")
            await run_arm(args, cli.output, label, prompt, control, manifest["profile_id"])
        status["status"] = "target_verify_trace_complete"
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        status.update(read_only_controls_unchanged=all(sha(path) == before[str(path)] for path in guards),
                      inference_sources_unchanged={path: sha(repo / path) for path in SOURCE_PATHS} == sources)
        ab.dump(cli.output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "target_verify_trace_complete" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
