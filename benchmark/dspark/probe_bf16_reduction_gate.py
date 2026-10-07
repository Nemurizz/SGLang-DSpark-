"""Gate one default-OFF precision candidate against original h21 B1 controls.

No candidate-made oracle, no cache reuse or performance matrix. A token OR any
of the five acceptance fields changing stops all owned generation immediately.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import dspark_prefix_ab as ab
from cache_profile_correctness import ORIGINAL, validate_profile, validate_row


def original_controls(folder):
    dataset = json.loads((folder / "dataset.json").read_text())
    records = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
    refs = {(r["cell"], r["variant"]): r for r in records if r["phase"] == "reference"}
    expected = {("p128_c1", 0)} | {("p128_c4", v) for v in range(4)}
    if refs.keys() != expected:
        raise ValueError("Requires the preserved five original h21 cold B1 controls")
    for key, row in refs.items():
        if row["original_revision"] != ORIGINAL:
            raise ValueError("Reference must be original a61, not the candidate")
        validate_row(row, dataset[key[0]], 32)
    return dataset, refs


async def main():
    import aiohttp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    cli = parser.parse_args()
    previous = json.loads((cli.baseline / "manifest.json").read_text())
    args = SimpleNamespace(**previous["args"])
    args.output = str(cli.output)
    args.deterministic_control_only = args.reference_verify_plans = args.enable_standard_graphs = False
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.repo, text=True).strip()
    if head != cli.candidate_revision or head == ORIGINAL or dirty:
        raise ValueError("Requires the explicitly pinned clean candidate commit")
    dataset, refs = original_controls(cli.baseline)
    cli.output.mkdir(parents=True, exist_ok=False)
    source = cli.baseline / "requests.jsonl"
    oracle_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"args": vars(args), "candidate_revision": head,
                "original_controls": str(source), "original_controls_sha256": oracle_sha,
                "candidate_oracle_generated": False, "paired_started": False,
                "precision_flag": "SGLANG_DSPARK_BF16_FULL_REDUCTION=1",
                "scope": "32-token five-case correctness probe only; performance unproven"}
    ab.dump(cli.output / "manifest.json", manifest)

    class PrecisionProbeServer(ab.ManagedServer):
        def command(self):
            cmd, env = super().command()
            env["SGLANG_DSPARK_BF16_FULL_REDUCTION"] = "1"
            return cmd, env

    server = PrecisionProbeServer(args, "full", cli.output)
    status = {"status": "running", "passed_cold_controls": 0, "passed_concurrent_rounds": 0,
              "paired_started": False, "performance_claim": "unproven"}
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
                        raise ab.SafetyFailure("Candidate startup timeout")
                    await asyncio.sleep(.5)
            info = await ab.api(session, base, "/server_info")
            ab.dump(cli.output / "server_info.json", info)
            validate_profile(args, info)
            log = (cli.output / "server.log").read_text(errors="replace")
            if "DSpark BF16 full reduction correctness probe ON" not in log:
                raise ab.SafetyFailure("Precision candidate did not confirm active flag")
            print("READY precision probe; frozen original controls; reuse OFF", flush=True)
            for _ in range(args.warmup_requests):
                row = await ab.generate(session, base, dataset["p512_c1"]["prompts"][0], 32, args.seed)
                validate_row(row, dataset["p512_c1"], 32)
                server.check()

            async def request(cell, variant, phase, repeat):
                server.check()
                row = await ab.generate(session, base, dataset[cell]["prompts"][variant], 32, args.seed)
                row.update(cell=cell, variant=variant, phase=phase, repeat=repeat, candidate_revision=head)
                ab.append_jsonl(cli.output / "requests.jsonl", row)
                validate_row(row, dataset[cell], 32)
                ab.check_reference(refs[cell, variant], row)
                server.check()

            for cell in ("p128_c1", "p128_c4"):
                status["active_cell"] = cell
                concurrency = dataset[cell]["concurrency"]
                for v in range(concurrency):
                    status.update(active_phase="cold_control", active_variant=v)
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    await request(cell, v, "cold_control", -1)
                    status["passed_cold_controls"] += 1
                for repeat in range(3):
                    status.update(active_phase="serial_repeat", active_repeat=repeat)
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    for v in range(concurrency):
                        await request(cell, v, "serial_repeat", repeat)
                    status["active_phase"] = "concurrent"
                    await ab.api(session, base, "/flush_cache?timeout=10", "POST")
                    tasks = [asyncio.create_task(request(cell, v, "concurrent", repeat)) for v in range(concurrency)]
                    try:
                        await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    status["passed_concurrent_rounds"] += 1
                    print(f"PASS {cell} round={repeat}", flush=True)
            status["status"] = "focused_original_control_gate_passed"
    except Exception as exc:
        status.update(status="safety_stopped", error_type=type(exc).__name__, error=str(exc))
        print("SAFETY STOP: " + str(exc), flush=True)
    finally:
        server.stop()
        status.update(managed_service_stopped=True, finished_at=time.time(),
                      original_controls_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == oracle_sha)
        ab.dump(cli.output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0 if status["status"] == "focused_original_control_gate_passed" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
