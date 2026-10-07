"""Replay original h21 gate with opt-in read-only first-anchor observations.

Only the harness launch command gains forward hooks. No server/model arithmetic
is patched. The original gate still stops on the first output/acceptance change.
The partial h21 records locate a diagnostic focus, not a complete usable oracle.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import cache_profile_correctness as gate
import dspark_prefix_ab as ab
from dspark_verify_trace_hook import prompt_hash


def focus_config(baseline, folder):
    data = json.loads((baseline / "dataset.json").read_text())
    refs = [json.loads(line) for line in (baseline / "requests.jsonl").read_text().splitlines()]
    refs = [row for row in refs if row["phase"] == "reference" and row["cell"] == "p128_c4"]
    if len(refs) != 4 or {row["variant"] for row in refs} != set(range(4)):
        raise ValueError("Requires original h21 p128_c4 cold diagnostic controls")
    original = next(row for row in refs if row["variant"] == 0)
    if original["original_revision"] != gate.ORIGINAL or original["cached_tokens"]:
        raise ValueError("Focus is not an original cache-off request")
    prompt = data["p128_c4"]["prompts"][0]
    return {"folder": str(folder / "verify_trace"), "prompt_sha256": prompt_hash(prompt),
            "prompt_length": len(prompt), "position": len(prompt), "anchor_token": original["output_ids"][0],
            "kv_layer": 0, "max_kv_prefix": 4096, "max_captures": 15}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trace-baseline", type=Path, required=True)
    parser.add_argument("--observe-projection", action="store_true")
    cli, rest = parser.parse_known_args()
    if "--output" not in rest:
        raise ValueError("A new output folder is required")
    folder = Path(rest[rest.index("--output") + 1])
    config = focus_config(cli.trace_baseline, folder)
    if cli.observe_projection:
        config.update(observe_layer0_projection=True, max_projection_rows=1024)

    class ReadOnlyTraceServer(ab.ManagedServer):
        def command(self):
            cmd, env = super().command()
            modules = [
                    "model.layers.0.self_attn.attn", "model.layers.0", "model.layers.19",
                    "model.layers.39", "logits_processor"]
            if cli.observe_projection:
                modules += ["model.layers.0.input_layernorm", "model.layers.0.self_attn.qkv_proj"]
            cmd += ["--forward-hooks", json.dumps([{
                "name": "original-cache-batch-read-only", "target_modules": modules,
                "hook_factory": "dspark_verify_trace_hook:make_hook", "config": config}])]
            env["PYTHONPATH"] += ":" + str(Path(__file__).resolve().parent)
            return cmd, env

    # Harness class only; the child imports original a61e0bb model code unchanged.
    gate.ab.ManagedServer = ReadOnlyTraceServer
    sys.argv = [sys.argv[0], *rest]
    result = asyncio.run(gate.main())
    provenance = {"focus_config": config, "original_diagnostic_source": str(cli.trace_baseline),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name("dspark_verify_trace_hook.py"))},
        "capture_count": len(list((folder / "verify_trace").glob("*.json"))),
        "scope": "read-only first committed anchor; no future draft row or performance evidence"}
    ab.dump(folder / "trace_provenance.json", provenance)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
