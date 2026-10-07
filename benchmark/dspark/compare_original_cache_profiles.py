"""CPU-only SHA-checked profile comparison; not a performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from trace_original_cache_profile import STAGES


def differences(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Profile snapshot shape/dtype mismatch")
    delta = (left.float() - right.float()).abs()
    return {"elements": left.numel(), "different_elements": int((left != right).sum()),
            "bitwise_equal": bool(torch.equal(left, right)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def load(folder):
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "original_profile_trace_complete":
        raise ValueError("Cannot interpret incomplete trace arm")
    stages = {}
    for path in sorted((folder / "verify_trace").glob("*.json")):
        meta = json.loads(path.read_text())
        tensor_path = path.with_suffix(".pt")
        if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != meta["tensor_sha256"]:
            raise ValueError("Snapshot SHA256 mismatch")
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        stages.setdefault(meta["stage"], []).append((meta, tensors))
    if set(stages) != STAGES or any(len(entries) != 2 for entries in stages.values()):
        raise ValueError("Expected two captures for each of five stages")
    for entries in stages.values():
        entries.sort(key=lambda item: item[0]["request_epoch"])
    return stages


def compare(left, right):
    lm, lt = left
    rm, rt = right
    for key in ("prompt_sha256", "prompt_length", "position", "anchor_token", "stage", "batch_size"):
        if lm[key] != rm[key]:
            raise ValueError("Snapshot identity mismatch: " + key)
    if lt.keys() != rt.keys():
        raise ValueError("Snapshot tensor set mismatch")
    return {name: differences(value, rt[name]) for name, value in lt.items()}


def identity(meta):
    return {key: meta.get(key) for key in ("prompt_sha256", "position", "anchor_token", "batch_size",
        "query_lengths", "backend_class", "prefill_split_tile_size", "wrapper_plan_info", "raw_argmax", "raw_top3")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--cache-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(2)
    det, cache = load(cli.deterministic), load(cli.cache_profile)
    report = {"scope": "original configuration diagnosis only; no oracle replacement or performance evidence",
              "stages": {}}
    for stage in sorted(STAGES):
        item = {"deterministic": identity(det[stage][0][0]), "cache_profile": identity(cache[stage][0][0]),
                "deterministic_repeat": compare(*det[stage]),
                "cache_profile_repeat": compare(*cache[stage]),
                "cross_profile": compare(det[stage][0], cache[stage][0])}
        if stage == "layer0.attention":
            item["written_kv_matches_inputs"] = {
                arm: {name: differences(tensors[f"current_{name}"].flatten(), tensors[f"written_{name}"].flatten())
                      for name in ("key", "value")}
                for arm, (_, tensors) in (("deterministic", det[stage][0]), ("cache_profile", cache[stage][0]))}
        report["stages"][stage] = item
        print(stage, {name: (value["different_elements"], value["max_abs"])
                      for name, value in item["cross_profile"].items()}, flush=True)
    repeats_equal = all(value["bitwise_equal"] for item in report["stages"].values()
                        for arm in ("deterministic_repeat", "cache_profile_repeat") for value in item[arm].values())
    report["two_repeats_bitwise_equal_within_each_profile"] = repeats_equal
    with cli.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if not repeats_equal:
        raise ValueError("Profile is not internally stable at observed stages")


if __name__ == "__main__":
    main()
