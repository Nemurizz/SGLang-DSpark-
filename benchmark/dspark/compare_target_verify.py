"""CPU-only, SHA-checked same-committed-anchor comparison, not timing."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from trace_target_verify import STAGES


def differences(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Snapshot shape/dtype mismatch")
    delta = (left.float() - right.float()).abs()
    return {"elements": left.numel(), "different_elements": int((left != right).sum()),
            "bitwise_equal": bool(torch.equal(left, right)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def load(folder):
    if json.loads((folder / "status.json").read_text())["status"] != "target_verify_trace_complete":
        raise ValueError("Incomplete trace arm")
    stages = {}
    for path in sorted((folder / "verify_trace").glob("*.json")):
        meta = json.loads(path.read_text())
        tensor_path = path.with_suffix(".pt")
        if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != meta["tensor_sha256"]:
            raise ValueError("Snapshot SHA256 mismatch")
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        stages.setdefault(meta["stage"], []).append((meta, tensors))
    if set(stages) != STAGES or any(len(rows) != 2 for rows in stages.values()):
        raise ValueError("Requires two repeats of each stage")
    for rows in stages.values():
        rows.sort(key=lambda row: row[0]["request_epoch"])
    return stages


def compare(left, right):
    lm, lt = left
    rm, rt = right
    for name in ("prompt_sha256", "prompt_length", "position", "anchor_token", "stage", "batch_size"):
        if lm[name] != rm[name]:
            raise ValueError("Committed-anchor identity mismatch: " + name)
    if lt.keys() != rt.keys():
        raise ValueError("Tensor sets differ")
    return {name: differences(value, rt[name]) for name, value in lt.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    arms = {label: load(args.trace / label) for label in ("ordinary_target", "dspark_full")}
    report = {"scope": "diagnosis; failed h26 independent-target correctness gate remains failed", "stages": {}}
    keys = ("prompt_sha256", "position", "anchor_token", "batch_size", "query_lengths", "mode",
            "backend_class", "raw_argmax", "raw_top3")
    for stage in sorted(STAGES):
        left, right = (arms[label][stage] for label in ("ordinary_target", "dspark_full"))
        item = {label: {key: rows[0][0].get(key) for key in keys}
                for label, rows in (("ordinary_target", left), ("dspark_full", right))}
        item.update(ordinary_repeat=compare(*left), dspark_repeat=compare(*right),
                    decode_vs_verify=compare(left[0], right[0]))
        if stage == "layer0.attention":
            item["written_kv_matches_inputs"] = {
                label: {name: differences(tensors[f"current_{name}"].flatten(), tensors[f"written_{name}"].flatten())
                        for name in ("key", "value")}
                for label, (_, tensors) in (("ordinary_target", left[0]), ("dspark_full", right[0]))}
        report["stages"][stage] = item
        print(stage, {name: (value["different_elements"], value["max_abs"])
                      for name, value in item["decode_vs_verify"].items()}, flush=True)
    equal = all(value["bitwise_equal"] for item in report["stages"].values()
                for arm in ("ordinary_repeat", "dspark_repeat") for value in item[arm].values())
    report["two_repeats_bitwise_equal_within_each_arm"] = equal
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if not equal:
        raise ValueError("Observed arms are not internally bitwise stable")


if __name__ == "__main__":
    main()
