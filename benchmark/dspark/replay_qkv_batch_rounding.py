"""Read-only Qwen3 layer-0 projection rounding diagnosis, NOT a benchmark.

Use checkpoint weights and SHA-checked observed input rows. Duplicate that row
to reproduce the two actual M shapes. Only if BOTH original outputs reproduce
bitwise do precision-flag probes support a causal inference. No server changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def diff(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Replay output shape/dtype mismatch")
    return {"different_elements": int((a != b).sum()), "bitwise_equal": torch.equal(a, b),
            "max_abs": float((a.float() - b.float()).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    captures = []
    for path in args.captures.glob("*.json"):
        meta = json.loads(path.read_text())
        if meta["stage"] != "layer0.attention":
            continue
        if hashlib.sha256(path.with_suffix(".pt").read_bytes()).hexdigest() != meta["tensor_sha256"]:
            raise ValueError("Snapshot SHA mismatch")
        tensors = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        captures.append((meta, tensors, path.name))
    captures.sort(key=lambda item: item[0]["request_epoch"])
    if len(captures) != 3 or [item[0]["batch_size"] for item in captures] != [1, 1, 4]:
        raise ValueError("Requires original B1, B1 repeat, B4 committed-anchor snapshots")
    serial, _, concurrent = captures
    for name in ("prompt_sha256", "position", "anchor_token"):
        if serial[0][name] != concurrent[0][name]:
            raise ValueError("Unequal committed anchor identity")
    if not torch.equal(serial[1]["qkv_input"], concurrent[1]["qkv_input"]):
        raise ValueError("Inputs differ; a pure M-shape replay is not a valid diagnosis")
    config = json.loads((args.model / "config.json").read_text())
    if (config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]) != (5120, 40, 8):
        raise ValueError("Only the checked unquantized TP1 Qwen3 geometry is supported")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    pieces, provenance = [], {}
    for name in ("q_proj", "k_proj", "v_proj"):
        key = f"model.layers.0.self_attn.{name}.weight"
        if key.replace(".weight", ".bias") in index:
            raise ValueError("Replay only supports the observed bias-free Qwen3 projection")
        with safe_open(args.model / index[key], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        pieces.append(value)
        provenance[key] = {"checkpoint_file": index[key], "shape": list(value.shape),
                           "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    weight_cpu = torch.cat(pieces)
    if weight_cpu.shape != (7168, 5120) or weight_cpu.dtype != torch.bfloat16:
        raise ValueError("Unexpected checkpoint QKV weight geometry/dtype")
    weight = weight_cpu.cuda()
    row = serial[1]["qkv_input"].cuda()
    inputs = {str(item[0]["batch_size"]): row.unsqueeze(0).repeat(item[0]["tensor_row_count"], 1)
              for item in (serial, concurrent)}
    expected = {str(item[0]["batch_size"]): item[1]["qkv_output"] for item in (serial, concurrent)}
    flags = torch.backends.cuda.matmul
    original = {"allow_bf16_reduced_precision_reduction": flags.allow_bf16_reduced_precision_reduction,
                "allow_tf32": flags.allow_tf32}
    variants = [("original", {})]
    variants.append(("disable_bf16_reduced_precision_reduction", {"allow_bf16_reduced_precision_reduction": False}))
    report = {"scope": "read-only arithmetic diagnosis; no timing or performance claim", "weights": provenance,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "original_flags": original,
              "capture_names": [item[2] for item in captures], "actual_M": {key: len(value) for key, value in inputs.items()},
              "probes": {}}
    for name, changes in variants:
        for key, value in changes.items():
            setattr(flags, key, value)
        outputs = {key: F.linear(value, weight)[0].cpu() for key, value in inputs.items()}
        if any(not torch.isfinite(value).all() for value in outputs.values()):
            raise ValueError("Nonfinite projection replay")
        report["probes"][name] = {"vs_observed": {key: diff(value, expected[key]) for key, value in outputs.items()},
                                 "B1_vs_B4": diff(outputs["1"], outputs["4"]),
                                 "B4_vs_original_B1": diff(outputs["4"], expected["1"])}
        for key, value in original.items():
            setattr(flags, key, value)
    report["original_shape_replay_verified"] = all(item["bitwise_equal"] for item in report["probes"]["original"]["vs_observed"].values())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "weights"}), flush=True)


if __name__ == "__main__":
    main()
