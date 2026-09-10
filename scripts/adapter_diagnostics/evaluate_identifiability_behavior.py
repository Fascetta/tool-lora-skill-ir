#!/usr/bin/env python
"""Evaluate same-tool adapters on shared validation prompts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from _bootstrap import add_src_to_path

add_src_to_path()

from tool_lora.evaluation.tool_call_metrics import score_tool_call
from tool_lora.utils.json_utils import iter_jsonl, loads


def predicted_tool(text: str) -> str | None:
    try:
        value = json.loads(text)
        return value.get("tool") if isinstance(value, dict) else None
    except Exception:
        return None


def paths(root: Path) -> list[tuple[int, str, Path]]:
    out = []
    for p in sorted(root.glob("seed_*/qwen3_0_6b/*/adapter_model.safetensors")):
        out.append((int(p.parents[2].name.removeprefix("seed_")), p.parents[0].name, p.parent))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    found = paths(args.root)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    result: dict[str, Any] = {"experiment": "multi_seed_lora_behavior_identifiability", "num_adapters": len(found), "groups": {}}
    for tool in sorted({tool for _, tool, _ in found}):
        rows = list(iter_jsonl(args.dataset_root / tool / "val.jsonl"))
        if args.limit is not None:
            rows = rows[: args.limit]
        predictions: dict[int, list[dict[str, Any]]] = {}
        for seed, _, adapter in sorted((x for x in found if x[1] == tool), key=lambda x: x[0]):
            base = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, str(adapter)).to(args.device)
            model.eval()
            pred_rows = []
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start : start + args.batch_size]
                batch = tokenizer([r["prompt"] for r in batch_rows], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.device)
                with torch.inference_mode():
                    generated = model.generate(**batch, max_new_tokens=128, do_sample=False)
                prompt_len = batch["input_ids"].shape[1]
                for row, output in zip(batch_rows, generated):
                    text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                    gold = loads(row["target"])
                    pred_rows.append({"example_id": row.get("example_id"), "prediction": text, "predicted_tool": predicted_tool(text), "scores": score_tool_call(text, gold)})
            predictions[seed] = pred_rows
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
        seeds = sorted(predictions)
        per_seed = {}
        for seed in seeds:
            scores = predictions[seed]
            per_seed[str(seed)] = {
                "num_examples": len(scores),
                "correct_tool_rate": sum(float(x["scores"].get("correct_tool_name", 0)) for x in scores) / max(len(scores), 1),
                "exact_argument_rate": sum(float(x["scores"].get("exact_argument_match", 0)) for x in scores) / max(len(scores), 1),
                "valid_json_rate": sum(float(x["scores"].get("valid_json", 0)) for x in scores) / max(len(scores), 1),
            }
        pairwise = []
        for i, left in enumerate(seeds):
            for right in seeds[i + 1 :]:
                lp, rp = predictions[left], predictions[right]
                same_text = sum(a["prediction"].strip() == b["prediction"].strip() for a, b in zip(lp, rp)) / max(len(lp), 1)
                same_tool = sum(a["predicted_tool"] == b["predicted_tool"] for a, b in zip(lp, rp)) / max(len(lp), 1)
                both_correct_tool = sum(a["scores"].get("correct_tool_name", 0) and b["scores"].get("correct_tool_name", 0) for a, b in zip(lp, rp)) / max(len(lp), 1)
                pairwise.append({"seed_left": left, "seed_right": right, "exact_prediction_agreement": same_text, "tool_prediction_agreement": same_tool, "both_correct_tool_rate": both_correct_tool})
        result["groups"][tool] = {"per_seed": per_seed, "pairwise": pairwise}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"num_adapters": len(found), "out": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
