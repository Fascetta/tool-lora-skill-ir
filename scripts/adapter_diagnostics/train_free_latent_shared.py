#!/usr/bin/env python
"""P2-A: jointly learn free tool latents and one shared LoRA generator."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from tool_lora.evaluation.tool_call_metrics import aggregate_scores, score_tool_call
from tool_lora.functional_hypernet.data import build_functional_prompt_from_oracle_row, canonical_target_text, load_jsonl
from tool_lora.functional_hypernet.functional_lora import apply_functional_lora, discover_target_linears
from tool_lora.functional_hypernet.losses import masked_cross_entropy, masked_distillation_kl
from tool_lora.functional_hypernet.model import FreeLatentToolLoRAMetanetwork, FunctionalHyperNetConfig
from tool_lora.functional_hypernet.functional_lora import load_peft_lora
from tool_lora.functional_hypernet.runtime import collate_functional_rows, tokenize_functional_row
from tool_lora.utils.json_utils import loads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--tool-ids", nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--examples-per-tool", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--latent-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--parameterization", choices=("direct_ab", "svd", "fixed_a"), default="direct_ab")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--objective", choices=("ce", "delta_norm", "factor_norm", "distill", "distill_delta_norm"), default="ce")
    p.add_argument("--lambda-regularization", type=float, default=1e-3)
    p.add_argument("--lambda-distill", type=float, default=0.5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--teacher-root", type=Path, default=None)
    p.add_argument("--latent-group-manifest", type=Path, default=None)
    p.add_argument("--eval-tool-ids", nargs="+", default=None)
    p.add_argument("--tools-per-step", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_specs = discover_target_linears(model, ("q_proj", "v_proj"), tuple(range(28)))
    config = FunctionalHyperNetConfig(
        documentation_embedding_size=args.latent_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
        target_layers=tuple(range(28)),
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        output_parameterization=args.parameterization,
        output_scale=0.1,
        bounded_output=True,
    )
    latent_group_names = list(args.tool_ids)
    tool_to_latent = None
    if args.latent_group_manifest is not None:
        groups = json.loads(args.latent_group_manifest.read_text())
        if not isinstance(groups, dict) or any(tool not in groups for tool in args.tool_ids):
            raise ValueError("Latent-group manifest must map every tool ID to a group")
        latent_group_names = sorted({str(groups[tool]) for tool in args.tool_ids})
        group_indices = {name: index for index, name in enumerate(latent_group_names)}
        tool_to_latent = [group_indices[str(groups[tool])] for tool in args.tool_ids]
    metanetwork = FreeLatentToolLoRAMetanetwork(
        config,
        target_specs,
        len(latent_group_names),
        args.latent_size,
        tool_to_latent=tool_to_latent,
    ).to(device)
    teacher_cache = {}
    if args.objective in {"distill", "distill_delta_norm"}:
        if args.teacher_root is None:
            raise ValueError("--teacher-root is required for distillation objectives")
        for tool in args.tool_ids:
            teacher_cache[tool] = load_peft_lora(args.teacher_root / tool, model)
    train_rows, val_rows = {}, {}
    for tool in args.tool_ids:
        train_rows[tool] = [tokenize_functional_row(row, tokenizer, args.max_seq_len) for row in load_jsonl(args.dataset_root / tool / "train.jsonl")[:256]]
        val_rows[tool] = load_jsonl(args.dataset_root / tool / "val.jsonl")[:64]
        if not train_rows[tool] or not val_rows[tool]:
            raise ValueError(f"Missing train/validation rows for {tool}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "stage4_p2_a_free_latent_shared_generator",
        "tool_ids": args.tool_ids,
        "seed": args.seed,
        "model": str(args.model),
        "config": config.to_dict(),
        "latent_size": args.latent_size,
        "max_steps": args.max_steps,
        "examples_per_tool": args.examples_per_tool,
        "tools_per_step": args.tools_per_step,
        "latent_groups": latent_group_names,
        "tool_to_latent": tool_to_latent,
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.smoke:
        print(json.dumps({"tools": args.tool_ids, "train_rows": {k: len(v) for k, v in train_rows.items()}, "val_rows": {k: len(v) for k, v in val_rows.items()}, "trainable_parameters": sum(p.numel() for p in metanetwork.parameters() if p.requires_grad)}, indent=2))
        return
    optimizer = torch.optim.AdamW(metanetwork.parameters(), lr=args.learning_rate, weight_decay=0.01)
    logs = []
    started = time.perf_counter()
    for step in range(args.max_steps):
        metanetwork.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        if args.tools_per_step <= 0 or args.tools_per_step >= len(args.tool_ids):
            selected_tool_indices = list(range(len(args.tool_ids)))
        else:
            start_index = (step * args.tools_per_step) % len(args.tool_ids)
            selected_tool_indices = [
                (start_index + offset) % len(args.tool_ids)
                for offset in range(args.tools_per_step)
            ]
        component_totals = {"ce": 0.0, "distill": 0.0, "regularizer": 0.0}
        for tool_index in selected_tool_indices:
            tool = args.tool_ids[tool_index]
            start = (step * args.examples_per_tool) % len(train_rows[tool])
            selected = [train_rows[tool][(start + j) % len(train_rows[tool])] for j in range(args.examples_per_tool)]
            batch = collate_functional_rows(selected, pad_token_id=tokenizer.pad_token_id, device=device)
            factors = metanetwork(torch.tensor([tool_index], device=device))[0]
            with apply_functional_lora(model, factors, scaling=metanetwork.scaling):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            ce = masked_cross_entropy(logits, batch["labels"])
            total = ce
            component_totals["ce"] += float(ce.detach())
            if args.objective in {"delta_norm", "factor_norm", "distill_delta_norm"}:
                if args.objective == "factor_norm":
                    regularizer = torch.stack([(pair.A.float().pow(2).mean() + pair.B.float().pow(2).mean()) for pair in factors.values()]).mean()
                else:
                    regularizer = torch.stack([((pair.B.float() @ pair.A.float()) * metanetwork.scaling).pow(2).mean() for pair in factors.values()]).mean()
                total = total + args.lambda_regularization * regularizer
                component_totals["regularizer"] += float(regularizer.detach())
            if args.objective in {"distill", "distill_delta_norm"}:
                teacher_factors, teacher_scaling, _ = teacher_cache[tool]
                with torch.no_grad(), apply_functional_lora(model, teacher_factors, scaling=teacher_scaling):
                    teacher_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                distill = masked_distillation_kl(logits, teacher_logits, batch["labels"], temperature=args.distill_temperature)
                total = total + args.lambda_distill * distill
                component_totals["distill"] += float(distill.detach())
            losses.append(total)
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(metanetwork.parameters(), 1.0))
        optimizer.step()
        divisor = float(len(selected_tool_indices))
        log = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "ce": component_totals["ce"] / divisor,
            "distill": component_totals["distill"] / divisor,
            "regularizer": component_totals["regularizer"] / divisor,
            "weighted_distill": args.lambda_distill * component_totals["distill"] / divisor,
            "weighted_regularizer": args.lambda_regularization * component_totals["regularizer"] / divisor,
            "grad_norm": grad_norm,
            "elapsed_seconds": time.perf_counter() - started,
        }
        logs.append(log)
        if (step + 1) % 10 == 0 or step == 0:
            print(json.dumps(log), flush=True)
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps:
            evaluation_tools = args.eval_tool_ids or args.tool_ids
            evaluation = evaluate(metanetwork, model, tokenizer, evaluation_tools, val_rows, device)
            evaluation["generated_geometry"] = generated_geometry(metanetwork, args.tool_ids, device)
            (args.output_dir / f"validation-{step + 1}.json").write_text(json.dumps(evaluation, indent=2) + "\n")
            torch.save({"model": metanetwork.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1, "metadata": metadata}, args.output_dir / f"checkpoint-{step + 1}.pt")
            print(json.dumps({"validation_step": step + 1, "metrics": evaluation["macro"]}, indent=2), flush=True)
    (args.output_dir / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    (args.output_dir / "training_summary.json").write_text(json.dumps({"completed_steps": args.max_steps, "runtime_seconds": time.perf_counter() - started, "trainable_parameters": sum(p.numel() for p in metanetwork.parameters() if p.requires_grad)}, indent=2) + "\n")


def evaluate(metanetwork, model, tokenizer, tool_ids, val_rows, device):
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    all_scores = []
    per_tool = {}
    for tool_index, tool in enumerate(tool_ids):
        factors = metanetwork(torch.tensor([tool_index], device=device))[0]
        scores = []
        for start in range(0, len(val_rows[tool]), 8):
            rows = val_rows[tool][start : start + 8]
            batch = tokenizer([r["prompt"] for r in rows], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.inference_mode(), apply_functional_lora(model, factors, scaling=metanetwork.scaling):
                generated = model.generate(**batch, max_new_tokens=128, do_sample=False)
            prompt_len = batch["input_ids"].shape[1]
            for row, output in zip(rows, generated):
                prediction = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                score = score_tool_call(prediction, loads(canonical_target_text(row)))
                scores.append(score)
                all_scores.append(score)
        per_tool[tool] = aggregate_scores(scores)
    model.config.use_cache = False
    return {"macro": aggregate_scores(all_scores), "per_tool": per_tool}


def generated_geometry(metanetwork, tool_ids, device):
    """Summarize generated factors without materializing dense model weights."""
    result = {}
    metanetwork.eval()
    with torch.no_grad():
        for tool_index, tool in enumerate(tool_ids):
            factors = metanetwork(torch.tensor([tool_index], device=device))[0]
            delta_norms = []
            spectral_norms = []
            ranks = []
            a_norms = []
            b_norms = []
            top_energy = []
            entropy_ranks = []
            participation_ranks = []
            module_geometry = {}
            for pair in factors.values():
                a = pair.A.float()
                b = pair.B.float()
                delta = b @ a * metanetwork.scaling
                # The nonzero spectrum of BA is the spectrum of an r x r core.
                q_b, r_b = torch.linalg.qr(b, mode="reduced")
                q_a, r_a = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
                del q_b, q_a
                singular = torch.linalg.svdvals(r_b @ r_a.transpose(0, 1)) * metanetwork.scaling
                threshold = singular.max() * 1e-4 if singular.numel() else singular.new_tensor(0.0)
                energy = singular.float().pow(2)
                raw_total_energy = energy.sum()
                if float(raw_total_energy) <= 1e-12:
                    total_energy = energy.new_tensor(1.0)
                    top_fraction = energy.new_tensor(0.0)
                    entropy_rank = energy.new_tensor(0.0)
                    participation_rank = energy.new_tensor(0.0)
                else:
                    total_energy = raw_total_energy
                    probabilities = energy / total_energy
                    top_fraction = energy[0] / total_energy
                    entropy_rank = torch.exp(
                        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                    )
                    participation_rank = total_energy.pow(2) / energy.pow(2).sum()
                module_geometry_path = next(
                    path for path, candidate in factors.items() if candidate is pair
                )
                module_geometry[module_geometry_path] = {
                    "delta_frobenius": float(delta.norm()),
                    "delta_spectral": float(singular.max()),
                    "effective_rank_1e-4": int((singular > threshold).sum()),
                    "top_energy_fraction": float(top_fraction),
                    "entropy_effective_rank": float(entropy_rank),
                    "participation_rank": float(participation_rank),
                    "singular_values": [float(value) for value in singular],
                }
                delta_norms.append(float(delta.norm()))
                spectral_norms.append(float(singular.max()))
                ranks.append(float((singular > threshold).sum()))
                a_norms.append(float(a.norm()))
                b_norms.append(float(b.norm()))
                top_energy.append(float(top_fraction))
                entropy_ranks.append(float(entropy_rank))
                participation_ranks.append(float(participation_rank))
            result[tool] = {
                "modules": module_geometry,
                "mean_delta_frobenius": sum(delta_norms) / len(delta_norms),
                "mean_delta_spectral": sum(spectral_norms) / len(spectral_norms),
                "mean_effective_rank": sum(ranks) / len(ranks),
                "mean_top_energy_fraction": sum(top_energy) / len(top_energy),
                "mean_entropy_effective_rank": sum(entropy_ranks) / len(entropy_ranks),
                "mean_participation_rank": sum(participation_ranks) / len(participation_ranks),
                "mean_a_frobenius": sum(a_norms) / len(a_norms),
                "mean_b_frobenius": sum(b_norms) / len(b_norms),
            }
    return result


if __name__ == "__main__":
    main()
