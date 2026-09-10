# Skill-IR Stage 2 closure

Status: **closed and validated on the synthetic hidden-protocol task**.

## Thesis

The persistent skill is semantic state. LoRA, IA3, and other parameterizations
are executor-specific compiled artifacts. The validated path is:

```text
demonstrations → semantic acquisition → persistent Skill IR
→ optional typed composition → executor compiler → ephemeral fast parameters
→ frozen base-model execution
```

## What was established

Stage 2A showed that the legacy 2048-dimensional hash IR can be compressed to
K=512 while quantized reconstruction preserves essentially oracle execution.
Stage 2B then separated opaque lexical identity from relational semantics and
validated exact hybrid persistence at K=16 and K=8. Stage 2C validated an
injective discrete semantic code: all 72 legal relation states use distinct
hard codes.

Stage 2D established that exact held-out recombination requires explicit typed
factor modularity. The unrestricted interaction and shared additive composers
failed; the hard factor-separable composer passed all coverage-safe folds,
including untouched-factor and bit-exact IR gates.

Stage 2E–H established zero-step executor independence across LoRA rank,
target profile, mechanism, and base model. The same persisted IR compiled to
rank-4/T1 LoRA, rank-8/T3 LoRA, IA3, Qwen3-0.6B, and SmolLM2-135M paths while
matched behavior remained better than shuffled and constant controls.

The complete experiment index, exact artifact paths, and status of positive
and negative results are in [artifacts/stage2_manifest.json](../artifacts/stage2_manifest.json).

## Preserved negative results

The following are evidence, not disposable failed attempts:

- anonymous support latents collapsed toward generic adaptation;
- vanilla VQ aliased the 72 states;
- unrestricted interaction composition failed held-out recombination;
- shared additive composition failed held-out recombination;
- fixed-pool rank-8 compiler training left constant IR too competitive.

They motivate explicit semantics, injective coding, typed modularity, and
streaming compiler protocol diversity respectively.

## Final defensible claim

On this synthetic task, demonstrations can be converted into a compact
structured/discrete persistent executable Skill IR. It supports exact typed
composition and zero-optimization compilation into multiple LoRA ranks,
target profiles, IA3, and two frozen base models with protocol-specific causal
behavior.

This does not establish natural-task semantic induction, ontology discovery,
or fully model-independent natural semantics. The semantic factors were still
manually supplied by the hidden-protocol task.

## Canonical references

- Stage 1 source/configuration and result directories are historical and are
  recoverable from git tag `pre-stage4-stage3-full-history`.
- Stage 2 experiment runners/configurations and bulk results are historical;
  compact conclusions are preserved in `artifacts/stage2_manifest.json` and
  `docs/experiment_history.md`.
- The maintained executor boundary is the source modules
  `src/tool_lora/skill_ir/protocol_ir.py`, `compiler.py`, and `executors.py`.

GPU reference runs are intentionally separate from the lightweight unit
suite. They require the local model cache and the manifest-linked checkpoints.

## Stage 3 boundary

Stage 3 must not reopen compression or add another executor variant. Its first
controlled question is whether semantic structure can be induced when the
ontology is not manually supplied.
