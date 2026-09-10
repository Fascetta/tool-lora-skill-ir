# Skill-IR Stage 1 — final technical record

## Status

**STAGE 1: COMPLETE** (2026-08-18). The canonical command and artifact
manifest are in [`artifacts/stage1_manifest.json`](../artifacts/stage1_manifest.json).

## Research question and claim boundary

On the synthetic hidden-protocol task, can support demonstrations become a
persistent explicit executable protocol representation that an independently
trained compiler can turn into fast weights for a frozen Qwen? Stage 1 answers
yes for the validated Qwen3-0.6B, rank-4/T1 executor. It does not claim a
universal, model-independent, compositional, continual, or arbitrary-executor
Skill IR.

## Validated architecture

```text
support S -> Hourglass(Qwen_frozen(S)) -> relational acquisition
           -> nine semantic values -> exact legacy ProtocolIR b_hat
           -> frozen OracleCompiler G -> ephemeral LoRA fast weights
           -> Qwen_frozen(query; DeltaW)
```

Functions (`fn_1`, `fn_2`) and enums (`enum_1`, `enum_2`) use contextual
candidate-relative relational inference. Fields use a joint six-way
permutation scorer over city/date/units and opaque key candidates. Date order
uses a componentwise permutation scorer between source `Y,M,D` and rendered
positions. Date separator uses a deterministic lexical/token-sequence pointer;
it is a copy relation, not a learned global classifier.

The learned date/field/function/enum predictions can be written as
`p_i(j) = RelationalScore(slot_i, contextual_candidate_j(H))`. For fields,
`p_field(pi)` is the softmax over the six permutations; date order is the
softmax over `{DMY, MDY, YMD}`.

## Nine-component explicit IR

The persistent schema is:

`fn_1, fn_2, field_1, field_2, field_3, enum_1, enum_2, date_order, date_separator`.

For value `x` at slot `i`,

```text
e_i(x) = one_hot(little_endian_uint32(SHA256(f"{i}:{x}")) % 2048)
b_hat  = sum_i e_i(x_hat_i), reshaped as [8,256]
```

The reshape does not imply eight semantic components. V6-L tested 100 random
protocols with 100 exact reconstructions and zero maximum error.

## Compiler and executor

The independently trained self-describing V6-G compiler is a 2048 -> 1024 ->
176,160,768 MLP with GELU. It emits 56 ordered A/B LoRA paths for all 28
layers, targeting `q_proj` and `v_proj`, rank 4, alpha 16 (T1). Its only
input is the explicit legacy IR tensor. It receives no support, query, or
contextual acquisition state.

## Persistence and zero-step reconnect

Temporary acquisition state includes support text, Qwen contextual states,
hourglass states, candidate states, and relational attention. The persistent
`ExplicitSkillIR` contains only nine semantic values, the exact `[8,256]`
legacy tensor, schema version, and optional confidence metadata. At execution,
the persistent IR is compiled into ephemeral LoRA fast weights.

V6-N loaded frozen V6-M and V6-G checkpoints in a fresh process with
`optimizer_steps = 0`. “Zero-step” means no LoRA fast-weight training,
calibration, or fine-tuning; the frozen compiler **does generate** LoRA fast
weights at inference time.

For every fully correct nine-component prediction, the evaluation asserts
bitwise `b_hat_hard == b*`, equal compiler outputs, equal generated LoRA
factors, and equal logits. Counterfactual support shares query, target,
candidate inventory/order, compiler, executor, and frozen base; only the
support protocol changes, enforced through `group_key` routing.

## Results and controls

The prior V6-N 64-group evaluation reported CE: oracle matched 1.993, oracle
shuffled 2.166, hard matched 1.993, hard shuffled 2.167, constant IR 1.989,
and frozen base 5.342. These totals include generic syntax tokens, so the
closure record also treats matched-vs-shuffled and binding-sensitive measures
as the protocol-specific evidence. The final machine-readable closure output
is `results/stage1_final/metrics.json`.

## What is not validated

Anonymous Perceiver Z alone was not sufficient; the old flattened-vector V6-B
probe and old seven-row/staging nine-row decoders were invalid/unstable;
contextual candidate-state bypass confounded early B2-A; context-free
candidate plus global support representations failed; the learned separator
matcher failed fresh generalization; independent field heads were unstable,
while joint permutation was robust. Total CE alone is not a sufficient
protocol-binding metric and must not be interpreted as “matched IR beats
constant IR” when it does not.

## Limitations and Stage 2 boundary

The current IR is explicit/procedural and hash-based, 2048-dimensional; the
compiler is validated only for this Qwen + LoRA r4/T1 executor; acquisition is
intentionally heterogeneous; and separator copying is deterministic. Stage 1
does not demonstrate basis compression, sparse coordinates, composition,
multi-executor compilation, cross-model transfer, or continual memory. Stage 2
may ask whether this executable IR can be compressed into a compact learned
persistent representation, but no such work is part of this repository closure.

## Reproduction

The original Stage-1 command and configuration are historical source,
recoverable from git tag `pre-stage4-stage3-full-history`; they are not
maintained commands.
