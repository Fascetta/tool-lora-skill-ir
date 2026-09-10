# Persistent Skill Representations for Tool-Using Language Models

Course project by **Christian Bianchi** for **Deep Learning and Applied AI, Sapienza University of Rome (2026)**.

This repository contains the complete project developed by Christian Bianchi after branching from the original `tool-lora-hypernet` codebase. It covers the research sequence from adapter-space diagnostics to the latest completed Skill IR transfer audit. Code authored by Luca Romani is not reproduced as project code here. Values originating from the inherited baseline are explicitly labelled as comparison baselines.

## Research question

A **tool-use skill** is a rule that maps a user request and observable environment state to a structured tool call. A **support set** is the set of demonstrations available while a skill is learned.

The project asks:

> Can a support set be converted into a persistent representation of a tool-use skill that remains sufficient for execution after the support set is deleted?

The project evaluates two candidate representations:

- An **execution adapter** is a temporary low-rank adaptation (LoRA) applied to a language model during execution.
- **Skill IR** is a serialized intermediate representation containing relations, behavioral effects, and opaque lexical identifiers learned from the support set.

The distinction is central: Skill IR is the persistent object; an execution adapter is temporary execution state.

## Project progression

### Phase 1 — Adapter parameters as the representation

The initial approach attempted to predict execution adapters from tool information. The inherited evaluation framework supplied three comparison points on a 15-tool validation split: nearest-neighbor adapter retrieval achieved **94.31% correct-tool accuracy**, an independently trained per-tool adapter achieved **99.34%**, and the strongest generated adapter achieved **79.36%**.

Christian's subsequent experiments investigated why generated adapters failed:

1. **Functional-delta supervision** added losses on the effective LoRA update and on the language model's output distribution.
2. **Adapter-identifiability audit** trained multiple adapters for the same tool. The adapters preserved tool identity while their effective updates remained weakly aligned, showing that independently optimized adapter weights are not a unique semantic target.
3. **Free-latent shared generator** removed semantic conditioning and learned one free vector per tool. It could fit training behavior but did not recover the validation behavior of independently trained adapters.
4. **Regularization and distillation controls** tested output-norm penalties and functional teacher distillation.
5. **Structured output controls** tested explicit singular-value generation and a fixed first LoRA factor while generating only the second factor.
6. **Diversity controls** separated surface variation from genuinely different functional tasks.

These controls did not establish generated LoRA parameters as a stable persistent representation. The scripts are retained under `scripts/adapter_diagnostics/`; raw inherited training infrastructure is intentionally excluded.

### Phase 2 — Controlled persistent Skill IR

The second approach represents the learned skill explicitly. Skill IR stores:

- **relational records**, which specify how request values determine tool-call values;
- **behavioral effects**, which distinguish state-preserving from state-changing operations when this information is available;
- a **lexical sidecar**, which stores opaque API names and literal values that cannot be reconstructed from relations alone.

The controlled pipeline is:

```text
support set
  -> relation acquisition
  -> component correspondence
  -> canonicalization
  -> Skill IR
  -> support-set deletion
  -> Skill-IR reload
  -> resolver
  -> tool execution
```

**Component correspondence** identifies which source component supplies each target component. The frozen learned model **P6-R1** contains 13,281 parameters. It predicts a permutation using byte-level component encodings and a one-to-one assignment. The **exact matcher** is a deterministic control that enumerates permutations; it applies only when every transformed source component is directly observable in the target.

On the immutable 72-case benchmark:

| Correspondence method | Any correct | Unique correct | Reload | Time/case |
|---|---:|---:|---:|---:|
| P6-R1 | 100% | 95.83% | 100% | 4.12 ms |
| Exact matcher | 100% | 100% | 100% | 0.45 ms |
| Rule-based acquisition | 100% | 100% | 100% | 0.34 ms |

**Any correct** means that the predicted set contains the benchmark permutation. **Unique correct** means that the benchmark permutation is the only prediction. **Reload** means that serialization and deserialization preserve Skill IR exactly.

This phase establishes persistence and information sufficiency in the controlled domain. It does not establish discovery of the evaluator's hidden semantic decomposition.

### Phase 3 — Transfer to AppWorld-style tools

The final phase tests whether the representation and acquisition logic transfer from synthetic dispatch operations to public schemas derived from AppWorld.

The main findings are:

- Oracle Skill IR and deterministic acquisition achieve perfect matched and counterfactual action, binding, and state accuracy in the evaluated AppWorld fixtures.
- Learned correspondence remains stable across three predeclared random seeds after a supervision-coordinate defect was corrected.
- A generic resolver fails on unseen Simple Note tools because two public APIs are structurally compatible with the same abstract roles. This is **resolver-interface non-identifiability**: the available representation does not uniquely determine the concrete API mapping.
- Adding an **effect anchor**, a stored description of whether an operation preserves or changes state, makes the mapping unique and restores perfect matched and counterfactual execution in the evaluated fixtures.
- A learned effect predictor fits its synthetic training data but initially memorizes schema semantics. Deconfounded training restores perfect matched execution across three seeds, yet intervention audits still detect dependence on semantic schema wording. The latest completed result is therefore classified as a **residual schema shortcut**, not zero-shot semantic invariance.

The final conclusion is narrower than “Skill IR solves tool learning”:

> Skill IR is persistent and execution-sufficient when the support set exposes the relations, effects, and lexical identifiers required by the resolver. Learning those fields robustly from new schema language remains open.

## Repository map

- `src/tool_lora/skill_ir/`: controlled Skill IR acquisition, canonicalization, persistence, and resolution.
- `src/tool_lora/stage4_p0/`: generic policy representation and learned correspondence.
- `src/tool_lora/stage4_b0/`, `stage4_b1/`, `stage4_b2/`: AppWorld resolver, deterministic acquisition, and transfer.
- `src/tool_lora/stage4_c0_r1/`, `stage4_c1/`, `stage4_c1_a0/`: unseen-tool identifiability and effect-anchored Skill IR.
- `benchmarks/`: executable experiments from the controlled benchmark through the latest schema-channel audits.
- `scripts/adapter_diagnostics/`: Christian's adapter-identifiability, free-latent, regularization, structured-generation, and diversity experiments.
- `results/`: frozen checkpoints, immutable fixtures, manifests, hashes, and recorded metrics.
- `docs/complete_project_history.md`: chronological research synthesis and claim boundary.
- `docs/AUTHORSHIP.md`: provenance and exclusion policy.
- `report/main.tex`: two-page course report covering the complete project.

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

Some AppWorld benchmarks require the external AppWorld package and its task data. The preserved manifests record the AppWorld version, commit, immutable input hashes, and whether execution was performed.

## Claim boundary

The repository supports three claims:

1. independently optimized LoRA parameters are unsuitable as unique semantic labels in this setting;
2. a structured Skill IR can survive support-set deletion and preserve executable information in controlled tasks;
3. effect-aware Skill IR can reconnect to evaluated AppWorld tools when public evidence uniquely identifies the mapping.

It does not claim general ontology discovery, unrestricted zero-shot tool learning, or robust invariance to unseen schema language.

## License

Code in this standalone repository is released under the MIT License. Qwen, AppWorld, external datasets, and the inherited `tool-lora-hypernet` implementation are not redistributed.
