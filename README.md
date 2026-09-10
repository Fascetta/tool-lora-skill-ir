# Persistent Skill IR for Tool-Using Language Models

Course project by **Christian Bianchi** for **Deep Learning and Applied AI, Sapienza University of Rome (2026)**.

## What this project is about

This repository contains one research project: learning persistent representations of tool-use skills for language models.

The motivating goal was to generate task-specific LoRA adapters directly from a description or demonstrations of a tool. The experiments showed that independently trained LoRAs are a poor semantic target: adapters that implement similar behavior can occupy very different regions of parameter space, and learned generation remained substantially worse than simply retrieving a nearby adapter.

The project therefore separates two objects:

- **Skill IR** is the persistent representation. It stores the relational structure and opaque lexical information acquired from demonstrations.
- **LoRA weights** are ephemeral execution state. They may be produced by a frozen compiler when the skill is executed, but they are not treated as the semantic representation itself.

The final research question is:

> Can learner-visible demonstrations be converted into a canonical, persistent Skill IR that remains sufficient for execution after the demonstrations are removed, without supplying the evaluator's semantic ontology?

## Maintained system

The final validated pipeline is:

```text
demonstrations
  -> ontology-free acquisition
  -> observable or learned correspondence
  -> canonical persistent Skill IR
  -> support removal and reload
  -> structured resolver
  -> frozen compiler / executor
```

Acquisition extracts generic relations from observable requests and responses. Correspondence aligns structured components, canonicalization combines evidence across demonstrations, and serialization stores relational records plus a lexical sidecar. The original demonstrations are then deleted. A structured resolver reloads the artifact and maps it into the independently validated frozen execution interface.

## Complete research synthesis

### 1. Direct adapter generation

The initial approach attempted to predict LoRA parameters from tool documentation and demonstrations. Full-vector regression, PCA/basis prediction, canonicalized updates, residual generation, structured token prediction, and corrected WIZARD-style decoders were tested. Decoder plumbing and serialization could be made correct, but generated adapters did not generalize reliably to unseen tools.

The strongest learned generator obtained **79.36% correct-tool accuracy**, compared with **94.31%** for nearest-neighbor adapter reuse and **99.34%** for an independently trained oracle LoRA. This established that the main problem was not simply finding a larger decoder or a different weight-space loss.

### 2. Persistent Skill IR

The second direction represented the skill explicitly instead of using adapter weights as memory. The resulting hybrid representation contains:

- generic relations such as exact copy, enumerated binding, and structured transformation;
- correspondence hypotheses between observable components;
- a lexical sidecar that preserves opaque API names, argument names, and literal values;
- no support text and no evaluator ontology labels.

The artifact can be serialized, the demonstrations can be removed, and the artifact can then be reloaded and executed.

### 3. Correspondence and canonicalization

P6-R1 is a frozen 13,281-parameter matcher. It embeds byte-level source and target components with a bidirectional GRU, scores a compatibility matrix, and selects a bijective assignment. A deterministic exact matcher is also included for cases where component occurrences are directly observable.

Evidence from multiple demonstrations is canonicalized by count, assignment score, and confidence margin. The resolver uses the persisted convention `target[j] = source[permutation[j]]`, which is tested over all six three-component permutations.

### 4. Validated result

The immutable evaluation contains 24 skill groups under matched, alternate, and counterfactual conditions, giving 72 cases.

| Correspondence path | Any correct | Singleton correct | Reload | Time/case |
|---|---:|---:|---:|---:|
| Learned P6-R1 matcher | 100% | 95.83% | 100% | 4.12 ms |
| Exact observable permutation | 100% | 100% | 100% | 0.45 ms |
| Deterministic acquisition | 100% | 100% | 100% | 0.34 ms |

All three paths retain at least one correct hypothesis in every case and survive serialization/reload. P6-R1 produces a unique correct hypothesis in 95.83% of cases. The exact observable matcher resolves all cases uniquely and is faster, but it applies only when source components remain visible.

The adapter-generation comparison that motivated Skill IR is:

| Adapter method | Correct tool |
|---|---:|
| Generated residual LoRA | 79.36% |
| Nearest-neighbor LoRA | 94.31% |
| Oracle per-tool LoRA | 99.34% |

### 5. What the project establishes

On the controlled protocol domain, demonstrations contain enough information to construct a canonical, persistent hybrid Skill IR. The representation preserves relational semantics and lexical identity, survives support removal, and reconnects to a frozen executor.

This is an **information-sufficiency and persistence result**. It is not evidence that the system discovers an arbitrary hidden ontology.

### 6. Negative findings and claim boundary

Several negative results are part of the contribution:

- unrestricted learned correspondence models exploited shortcuts;
- global learned bridges failed to translate generic records into executor semantics reliably;
- sparse factorization and anti-collapse objectives did not discover stable semantic modules;
- statistically stable groups were not independently manipulable or compositional;
- generated LoRAs did not beat nearest-neighbor adapter reuse.

These findings distinguish semantic availability and executor sufficiency from causal modularity and ontology discovery. The maintained system consequently uses a structured resolver and treats generated adapters as temporary execution artifacts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## Reproduce the maintained evaluation

```bash
pytest
python benchmarks/evaluate_component_replacements.py
```

The tests verify the frozen checkpoint and suite hashes, persistence after support removal, the resolver's permutation convention, and Stage 1/2 invariants. The benchmark reproduces the component comparison in `results/component_replacement_evaluation.json`.

## Repository layout

- `src/tool_lora/skill_ir/`: acquisition, correspondence, canonicalization, persistence, resolution, compilation, and execution.
- `results/stage3d_p6_r1/`: frozen P6-R1 checkpoint, immutable suite, and provenance manifest.
- `benchmarks/`: reproducible component evaluation.
- `tests/`: focused maintained-path tests.
- `docs/`: architecture, scientific scope, experiment history, and reproducibility notes.
- `report/`: the two-page DLAI course report and official template files.

## Limitations

- The positive evaluation uses a controlled synthetic protocol domain.
- The exact matcher relies on directly observable structured components.
- Statistical organization did not imply causal modularity or discovered composition.
- The generated-LoRA experiments did not beat nearest-neighbor adapter reuse.

See [the research synthesis](docs/stage3_research_synthesis.md) for the complete claim boundary and [the LaTeX report](report/main.tex) for the course submission.

## License

MIT. The Qwen model and external datasets are not redistributed by this repository.
