# Persistent Skill IR for Tool-Using Language Models

Course project by **Christian Bianchi** for **Deep Learning and Applied AI, Sapienza University of Rome (2026)**.

This repository studies whether demonstrations can be converted into a compact, persistent skill representation that remains executable after the demonstrations are removed. The maintained system stores generic relational structure together with a lexical sidecar for opaque identifiers. It then resolves the saved artifact into a frozen execution interface that can produce ephemeral LoRA updates for a frozen language model.

```text
demonstrations
  -> ontology-free acquisition
  -> observable or learned correspondence
  -> canonical persistent Skill IR
  -> support removal and reload
  -> structured resolver
  -> frozen compiler / executor
```

The central positive result is deliberately narrow: on the controlled protocol task, the persisted representation retains the information needed for execution without storing support text. The project does **not** claim to discover an arbitrary latent ontology or independently compositional semantic modules.

## Results

The immutable benchmark has 24 groups and three conditions per group (72 cases).

| Correspondence path | Any correct | Singleton correct | Reload | Time/case |
|---|---:|---:|---:|---:|
| Learned P6-R1 matcher | 100% | 95.83% | 100% | 4.12 ms |
| Exact observable permutation | 100% | 100% | 100% | 0.45 ms |
| Deterministic acquisition | 100% | 100% | 100% | 0.34 ms |

The exact path is appropriate only when structured components remain observable. P6-R1 is retained as the learned control for opaque correspondence.

An accompanying adapter-generation study found that direct generation remained substantially below retrieval and independently trained adapters on correct-tool accuracy:

| Adapter method | Correct tool |
|---|---:|
| Generated residual LoRA | 79.36% |
| Nearest-neighbor LoRA | 94.31% |
| Oracle per-tool LoRA | 99.34% |

This negative result motivated the separation between persistent semantic state (Skill IR) and ephemeral neural execution state (LoRA weights).

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
