# Persistent Skill IR for Tool-Using Language Models

Course project by **Christian Bianchi** for **Deep Learning and Applied AI, Sapienza University of Rome (2026)**.

## Research question

A **tool-use skill** is the behavior that maps a user request and prior observations to a structured tool call. A **support set** is the collection of demonstrations from which the system learns that behavior.

This project asks:

> Can a support set be converted into a persistent intermediate representation of a tool-use skill that remains sufficient for execution after the support set is deleted, without giving the learner the evaluator's semantic labels?

We call the persistent object **Skill IR**, where *IR* means *intermediate representation*. Throughout this repository, **Skill IR** refers only to the serialized representation learned from demonstrations. We use **execution adapter** only for the temporary LoRA parameters applied during language-model execution. These terms are not interchangeable.

## Why Skill IR is separate from LoRA parameters

Low-rank adaptation (LoRA) modifies a frozen model weight matrix through

```text
updated weight = frozen weight + B @ A
```

Here, `A` and `B` are low-rank parameter matrices and `B @ A` is the effective weight update. Different pairs of matrices can implement the same effective update. Independently trained adapters can also implement similar behavior while remaining far apart in parameter space. Consequently, an execution adapter is not a stable semantic identifier for a tool-use skill.

The project therefore assigns different roles to the two objects:

- **Skill IR** is persistent. It stores what was learned about the tool-use skill.
- **Execution adapter** is temporary. A frozen compiler produces it when the skill is executed.

## Maintained pipeline

```text
support set
  -> relation acquisition
  -> component correspondence
  -> canonicalization
  -> Skill IR
  -> support-set deletion and Skill-IR reload
  -> resolver
  -> frozen compiler
  -> execution adapter
  -> frozen language model
```

Each stage has one role:

1. **Relation acquisition** extracts observable relations between request fields and tool-call fields. The relation types are exact copy, enumerated binding, and structured transformation.
2. **Component correspondence** identifies which input component supplies each output component.
3. **Canonicalization** combines correspondence evidence from multiple demonstrations into one deterministic Skill IR. It abstains when the best hypotheses remain tied.
4. **Skill IR** stores canonical relational records and a lexical sidecar. The **lexical sidecar** stores opaque strings, such as API names and literal identifiers, that cannot be reconstructed from relational structure alone. Skill IR does not store the support set.
5. **Resolver** deterministically maps Skill IR into the semantic fields expected by the frozen compiler.
6. **Frozen compiler** maps the resolved fields into an execution adapter. *Frozen* means that its parameters are not trained or modified in this experiment.
7. **Frozen language model** executes the request while the execution adapter is temporarily active.

## Complete research synthesis

### Stage 1: adapter generation

The initial objective was to generate LoRA parameters from tool documentation and demonstrations. The evaluated methods included direct parameter regression, principal-component bases, canonicalized effective updates, nearest-neighbor residuals, structured adapter tokens, and WIZARD-style decoders.

The strongest generated execution adapter achieved **79.36% correct-tool accuracy**, where correct-tool accuracy is the fraction of generated calls that name the required tool. **Nearest-neighbor adapter retrieval**, which reuses the adapter of the most similar training tool, achieved **94.31%**. An independently trained adapter for each evaluated tool achieved **99.34%**. Therefore, generated execution adapters did not provide a competitive persistent representation.

### Stage 2: persistent Skill IR

The second stage replaced adapter parameters with Skill IR. Skill IR contains:

- the output field selected by the tool-use skill;
- generic request-to-output relations;
- component permutations for structured values;
- a lexical sidecar for opaque names and literals.

Serialization tests delete the support set, reload Skill IR from JSON, and execute new requests. Passing this test means that execution does not depend on hidden access to the original demonstrations.

### Stage 3: learned correspondence

**P6-R1** is the frozen learned correspondence model used in the final evaluation. It contains 13,281 parameters; *frozen* means that these parameters are not updated during the reported evaluation. P6-R1 encodes source and target strings with a bidirectional gated recurrent unit and produces a compatibility score for every source–target component pair. A bijective assignment then selects one permutation.

The repository also includes the **exact matcher**. This deterministic method enumerates all permutations and selects the unique permutation that reconstructs the observed target. It applies only when every source component remains directly observable in the target.

For both matchers, the stored permutation follows one convention:

```text
target[j] = source[permutation[j]]
```

`j` is an output position, and `permutation[j]` is the input position copied into it. Tests cover all six permutations of three components.

### Stage 4: canonical persistence and execution

The final experiment evaluates 24 skill groups. Each group has three conditions:

- **matched:** demonstrations and evaluation use the same skill configuration;
- **alternate:** demonstrations express the same skill through different examples;
- **counterfactual:** the underlying component transformation is deliberately changed.

The three conditions produce 72 evaluation cases. The reported metrics are:

- **Any correct:** the predicted set contains the ground-truth permutation.
- **Unique correct:** the ground-truth permutation is the only member of the predicted set.
- **Reload:** Skill IR is unchanged after serialization and deserialization.
- **Time/case:** mean CPU evaluation time per case in the recorded run.

| Correspondence method | Any correct | Unique correct | Reload | Time/case |
|---|---:|---:|---:|---:|
| P6-R1 | 100% | 95.83% | 100% | 4.12 ms |
| Exact matcher | 100% | 100% | 100% | 0.45 ms |
| Deterministic acquisition control | 100% | 100% | 100% | 0.34 ms |

P6-R1 retains a correct hypothesis in every case but leaves three cases ambiguous. The exact matcher resolves every case uniquely, but its observability assumption is stronger. All methods preserve Skill IR exactly across reload.

## Conclusions

The experiment establishes the following result:

> In the controlled protocol domain, the support set contains enough information to construct a canonical Skill IR that preserves relational structure and opaque lexical identity, survives support-set deletion, and reconnects to the frozen execution pipeline.

This is an **information-sufficiency result**: Skill IR contains enough information for the tested execution task. It is not an ontology-discovery result. In this repository, **semantic ontology** means the evaluator's predefined decomposition of a skill into named semantic factors. The learner was not given those names, and the experiments did not show that it independently recovered the same decomposition.

## Negative results and limitations

- Generated execution adapters did not outperform nearest-neighbor adapter retrieval.
- Learned global mappings from generic records to compiler fields did not reliably replace the deterministic resolver.
- Stable statistical clusters did not yield independently controllable semantic factors.
- The exact matcher requires visible source components.
- The positive result is limited to a controlled synthetic protocol domain.

These limitations are part of the experimental conclusion. Statistical regularity, semantic information, deterministic executability, and compositional structure are different properties; success on one does not imply success on the others.

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

The tests verify the frozen checkpoint and evaluation-suite hashes, support-set deletion, Skill-IR reload, the resolver convention, and the Stage 1/2 invariants. The benchmark reproduces `results/component_replacement_evaluation.json`.

## Repository structure

- `src/tool_lora/skill_ir/`: maintained Skill-IR implementation.
- `src/tool_lora/functional_hypernet/functional_lora.py`: temporary execution-adapter application.
- `results/stage3d_p6_r1/`: frozen P6-R1 checkpoint, immutable evaluation suite, and provenance metadata.
- `benchmarks/`: component evaluation.
- `tests/`: maintained-path tests.
- `docs/`: architecture, research history, and reproducibility notes.
- `report/`: DLAI LaTeX report and the course template files.

The concise scientific report is available in [report/main.tex](report/main.tex). The detailed claim boundary is documented in [docs/stage3_research_synthesis.md](docs/stage3_research_synthesis.md).

## License

The project code is released under the MIT License. The Qwen model and external datasets are not redistributed.
