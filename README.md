# Persistent Skill Representations for Language-Model Tool Use

Course project by **Christian Bianchi** for **Deep Learning and Applied AI, Sapienza University of Rome (2026)**.

This repository contains the complete experimental project on persistent representations for language-model tool use. The project progresses from generated-adapter diagnostics to a structured persistent representation, **Skill IR**, and finally to transfer experiments on public tool schemas derived from AppWorld.

The central goal is not only to execute a tool correctly, but to determine **what representation of a learned tool-use skill should persist once the demonstrations used to acquire it are removed**.

## Research question

A **tool-use skill** maps a user request and observable environment state to a structured tool call. A **support set** is the set of demonstrations available while the skill is acquired.

The project asks:

> Can a support set be converted into a persistent representation of a tool-use skill that remains sufficient for execution after the support set is deleted?

We evaluate two possible representations:

- An **execution adapter** is a temporary low-rank adaptation (LoRA) applied to a language model during execution.
- **Skill IR** is a serialized intermediate representation containing relations and opaque lexical identifiers acquired from the support set.

When these fields are insufficient to identify a concrete API, Skill IR is extended with an **effect anchor** describing whether an operation preserves or changes environment state.

The distinction is central:

> **Skill IR is the persistent object; LoRA parameters are temporary execution state.**

The project therefore lies primarily at the intersection of **representation learning** and **language-model tool use**.

---

## Project progression

### Phase 1 — Adapter parameters as the persistent representation

The initial approach investigated whether generated LoRA parameters could themselves represent a tool-use skill.

Using the inherited 15-tool validation setup, the relevant comparison is:

| Execution strategy | Correct-tool accuracy |
|---|---:|
| Generated execution adapter | **79.36%** |
| Nearest-neighbor adapter retrieval | **94.31%** |
| Per-tool trained adapter | **99.34%** |

**Correct-tool accuracy** measures whether the generated call names the required tool.

The 14.95-point gap between generation and retrieval motivated a sequence of diagnostic experiments.

Christian's subsequent experiments investigated whether the failure could be explained by semantic conditioning, optimization, or the structure of the generated parameter target:

1. **Functional-delta supervision** added objectives on the effective LoRA update and the language model's output distribution.
2. **Adapter-identifiability audit** independently trained multiple adapters for the same tool. The adapters preserved tool identity while their effective updates remained weakly aligned, showing that independently optimized adapters do not provide a unique numerical target for the skill.
3. **Free-latent shared generator** removed semantic conditioning by learning one adapter-generating latent per training tool. It fit training behavior but did not recover the validation behavior of independently trained adapters.
4. **Regularization and distillation controls** tested output-norm penalties and functional teacher distillation.
5. **Structured generation controls** tested explicit singular-value generation and a fixed first LoRA factor while generating only the second factor.
6. **Diversity controls** separated surface variation from genuinely different functional tasks.

These experiments did not establish generated LoRA parameters as a stable persistent representation.

The resulting conclusion is deliberately limited:

> Independently optimized LoRA parameters are not reliable unique semantic targets for a tool-use skill in the tested setting.

The corresponding experiments are retained under `scripts/adapter_diagnostics/`.

---

### Phase 2 — Controlled persistent Skill IR

The second approach separates the persistent skill representation from the parameters used during execution.

The base Skill IR is

\[
\mathcal{S}_0 = (\mathcal{R}, \mathcal{L}),
\]

where:

- **relation records** \(\mathcal{R}\) specify how request values determine tool-call values, including direct copies, bindings, and reorderings;
- the **lexical sidecar** \(\mathcal{L}\) stores opaque API names and literal values that cannot be reconstructed from relational structure alone.

Evidence from multiple demonstrations is combined through **canonicalization** into one deterministic representation.

The controlled pipeline is:

```text
support set
  -> relation acquisition
  -> component correspondence
  -> canonicalization
  -> Skill IR
  -> support-set deletion
  -> Skill-IR serialization/reload
  -> resolver
  -> frozen compiler
  -> temporary LoRA execution adapter
  -> frozen language model
````

The support set is deleted before reload and execution. Subsequent execution therefore cannot recover information directly from the original demonstrations.

### Learned component correspondence

**Component correspondence** identifies which source component supplies each target component.

The frozen **learned correspondence model** contains **13,281 parameters**. It:

1. encodes source and target components with byte-level bidirectional GRUs;
2. scores every source-target component pair;
3. predicts a one-to-one assignment between component positions.

Training examples contain balanced component permutations rather than semantic class labels.

Two deterministic controls are also evaluated:

* the **exact matcher**, which enumerates possible permutations and returns the unique correspondence consistent with the observations;
* **rule-based acquisition**, which directly extracts relations when correspondence is explicit.

The exact matcher applies only when every relevant transformed source component remains observable in the target.

### Controlled benchmark

The benchmark contains **24 skill groups** evaluated under three conditions:

* **Matched:** the acquisition configuration is preserved.
* **Alternate:** demonstrations change while the underlying skill is preserved.
* **Counterfactual:** the component transformation itself changes.

This yields **72 evaluation cases**.

| Correspondence method  | Any correct | Unique correct |   Reload |   Time/case |
| ---------------------- | ----------: | -------------: | -------: | ----------: |
| Learned correspondence |    **100%** |     **95.83%** | **100%** | **4.12 ms** |
| Exact matcher          |    **100%** |       **100%** | **100%** | **0.45 ms** |
| Rule-based acquisition |    **100%** |       **100%** | **100%** | **0.34 ms** |

**Any correct** means that the prediction set contains the benchmark-defined permutation.

**Unique correct** means that the benchmark-defined permutation is the only predicted solution.

**Reload** verifies that serialization and deserialization preserve Skill IR exactly.

The learned model retains the correct correspondence in all 72 cases and identifies it uniquely in 69 of 72.

The deterministic methods solve all 72 cases uniquely. This is not evidence of superiority of the learned model: rather, it shows that the controlled benchmark is sufficiently observable to admit an exact symbolic solution.

The role of this benchmark is therefore to establish:

* persistence after support-set deletion;
* information sufficiency of the representation;
* correct serialization and reload;
* reconnection from the persisted representation to execution.

It does **not** establish discovery of an evaluator-defined hidden semantic decomposition.

---

### Phase 3 — Transfer to AppWorld-derived tools

The final phase tests whether the representation and acquisition machinery transfers from controlled synthetic transformations to public schemas derived from **AppWorld**.

#### Todoist transfer

An **oracle Skill IR** supplies the correct representation fields and therefore isolates representation and execution from acquisition errors.

On the evaluated Todoist fixtures:

* oracle Skill IR achieves **100% matched and counterfactual execution**;
* rule-based acquisition achieves **100% matched and counterfactual execution**;
* learned correspondence satisfies the predefined transfer controls across three predeclared random seeds.

These results establish that persisted Skill IR can reconnect to the evaluated tool interfaces when the representation contains enough information.

#### Simple Note identifiability failure

Previously excluded Simple Note tools expose a qualitatively different failure.

Two candidate APIs are structurally compatible with the same base representation

$$
\mathcal{S}_0 = (\mathcal{R}, \mathcal{L}).
$$

The generic resolver therefore reaches **0% mapping accuracy despite exact Skill IR serialization**.

This is not a persistence failure and not an optimization failure.

The representation simply does not contain enough information to distinguish the two candidate APIs.

We call this **resolver-interface non-identifiability**:

> Multiple concrete API mappings are compatible with the information available to the resolver.

#### Effect-aware Skill IR

To resolve this ambiguity, Skill IR is extended to

$$
\mathcal{S}_e = (\mathcal{R}, \mathcal{L}, e),
$$

where the **effect anchor**

$$
e \in \{\text{preserve}, \text{change}\}
$$

records whether executing an operation preserves or changes environment state.

The competing Simple Note operations differ in this observable property.

Adding the effect anchor therefore makes the evaluated mapping identifiable and restores **100% matched and counterfactual execution** in the evaluated fixtures.

---

### Learned effect acquisition

The final experiments investigate whether the effect anchor can itself be learned rather than provided directly from the observed state transition.

The **effect predictor** receives:

* the user request;
* the public tool schema;
* the executed call;
* the returned result;
* the observed state transition.

The initial predictor reaches **100% accuracy on its synthetic training data** but fails on held-out concrete schemas.

This shows that training accuracy alone does not establish acquisition of the intended behavioral concept.

A **deconfounded training set** is therefore constructed so that schema wording varies independently of the effect label.

Training on this data restores **100% matched execution across seeds 17, 29, and 43**.

However, a subsequent intervention audit changes semantic schema descriptions while keeping the underlying operation fixed. Some predictions change under this intervention.

The final learned result therefore still exhibits a **residual schema shortcut**:

> Effect prediction remains partially dependent on semantic schema wording rather than exclusively on observed behavioral evidence.

The project does **not** claim zero-shot invariance to unseen schema language.

A stronger zero-shot schema-language invariance experiment remains future work.

---

## Main conclusion

The complete experimental progression supports a conditional answer to the original research question.

> A support set can be replaced by a persistent representation when that representation captures the information required to identify the skill's concrete execution.

The experiments distinguish two different barriers:

1. **Parameter-target ambiguity.**
   Independently optimized LoRA parameters do not provide reliable unique numerical targets for a tool-use skill.

2. **Representation insufficiency.**
   Even perfectly persisted Skill IR can fail when the representation omits information required to distinguish concrete APIs.

Structured Skill IR solves the persistence problem in the controlled setting. Adding observable state-transition effects resolves the evaluated AppWorld ambiguity.

The remaining bottleneck is therefore **robust acquisition rather than persistence**.

---

## Repository map

* `src/tool_lora/skill_ir/`
  Controlled Skill IR acquisition, canonicalization, persistence, and resolution.

* `src/tool_lora/stage4_p0/`
  Generic policy representation and learned component correspondence.

* `src/tool_lora/stage4_b0/`, `stage4_b1/`, `stage4_b2/`
  AppWorld resolver, deterministic acquisition, and transfer experiments.

* `src/tool_lora/stage4_c0_r1/`, `stage4_c1/`, `stage4_c1_a0/`
  Unseen-tool identifiability and effect-aware Skill IR experiments.

* `benchmarks/`
  Executable benchmarks from the controlled correspondence evaluation through the latest schema-channel interventions.

* `scripts/adapter_diagnostics/`
  Adapter-identifiability, free-latent, regularization, structured-generation, and diversity experiments.

* `results/`
  Frozen checkpoints, immutable fixtures, manifests, hashes, and recorded metrics.

* `docs/complete_project_history.md`
  Chronological research history and experimental synthesis.

* `docs/AUTHORSHIP.md`
  Detailed provenance and exclusion policy.

* `report/main.tex`
  Source of the final DLAI course report.

---

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

Some AppWorld benchmarks additionally require the external AppWorld package and its task data.

Preserved experiment manifests record the relevant AppWorld version, commit, immutable input hashes, and whether external execution was performed.

---

## Scope and claim boundaries

The repository supports three principal claims:

1. **Independently optimized LoRA parameters are unsuitable as unique semantic skill targets in the tested setting.**

2. **Structured Skill IR can preserve execution-relevant information after support-set deletion in the controlled benchmark.**

3. **Effect-aware Skill IR can reconnect to the evaluated AppWorld tools when observable public evidence uniquely determines the concrete mapping.**

The project does **not** establish:

* unrestricted ontology discovery;
* general zero-shot tool learning;
* robust invariance to unseen schema language.

In particular, reproducible statistical structure should not automatically be interpreted as an independently controllable or compositional semantic factor.

---

## Provenance

This project was developed after branching from the original `tool-lora-hypernet` codebase.

Code authored by **Luca Romani** is not reproduced as project code in this standalone repository. Values originating from the inherited evaluation framework are explicitly identified as comparison baselines.

The experiments, diagnostics, Skill IR development, transfer analyses, and claim-boundary investigations presented as this course project were developed by **Christian Bianchi**.

See `docs/AUTHORSHIP.md` for the detailed provenance and exclusion policy.

---

## License

Code in this standalone repository is released under the **MIT License**.

Qwen, AppWorld, external datasets, and the inherited `tool-lora-hypernet` implementation are not redistributed.

```
