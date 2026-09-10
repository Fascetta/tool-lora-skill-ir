# Complete project history

This document records the complete scientific path of the project. A **gate** is an experiment whose result determines whether the next stage is scientifically justified.

## 1. Functional adapter supervision

The project began by asking whether tool descriptions could generate LoRA adapters. A LoRA update is `B @ A`, where `A` and `B` are low-rank matrices. Christian added function-space objectives that compare effective updates and output distributions rather than relying only on raw factor reconstruction. The resulting generator remained below adapter retrieval on held-out tools.

## 2. Adapter identifiability

Multiple independently trained LoRAs for the same tool preserved tool identity while yielding poorly aligned effective updates. This rejected the assumption that one independently trained adapter is the unique numerical label for a skill.

## 3. Shared generated-adapter manifold

A free-latent generator assigns one learned vector to each known tool and jointly generates its LoRA. Removing semantic conditioning did not close the generalization gap. Single-tool controls showed that the generator can fit training behavior, directing attention from representational capacity toward optimization bias and validation generalization.

Output-norm regularization, functional distillation, explicit singular-value generation, and fixed-first-factor generation were tested. They changed adapter geometry but did not reliably reproduce direct-LoRA generalization. Surface and functional diversity controls then tested whether a narrow task distribution caused the shortcut. The branch did not establish a stable generated-adapter substrate and was not connected to Skill IR.

## 4. Controlled Skill IR

The project replaced adapter parameters with a serialized representation. The representation separates relations from opaque lexical values, canonicalizes evidence across demonstrations, and survives deletion of the support set.

The final controlled matcher, P6-R1, retains the benchmark permutation in all 72 cases and identifies it uniquely in 69. An exact observable matcher and a rule-based acquisition control identify all 72 uniquely. These controls show both that persistence works and that the synthetic benchmark exposes enough structure for symbolic correspondence.

## 5. Generic policy Skill IR

Stage 4 introduced a generic policy representation that stores action branches and argument bindings without evaluator policy identifiers. Oracle and learned acquisition were tested under matched, alternate, paraphrased, counterfactual, relation-shuffled, lexical-shuffled, and constant controls. After correcting the supervision coordinate convention, three learned-correspondence seeds satisfied the predefined thresholds.

## 6. AppWorld transfer

The next experiments mapped the generic representation to public AppWorld schemas. Oracle reconnection and deterministic acquisition passed on Todoist fixtures. Learned transfer preserved canonical equality and causal control behavior.

An unseen Simple Note evaluation then exposed a genuine boundary: two tools had compatible public schemas, so an abstract role could not be mapped uniquely. A resolver cannot infer information absent from its input.

## 7. Effect-anchored Skill IR

The representation was extended with an effect anchor describing whether an operation is state-preserving or state-changing. This additional observable property uniquely identifies the evaluated mappings and restores execution. Lexical-binding, distractor, candidate-order, and intervention controls passed.

An evaluator-coordinate inconsistency initially produced false acquisition failures. A joint-assignment audit identified the defect; replay with corrected coordinates obtained 100% effect-anchor recovery, full semantic recovery, intervention equivariance, and relation-effect assignment over 60 cases.

## 8. Learned effect acquisition and latest result

A learned effect predictor reached 100% training accuracy but failed on held-out concrete schemas. Deconfounded training restored perfect matched execution across seeds 17, 29, and 43. However, the final in-distribution channel audit showed that opaque identity alone was inert while semantic schema content remained influential. The latest completed classification is therefore **residual schema shortcut**.

The unexecuted `stage4_c3_zero_shot_schema_language_invariance.py` benchmark is retained as the next planned test; no result is claimed for it.

## Final conclusion

The project establishes that persistent execution is possible when Skill IR contains identifiable relations, effects, and lexical values. It also identifies two failure modes: adapter-space non-identifiability and semantic-field acquisition shortcuts. Robust zero-shot acquisition from unseen schema language remains unresolved.
