# Experiment history

This ledger records the scientific progression. Executable implementations of rejected branches are intentionally excluded from this standalone maintained release. Metrics below are compact values extracted from preserved result manifests and audit reports.

## Stage 1 — semantic acquisition

The initial control validated semantic acquisition and the explicit executor-facing representation on the controlled protocol task. The question was whether demonstrations could supply the semantic values required by the existing execution boundary. Explicit protocol IR, compiler preflight, and executor controls passed. This established semantic availability and executor sufficiency, not ontology discovery. It was retained as the predecessor to the persistent IR work.

## Stage 2 — persistent IR, compression, composition, compiler

| Version | Question / change | Controls and result | Classification / next step |
|---|---|---|---|
| Explicit persistent IR | Can semantic values be represented independently of the prompt? | Round-trip and schema invariants passed. | Retained as the validated compiler interface. |
| Compression and hybrid IR | Can relational codes reduce storage while retaining opaque lexical identity? | Capacity, collision, quantization, and hybrid sidecar controls were run. | Hybrid structure plus lexical sidecar retained. |
| Composition | Can independently represented skills compose? | Pairwise and compiler tests separated composition behavior from storage. | Retained only as tested compiler behavior; not evidence of discovered modules. |
| Independent compiler tests | Can the executor reconnect be validated independently? | Frozen V6-G compiler and executor tests passed. | Retained as the final execution boundary. |

## Stage 3A/3B — transformation and correspondence progression

| Version | Research question and implementation idea | Key controls / finding | Decision and successor |
|---|---|---|---|
| Deterministic transformation proof of concept | Can observable structured values be converted into generic relations? | Direct extraction worked on controlled examples. | Retained as the generic baseline; followed by learned matching. |
| Byte-GRU matcher | Can a byte-level model infer component correspondence without semantic names? | Unrestricted correspondence learned shortcuts and failed out-of-distribution structure. | Rejected as the maintained matcher; correspondence bias followed. |
| Streaming-diversity diagnostics | Does a diverse stream fix the shortcut? | Streaming diversity alone did not solve learning. | Rejected; balanced permutation training followed. |
| Balanced permutation training / seed replication | Does explicit balance produce stable correspondence? | Balanced controls and three seeds improved the generic alignment; same-seed reruns exposed artifact/provenance concerns. | Retained as the route to P6, with checkpoint freezing. |
| Pairwise aggregation | Does aggregating multiple pairwise evidence items stabilize a transformation? | Aggregation improved consensus but did not eliminate ambiguity. | Followed by multi-example pooling. |
| Multi-example pooling | Does support pooling improve transformation recovery? | Same-skill alternate demonstrations were compared with matched and counterfactual controls. | Followed by correspondence-biased matcher P6. |
| P6 correspondence bias | Does a shared byte component encoder plus 3×3 compatibility and bijective assignment learn generic relations? | 1,200-step P6-R1 training; frozen checkpoint hash above; generic alignment and reload were reproducible. | Retained as the final learned correspondence matcher. |
| X5 lexical sidecar | Can opaque literals survive while relational structure remains unchanged? | Relational equality and literal intervention locality were preserved; sidecar was executor-sufficient. | Retained in persistent hybrid IR. |
| Learned bridges | Can a global learned Stage-3→Stage-2 bridge reconnect the artifact? | Aggregate-CE, semantic-selective, semantic-supervised, and dense/global variants failed causal semantic translation despite a successful structured resolver. | Removed; structured resolver retained. |
| Structured resolver | Can generic persisted records be mapped without ontology labels? | Deterministic relation lookup, canonical record ordering, literal pointers, and transformation conversion passed support-removal and executor audits. | Retained as the only reconnect path. |
| Provenance audit | Are artifacts and conclusions reproducible? | P6/RSL5 learned path and X5 deterministic path were identified as distinct; checkpoint and suite were frozen. | Followed by P6-R1 and permutation correction. |
| P6-R1 | Can one preserved learned reference support downstream audits? | Checkpoint and immutable suite are preserved with hashes; canonicalization selected conservative rule D. | Final reference. |
| Permutation correction | Which index convention is persisted? | Regression fixed `target[j] = source[permutation[j]]`; all six 3-component permutations are tested. | Retained in resolver and documentation. |
| CA1 generic canonicalization | Can multi-hypothesis evidence be canonicalized without evaluator correctness? | Learner-visible development selected strict consensus/tie abstention; matched, alternate, and counterfactual canonical correctness were 1.0 in the final audit. | Retained as canonicalization. |

## Stage 3C — negative result and identifiability boundary

Stage 3C ended at RID0. These executable experiments were research diagnostics, not production functionality.

| Experiment | Question / controls | Finding | Classification / disposition |
|---|---|---|---|
| F1 dense factorizer | Does generic atom reconstruction reveal stable latent slots? Typed, fixed-hash, random, shuffled, and monolithic controls. | Stable representation but no intervention locality. | `F1_no_stable_partition`; removed. |
| F3 sparse competition | Does sparse slot competition discover factors? Dense F1 control and utilization/entropy checks. | Sparse slot assignment collapsed. | `F3`; removed. |
| F10 anti-dead-slot factorizer | Does occupancy pressure prevent collapse? Dead-slot and occupancy controls. | Pressure created artificial occupancy, not semantic modules. | `F10`; removed. |
| CV0 co-variation audit | Is the original distribution sufficient for co-variation? Marginal-preserving controls. | Original distribution exposed no trustworthy co-variation signal. | `CV0`; removed. |
| IV1 intervention-rich audit | Does an intervention-rich population expose co-variation? Target-unspecified intervention controls. | Non-null co-variation geometry appeared. | `IV1`; diagnostic only, removed. |
| CG2 co-variation factorizer | Can a small learner internalize that geometry? 85-parameter learner and monolithic/null controls. | Co-change partition was learned, but only partially aligned with evaluator factors. | `CG2`; removed. |
| PS4 payload separability | Are payload changes separable from relational changes? Payload permutation and identity controls. | Payload separability did not establish causal modularity. | `PS4`; removed. |
| PN0 payload normalization | Does normalizing payloads reveal cleaner factors? Raw/normalized population controls. | Normalization did not create identifiable modules. | `PN0`; removed. |
| EI2 partition stability | Are discovered groups stable across partitions? Train/held-out and null partitions. | Statistical groups were stable but not necessarily compositional. | `EI2`; removed. |
| SH1 soft/hierarchical | Does soft relational organization provide useful hierarchy? Soft similarity and hard-boundary controls. | Continuous geometry was reproducible; hard boundaries were near-degenerate. | `SH1`; removed. |
| SC1 soft composition | Can soft neighborhoods support localized semantic swaps? Representation swap and random size-matched controls. | Soft statistical neighborhoods did not support localized semantic swaps. | `SC1`; removed. |
| IR1 intervention-response | Do target-unspecified response groups identify factors? Response locality and cross-factor controls. | Groups were sharp but coarse. | `IR1`; removed. |
| RID0 repeated interventions | Do repeated independently varied edges resolve the remaining ambiguity? 64 contexts, 768 balanced edges, random artifact pairing, rate-preserving nulls. | Within-group Jaccard 0.994, between-group 0.002, held-out evaluator ARI 0.213; random artifact pairing reproduced the alignment. | `RID0`; Stage-3C identifiability boundary; removed. |

The failures are substantive: unrestricted correspondence models learned shortcuts; streaming diversity did not solve learning; sparse assignment collapsed; anti-dead-slot pressure made artificial occupancy; global learned bridges failed causal translation; stable statistical groups were not compositional; intervention-response groups remained coarse; RID0 showed that repeated target-unspecified interventions did not identify fine factors.

## Conclusions and conceptual distinctions

The validated positive claim is ontology-unsupplied acquisition of a canonical, persistent, executable hybrid Skill IR on the controlled task. It does not establish evaluator-ontology discovery or discovered modular composition.

The experiments permanently distinguish:

| Availability and execution properties | Organization properties |
|---|---|
| semantic availability | statistical grouping |
| semantic uniqueness | soft relational organization |
| canonical persistence | intervention-response grouping |
| lexical identity | causal modularity |
| ontology-free resolvability | compositional modularity |
| executor sufficiency | |

These properties are not equivalent. In particular, executor sufficiency can succeed through deterministic compilation even when learned global bridges fail, and statistical grouping can be reproducible without yielding independently manipulable semantic modules.
