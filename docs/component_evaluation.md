# Component replacement evaluation

This evaluation compares replacement candidates against the frozen P6-R1
reference without retraining or changing the checkpoint or immutable suite.
The suite contains 24 groups and three conditions per group, for 72 cases.
The benchmark is `benchmarks/evaluate_component_replacements.py` and its
recorded output is `results/component_replacement_evaluation.json`.

## Results

| Component/path | Singleton recovery | Mean hypotheses | Reload | Time/case |
|---|---:|---:|---:|---:|
| P6-R1 learned correspondence | 95.83% | 1.042 | 100% | 4.16 ms |
| Exact observable permutation | 100% | 1.000 | 100% | 0.42 ms |
| Existing deterministic acquisition | 100% | 1.000 | 100% | 0.34 ms |

All paths achieved 100% “any correct hypothesis” recovery. The exact path is
approximately 10× faster than P6-R1 and removes the three residual ambiguous
artifacts on this synthetic suite. It is valid only when component occurrences
are directly observable; it is not a general replacement for opaque semantic
correspondence.

The JSON-decoder parser check achieved 100% parsing on the immutable suite,
but produced no semantic improvement because the existing examples are already
well-formed JSON. It remains a robustness candidate rather than a production
change.

Canonicalization rules were also compared:

| Rule | Canonical singleton | Abstention |
|---|---:|---:|
| Margin-first | 98.61% | 0% |
| Score-first | 100% | 0% |
| Count-first | 100% | 0% |
| Conservative count/score/margin with tie handling | 100% | 0% |

The existing conservative rule remains preferable because it preserves explicit
tie abstention, even though the immutable suite does not trigger a tie.

## Decisions by component

- Acquisition: retain the deterministic parser and relation extractor for the
  current domain. A neural extractor would add cost without improving the
  immutable-suite result.
- Correspondence: retain P6-R1 as the frozen learned control, but use exact
  observable permutation inference wherever the representation exposes the
  components. This is a synthetic-domain efficiency optimization, not a new
  claim about opaque correspondence.
- Support aggregation: retain deterministic order-independent aggregation. The
  current suite has no failure that justifies a learned Set Transformer or
  another trainable aggregator.
- Canonicalization: retain the conservative deterministic rule.
- Resolver: retain the symbolic resolver; it resolves all 72 canonicalized
  cases and avoids the failed learned bridge class.
- IR serialization: retain JSON because the current artifacts reload exactly
  and are small. A binary format would optimize storage, not semantics.
- V6-G compiler/executor: retain frozen. Replacing it would confound the
  representation comparison with a new execution experiment.

## Interpretation

The result does not show that P6-R1 was unnecessary in the original research.
It shows that the synthetic task exposes enough structure for an exact
constraint solver. On real demonstrations, the exact candidate may abstain;
P6-R1 or a newer learned matcher would then become relevant. The next valid
comparison is therefore a real-data or deliberately opaque-structure benchmark,
not further optimization of the synthetic exact path.
