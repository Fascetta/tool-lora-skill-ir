# Reproducibility

The frozen Stage-3 reference assets are checked in under `results/stage3d_p6_r1/`:

| Asset | SHA-256 |
|---|---|
| P6-R1 checkpoint | `d2721f3684ec7f0dc16bf46834098a35cf9e07d42499e89fe439ea47ac5cb827` |
| immutable suite | `8a8b1d71b1a7dd61973086c1041d4503887caeddced3f1562ab91fd22e6263a5` |

The frozen P6 manifest records the original training commit, training-script hash, Python/Torch versions, device, and deterministic settings. This standalone release does not regenerate either frozen asset; it verifies and evaluates them directly.

Inference must load the preserved checkpoint and immutable inputs. Deterministic settings are enabled by `load_p6`; the final tests also verify both hashes. Cross-process and serialization checks compare the bytes and the reloaded persistent artifact, not merely in-memory equality.

The persisted transformation tuple is `(source_index_for_target_0, source_index_for_target_1, source_index_for_target_2, source_delimiter, target_delimiter)`, with:

```text
target[j] = source[permutation[j]]
```

Canonicalization is a generic consensus over matcher evidence, ordered by count, score, and margin, with deterministic tie abstention. It does not consult evaluator labels.

Same random seed != artifact identity. A seed only describes a stochastic procedure; it does not preserve the exact learned checkpoint, input suite, optimizer state, or serialization bytes. Downstream scientific audits must therefore load the preserved checkpoint and immutable suite and record their checksums.

Run the maintained checks with:

```bash
PYTHONPATH=src pytest -q tests/test_skill_ir_final.py tests/test_resolver.py tests/test_skill_ir_stage1_closure.py tests/test_skill_ir_stage2_invariants.py
```
