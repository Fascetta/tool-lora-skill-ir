# Maintained Skill IR architecture

Skill IR is the persistent, executor-facing representation learned from demonstrations. The maintained pipeline is:

```text
demonstrations
  -> ontology-unsupplied acquisition
  -> exact observable correspondence or frozen P6-R1 learned correspondence
  -> canonicalization
  -> persistent hybrid Skill IR
  -> support removal and reload
  -> ontology-free structured resolver
  -> validated semantic execution interface
  -> frozen V6-G compiler
  -> ephemeral LoRA adapter
  -> frozen Qwen execution
```

The public implementation is in `src/tool_lora/skill_ir/`:

- `acquisition.py` parses observable demonstrations and stores generic relation records.
- `correspondence_matcher.py` contains both the frozen P6-R1 matcher and the
  exact observable-permutation candidate. The deterministic candidate is the
  efficient path for the current synthetic representation; P6-R1 remains the
  frozen learned control for opaque correspondence.
- `canonicalization.py` performs deterministic consensus selection and abstains on unresolved ties.
- `resolver.py` maps generic persisted relations to the validated semantic interface. Its permutation convention is `target[j] = source[permutation[j]]`.
- `protocol_ir.py`, `compiler.py`, and `executors.py` provide the validated V6-G compilation/execution boundary.

Each persisted record contains an output key, generic relation labels and keys, transformation hypotheses, and a lexical sidecar of opaque literals. Serialization deliberately excludes support text and evaluator ontology labels. A reload therefore exercises the support-removal boundary rather than a cache.

Skill IR != adapter weights. Skill IR is persistent scientific state: relational structure plus lexical identity. The compiler is frozen, and the generated LoRA factors are ephemeral execution state. They are produced for one invocation, applied to the frozen base model, and are not used as the Skill IR representation.

The existing compiler/executor integration remains the independent validation boundary. The cleaned repository does not support learned global bridge alternatives; the maintained reconnect is the structured resolver.
