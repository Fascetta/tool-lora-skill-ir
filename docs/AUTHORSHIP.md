# Authorship and provenance

This standalone repository isolates the work of Christian Bianchi (`Fascetta`) from the shared source repository.

## Git boundary

The shared repository contains four commits by Luca Romani, ending at commit `2948f0b`, followed by the project commits authored by `Fascetta`:

- `18ff604`: functional-delta adapter supervision;
- `cad5d32`: multi-executor Skill IR;
- `1574be4`: exhaustive executor validation;
- `77b020d`: hidden-protocol Skill IR;
- `82221c9`: decoupled generic and skill fast weights;
- `c3f1b44`: validated Stage 1 closure;
- `acd8e0a`: Stage 1 documentation correction;
- `f6c9d7a`: Stage 2 closure;
- `ca9b86b`: Stage 3 closure and maintained pipeline.

The later Stage 4 benchmarks, modules, checkpoints, and manifests were developed in the same workspace after `ca9b86b` and are included because they continue Christian's project.

## Exclusion policy

The original data loaders, generic hypernetwork training framework, oracle-LoRA trainer, and conversation-training pipeline authored before Christian's branch are not reproduced here. Adapter metrics produced by that inherited framework are labelled as comparison baselines in the report and README.

Files under `src/tool_lora/skill_ir/`, `src/tool_lora/stage4_*`, the corresponding benchmarks and tests, and the adapter-diagnostic scripts constitute the isolated project code. The course-template files under `report/` are redistributed only to preserve the required report format.
