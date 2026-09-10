# Stage 3A design boundary: hidden-ontology induction

Stage 2 is closed. Stage 3 must ask whether useful semantic Skill IRs can be
induced when the ontology is not manually supplied by the task definition.

## Controlled target

Use a hidden-ontology generator with ground-truth factors reserved for
evaluation. The learner receives demonstrations but not factor names, slot
structure, or an ontology. Alternate demonstration sets should express the
same underlying skill, while counterfactual supports express incompatible
skills over similar surface forms.

## Required separation

```text
demonstrations → semantic acquisition → persistent/canonical Skill IR
→ optional composition → executor-specific compilation → behavior
```

Acquisition and persistence must be evaluated independently of compilation.
The Stage-2 matched/shuffled/constant controls remain mandatory.

## Required controls

- matched support;
- counterfactual support;
- constant or generic representation;
- same-skill alternate demonstrations;
- different-skill similar demonstrations;
- held-out semantic combinations;
- fresh-process persistence;
- executor reconnect after persistence.

## Explicit non-goals for the scaffold

Do not choose a discovery architecture prematurely. Do not add another
compression sweep or executor variant. Do not implement a large model until
the generator, canonicalization criteria, persistence format, and causal
evaluation matrix are specified.
