# Stage-1 consolidation inventory

This inventory was completed before consolidation. Historical V6 entry points
and results remain in place.

| Area | Classification | Canonical location / reason |
|---|---|---|
| Legacy SHA-256 IR | KEEP AS CANONICAL | `src/tool_lora/skill_ir/protocol_ir.py` |
| Hidden protocol generator and `group_key` routing | HISTORICAL TEST SUPPORT; removed from maintained runtime | recoverable from `pre-stage4-stage3-full-history` |
| V6-M seven relational heads + hourglass | HISTORICAL; removed from maintained runtime | compact V6-M support artifact is preserved |
| V6-M joint field scorer | PORT INTO `src/` | `field_permutation.py` |
| V6-J date-order scorer | HISTORICAL; removed from maintained checkout | recoverable from `pre-stage4-stage3-full-history` |
| V6-K lexical separator pointer | PORT INTO `src/` | `separator_pointer.py` |
| V6-G compiler | PORT INTO `src/` | `compiler.py`, active-tree checkpoint |
| V6-N reconnect | HISTORICAL; removed from maintained checkout | recoverable from `pre-stage4-stage3-full-history` |
| V6-L migration | KEEP AS HISTORICAL DIAGNOSTIC | provenance and equivalence record |
| V6-M scripts | KEEP AS HISTORICAL DIAGNOSTIC | training/reproducibility evidence |
| Anonymous Perceiver-only path | DO NOT IMPORT | failed/non-validated acquisition path |
| Flattened-vector V6-B, old 7-row/staging nine-row decoders | DO NOT IMPORT | invalid or unstable probes |
| Learned separator contextual-to-static matcher | DO NOT IMPORT | failed fresh generalization |
| ProtocolIRv2, V4/V5 latent machinery | DEPRECATE FOR STAGE 1 | retained only as history |

No experimental artifact was deleted.
