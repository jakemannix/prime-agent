---
type: decision
status: accepted
date: 2026-10-08
last_updated: 2026-10-08
tags: [doctrack/type/decision, doctrack/status/accepted, doctrack/audience/claude]
---
# Preserve source; reset serving lineage without undoing edits

The journal is the auditable conversation, not editable model memory. A separate derived
view permits compaction-like edits while retaining provenance. A hard reset preserves the
committed effective view and rotates the cache namespace. Cold reopen does not trust a
remote server to retain valid cache continuity.

Rejected alternatives:
- Rewrite the transcript: loses the original record and breaks provenance/resume.
- Restore source on reset: resurrects deliberately removed context.
- Extension-only controller: caught hook failures can silently dispatch unsafe context.
- Shared parent/child cache namespace: auxiliary calls can corrupt ownership and lineage.
- Label an SCR-enabled endpoint exact using a salt: salt does not disable the algorithm.
- Claim suffix reuse is exact: reused hidden states can retain removed-prefix information.

Consequences: a deliberately restricted single-root experiment, explicit exact/SCR endpoints,
and conservative fail-closed behavior. No retraining is needed to implement this mechanism;
quality and performance still require measured evaluation.

Related: [[features/context-editing|feature]],
[[components/context-editing/controller-store|controller/store]],
[[interfaces/context-request-boundary|boundary contract]].
