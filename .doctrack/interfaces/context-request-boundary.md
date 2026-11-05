---
type: interface
status: experimental
last_updated: 2026-10-08
files:
  - packages/coding-agent/src/core/sdk.ts
  - packages/coding-agent/src/core/agent-session.ts
  - packages/coding-agent/src/core/context-editing/types.ts
tags: [doctrack/type/interface, doctrack/status/active, doctrack/audience/claude]
---
# Mandatory request boundary

`AgentSession` drains its queued lifecycle persistence before reading branch source entries.
Do not wait for Agent idle from an active generation and do not await the queue from itself.
The controller then prepares a frozen view. SDK wrappers bind that view to the request ID
and cache epoch. Optional extension error handling cannot enforce this contract: mandatory
validation must throw outside catch-and-continue hooks. No late hook may silently change
the prompt after budget/protocol validation.

SGLang metadata belongs at the top level (`cache_salt`, `rid`) and only on the explicitly
configured route. Salt is stable within an epoch; reset/reopen rotates it. Ollama gets
neither field. HTTP retries and outer Agent retries need explicit snapshot semantics.

Queued journal failures are sticky for the opt-in session. `flushNow()` runs before
source enumeration, but the native journal does not guarantee file/directory fsync
against power loss. A new rejected receipt stops dispatch once per session object;
the next clean boundary may use last-good context. Reopen reports a persisted rejected
receipt once again. Controller generations are fsynced; journal divergence fails closed.

CLI locality must be enforced before `main.ts` loads: `shouldStartDaemonEarly` and
owned-worker frontend classification both bypass `--context-editing`. The later
`forceLocal` runtime choice alone cannot prevent a pre-main global-daemon probe.

Related: [[features/context-editing|feature]],
[[components/context-editing/controller-store|controller/store]],
[[decisions/preserve-source-and-isolate-epochs|decision]].
