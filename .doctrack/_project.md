---
project: prime-agent
monorepo: true
type: index
doctrack_version: "3.0.0"
last_updated: 2026-10-08
tags: [doctrack/type/index, doctrack/status/active, doctrack/audience/claude]
---
# Scoped implementation knowledge

Coverage is limited to the experimental context-editing change. This is not a full
repository initialization. No MCP, Obsidian, or user configuration was changed.

## Feature

[[features/context-editing|Context editing]] connects the SDK/session integration to
[[interfaces/context-request-boundary|the request boundary]] and the
[[components/context-editing/controller-store|durable controller]].

## File registry

| Source file | Component |
| --- | --- |
| packages/coding-agent/src/cli/daemon-launch.ts | Skip early daemon start for context editing |
| packages/coding-agent/src/cli/owned-session-worker.ts | Skip worker frontend for local-only mode |
| packages/coding-agent/src/core/sdk.ts | Runtime request wrappers |
| packages/coding-agent/src/core/agent-session.ts | Persistence barrier and guards |
| packages/coding-agent/src/core/context-editing/controller.ts | Proposal validation and snapshots |
| packages/coding-agent/src/core/context-editing/store.ts | Durable source and effective view |
| packages/coding-agent/src/core/context-editing/types.ts | Protocol contracts |
| packages/coding-agent/src/core/context-editing/sglang-adapter.ts | Explicit serving metadata |
| scripts/clm/ollama-demo.ts | Local SDK demonstration |
| scripts/clm/replay.ts | Endpoint benchmark |
| scripts/clm/compare-reports.ts | Offline sequential-server comparison |
| scripts/clm/server-evidence.template.json | Unverified serving-evidence checklist |
