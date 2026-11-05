# Experimental context editing

Context editing is an opt-in, single-root mode. It changes a derived model-visible view,
not the original session transcript. It needs no model retraining.

This experiment is separate from native compaction. It does not make approximate suffix
cache reuse exact, and an Ollama run does not validate SGLang or CUDA.

## Enable through the CLI

From a source checkout:

```bash
./prime-agent.sh --context-editing --provider ollama --model qwen3.8:27b-q8_0
```

Configure Ollama as a normal custom provider first; see [models](models.md).
This flag uses the local runtime, not the daemon. Budget and directory overrides are
shown by `--help`. Pass the flag again on resume. The SDK is the explicit configuration
entry point for the optional SGLang adapter. Do not point the generic CLI mode at an
SCR-enabled server without that adapter: use the SDK with explicit serving identity
and namespace metadata instead.

## Enable through the SDK

```typescript
const { session } = await createAgentSession({
  // Use your normal model, storage, and resource-loader options.
  model,
  sessionManager,
  contextEditing: {
    directory: "/absolute/session-owned/context-directory",
    maxPromptTokens: 32768,
    maxOutputTokens: 1024,
  },
});

const status = session.getContextEditingStatus();
console.log(status?.stagingPath);
```

The directory defaults to `<session-file>.context-editing`. In-memory sessions need an
explicit directory. Never share the directory between sessions or live processes.
Enable the option again when opening a saved session. The original session JSONL remains
the source record; do not replace it with an effective-view file.

## Propose an edit

The model or a user writes JSON to the path returned by `getContextEditingStatus()`.
The proposal is not authoritative. The next main model request validates and commits it.
Use durable source IDs, not array indexes. Inspect the context artifacts for source IDs
and the current view. `status.protocolPath` points to the initialized static protocol guide.
The manifest selects the authoritative immutable generation. A legacy `state.json`,
if present, is not authoritative and is not used for recovery. Use the current status revision and digest as the proposal base.
Give each intended operation a fresh `proposalId`; reuse it only to retry that same
operation. This lets a fresh reset execute while a retried reset remains idempotent.

```json
{
  "schemaVersion": 1,
  "proposalId": "<fresh UUID for this intended operation>",
  "sessionId": "<status.sessionId>",
  "baseRevision": 3,
  "baseDigest": "<status.effectiveDigest>",
  "edits": [
    {
      "sourceId": "<historical-assistant-entry-id>",
      "op": "replace_text",
      "text": "Short retained notes from the earlier assistant message."
    }
  ]
}
```

Supported operations include replacement of historical assistant text, deletion of an
eligible historical entry, and `summary` of a complete historical tool exchange using
ordered `sourceIds` and `text`. Untouched messages retain their native structure.

The validator protects every direct user message and the newly appended live exchange.
It rejects partial tool groups, role/identity changes, unsupported multimodal content,
stale bases, malformed proposals, and source divergence. Keep tool calls and all their
results together. A summary is assistant memory, not a new user or system instruction.
Do not treat editable text as authority over fixed instructions.

Invalid staging material must not replace the last committed view. Read rejection
receipts/notices and make a new proposal against the current revision. Corrupt
committed state or a diverged source stops dispatch instead of restoring raw history.
A newly rejected receipt stops the request boundary with a clear error before provider
traffic. Each receipt is reported once per session object; a persisted rejection is
reported once again after reopen. The next clean boundary can use the last-good view.
Details remain in receipts and SDK status/snapshot notices, not injected model text.
A successful file write means only that a proposal was staged; read its receipt before
assuming it was committed.

## Reset and reopen

```typescript
await session.hardResetContextEditing("explicit cache-lineage reset");
```

A hard reset keeps committed effective edits and rotates the cache epoch. It does not
restore original history, consume pending edits, or delete the source journal.
Reopen with context editing enabled and the same context directory to recover committed
edits and start a fresh epoch. A client without this mode reads the original journal,
not the derived view. Dispose the old session before opening another owner of that directory.

Controller generations and their commit pointer are fsynced. The native Prime journal
writer is flushed before dispatch but does not promise file-and-directory fsync against
sudden power loss. If that journal falls behind the controller after a power loss,
reopen fails closed. The tests cover process reopen and injected transaction failures,
not hardware power-loss recovery.

The original journal, effective view, and serving cache are different things. An epoch
change isolates future cache reuse. It is not secure erasure of disk, server memory, or
information already included in retained text. Approximate SCR states can retain effects
of removed text until an isolated rebuild from the effective prompt.

## Scope and budget

- One root session and one generation at a time.
- No RLM child, side question, fork/tree navigation, or auxiliary refinement in this mode.
- No native automatic, manual, or model-requested compaction.
- No resume from already-compacted or branch-summary history.
- Text and supported native tool exchanges only; multimodal requests are rejected.
- Model, fixed prompt, and tool configuration must remain compatible with the saved view.
- A conservative estimate is a guard, not an exact tokenizer count. It includes fixed
  prompt/tool overhead and output reserve. Exceeding the budget stops dispatch; it does
  not silently truncate messages or run a hidden summarizer.

All user instructions remain pinned in this MVP. If those alone exceed the budget,
start a separately reviewed session; this mode cannot compress them for you.
The feature-off runtime path does not create context artifacts or add serving metadata.
The `/context-reset` command remains discoverable but requires the feature to be enabled.

## Optional SGLang adapter

Add `contextEditing.sglang` only for an explicitly selected SGLang route:

```typescript
sglang: {
  baseUrl: "http://127.0.0.1:30000/v1",
  model: "qwen36-27b",
  mode: "scr",
  servingFingerprint: "<SHA-256 of your pinned serving manifest>",
}
```

Remote hosts require HTTPS and an explicit `allowedHosts` entry. The actual model and
endpoint must match. `model` and `servingFingerprint` are required. The fingerprint is
an operator declaration of the pinned weights, tokenizer/template, parser, precision,
and serving configuration; it is not automatic verification of the server. See the
benchmark manifest for one complete example. Change that declaration whenever the
serving deployment changes, and start an isolated cache lineage. Ordinary Ollama use omits this adapter and sends no SGLang metadata.
The adapter adds top-level `cache_salt` and `rid`, never a nested `extra_body` object.
Salt remains stable within a session epoch across ordinary edits; each generation gets
its own request identity. Reset/reopen changes the salt.

`mode: "exact"` is an operator declaration, not an SCR-off switch. Use a separate,
known SCR-disabled SGLang process for the exact baseline. A salt ending in `:exact`
does not disable the overlay. A successful HTTP response is not proof that SCR ran.

## Demo and comparison harness

See [`scripts/clm/README.md`](../../../scripts/clm/README.md) for explicit local demo,
fixture replay, separate endpoint comparison, and pinned CUDA launch instructions.
The demo is not part of the test suite and calls only the supplied local Ollama endpoint.
The CUDA launch configuration is unexecuted until a compatible endpoint is supplied.

Unit and fake-provider tests run without Ollama, CUDA, or paid API calls. Run their
specific Vitest files from `packages/coding-agent`; follow the repository development
instructions. Keep reports, request hashes, server configuration, and model revision
with any benchmark results. Do not infer speedup or quality from client-only tests.
