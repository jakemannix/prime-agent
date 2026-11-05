# CLM/SCR replay harness

This directory contains a reproducible, client-side quality and latency harness for operator-provided SGLang endpoints. It does not include a CUDA endpoint or measured CUDA result.

The harness uses Prime's real `applySglangContextMetadata` adapter. Every request sends top-level `cache_salt` and `rid` fields. It does not send adapter data in `extra_body`.

## What a scenario measures

Each JSONL row is one ordered three-request scenario:

1. `original` warms the declared server under one cache epoch.
2. `edited` changes a middle message, retains the suffix byte-for-byte, and reuses that epoch.
3. `reset` repeats the edited semantic request under a new epoch.

The original and edited requests have the same `cache_salt`. The reset request has a different salt. Every logical generation has a unique `rid`. A new scenario also starts in a new epoch. If the original HTTP request fails, its warm edit is recorded as `aborted`; the cold reset still runs and is recorded.

The included dossier fixture changes the authoritative codename in the middle of a multi-turn prompt. An unchanged later table maps both codenames to different bays and provides a separate checksum. Structured output assertions require the model to select the edited codename, derive the matching bay from the retained suffix, and retain the checksum. This is not a one-turn fact-parroting case. The fixture contains four scenarios: a small
semantic case plus 4k/16k/28k **target-size candidates** with a substantial unchanged
prefix and suffix. These labels are not measured tokenizer counts. The reviewed SCR
matcher needs at least 1,024 common-prefix tokens before the edit. Client reports keep
SCR performance eligibility false until server token counts and relocation-plan telemetry
establish that the matching path actually ran.

Each invocation adds and records a fresh nonce, even when a human-readable run ID is
reused. This prevents a new invocation from accidentally inheriting the previous run's
salts or request IDs.

## Pinned inputs and hashes

`benchmark-manifest.json` pins the exact model, tokenizer, chat-template, SGLang, and SCR source revisions, along with serving, generation, timeout, and byte-limit inputs. Source revisions are full commit hashes, not descriptive labels. Its `servingFingerprint` is the canonical SHA-256 of the pinned model, tokenizer, software, serving, and SCR fields.

At startup the harness:

- hashes the exact fixture and manifest bytes with SHA-256;
- rejects a fixture whose hash differs from the manifest pin;
- validates both files strictly and rejects unknown fields;
- requires `--serving-fingerprint` to exactly match the manifest and verifies it against the pinned fields;
- records the file hashes and a canonical shared-input hash in the report;
- records a canonical semantic-input hash and exact wire-body hash for every attempted request.

After intentionally editing the fixture, update `fixtureSha256` with the exact file hash before running:

```sh
shasum -a 256 scripts/clm/fixture.jsonl
```

Do not change pins to describe an environment that was not used.

## Run one operator-declared endpoint

Streaming is pinned in the manifest. TTFT is measured from the monotonic request start to the first nonempty SSE `content`, `reasoning_content`, `reasoning`, or `thinking` delta. Role headers, keepalives, empty deltas, finish events, and usage events do not count. If a server returns ordinary JSON, TTFT is `null`; response arrival is never mislabeled as token latency.

```sh
npx tsx scripts/clm/replay.ts \
  --mode scr \
  --run-id scr-eval-001 \
  --serving-fingerprint 18467467f7b3b7f9e1cb344fa23b035914852a8d86f09b6099ea298ef25fe7f7 \
  --base-url http://127.0.0.1:30000/v1/ \
  --fixture scripts/clm/fixture.jsonl \
  --manifest scripts/clm/benchmark-manifest.json \
  --output artifacts/clm-scr-eval-001.json
```

Local hosts are accepted by default. A remote host must be explicitly allowlisted and should be private or authenticated outside the URL:

```sh
npx tsx scripts/clm/replay.ts \
  --mode scr \
  --run-id scr-eval-remote-001 \
  --serving-fingerprint 18467467f7b3b7f9e1cb344fa23b035914852a8d86f09b6099ea298ef25fe7f7 \
  --base-url https://gpu.example.internal/v1/ \
  --allow-host gpu.example.internal \
  --output artifacts/clm-scr-eval-remote-001.json
```

The `--mode exact` and `--mode scr` values are operator declarations. The client does not detect or enable SCR.

## Paired exact/SCR comparison

A valid baseline is a separately launched, operator-declared SCR-disabled server. Do not toggle reuse within an existing cache lineage and call the result exact. Comparison mode rejects URLs on the same server authority, even if their paths differ. It also requires the exact same manifest hash, fixture hash, model revision, tokenizer/template revisions, serving configuration, sampling inputs, limits, and paired semantic request hashes.

```sh
npx tsx scripts/clm/replay.ts \
  --mode compare \
  --run-id paired-eval-001 \
  --serving-fingerprint 18467467f7b3b7f9e1cb344fa23b035914852a8d86f09b6099ea298ef25fe7f7 \
  --exact-base-url http://127.0.0.1:30001/v1/ \
  --scr-base-url http://127.0.0.1:30000/v1/ \
  --output artifacts/clm-paired-eval-001.json
```

The comparison emits paired `qualityPassDelta`, `wallLatencyMsDelta`, and nullable `ttftMsDelta` values. Deltas are SCR minus exact. Request and protocol failures remain in the report rather than terminating the run. Exact/SCR configuration errors fail before network traffic.

## Sequential comparison on one GPU

Two BF16 27B servers do not fit together on one 80 GB GPU. Use separate serving
processes **at different times**, and keep their endpoint ports distinct:

1. Start `launch-exact.sh` in a clean SCR-disabled environment on port 30001.
2. Run `replay.ts --mode exact` with the same pins shown above and save `exact.json`.
3. Stop that server and confirm its GPU allocation has been released.
4. Start `launch-scr.sh` in the pinned SCR environment on port 30000.
5. Run `replay.ts --mode scr` with the same pins and save `scr.json`.
6. Compare the saved reports offline. No server is needed for this step:

```sh
npx tsx scripts/clm/compare-reports.ts \
  --exact artifacts/exact.json --scr artifacts/scr.json \
  --output artifacts/comparison.json
```

The offline comparator applies the same pin, endpoint, and semantic-input guards as
live comparison. Do not change the manifest between runs. Record server startup
evidence for each process. Use several paired runs before drawing latency conclusions.

## Server evidence before a GPU result

The launch scripts are templates, not a deployment verifier. They explicitly request
BF16 and the model revision, but do not establish the installed SGLang/SCR version,
loaded tokenizer/template, absence of an exact-server overlay, or actual CUDA execution.

Copy `server-evidence.template.json` beside each run report. In the **serving environment**,
capture Python/package versions, the SCR checkout commit, GPU/driver information, the
exact launch command, and startup logs. Check every observed setting against the manifest.
Verify the loaded tokenizer/template as well as model weights. Never fill an observation
with a desired value without evidence. For an SCR performance claim, also retain actual
common-prefix token counts and relocation-plan telemetry. The client does not ingest or
verify this sidecar; its `cudaMeasured`, `scrCapabilityVerified`, and performance-eligibility
flags stay false. Server-side verification is a separate required evaluation step.

For a private remote host, a loopback SSH tunnel can keep the server bound to localhost.
The replay CLI does not implement an authentication-header option. Do not put credentials
in a URL, manifest, report, or this evidence file.

## Interpretation limits

Every report sets:

```json
{
  "cudaMeasured": false,
  "scrCapabilityVerified": false
}
```

A successful response, a `cached_tokens` usage field, or a favorable latency delta does not prove SCR relocation, cache hits, CUDA execution, or exact inference. Establish those claims separately with server startup evidence, SCR plan telemetry, GPU measurements, and the lifecycle/provenance tests described in the source audit. SCR is approximate and history-dependent.

The manifest targets SGLang `0.5.16`, Qwen/Qwen3.6-27B revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, and SCR revision `18dc11115f50f261233c5bba7937834491e307e8`. The launch scripts are operator templates for reviewed Linux/NVIDIA environments. They were not executed here.

## Real local Ollama SDK demo

Run these explicitly from the repository root. They contact local Ollama and are **not**
part of the automated test suite. Install/pull the named model first. Use a fresh output
directory for each main run; the resume command uses that same directory.

```sh
npx tsx scripts/clm/ollama-demo.ts \
  --model qwen3.8:27b-q8_0 \
  --out artifacts/clm-ollama-demo

npx tsx scripts/clm/ollama-demo.ts \
  --model qwen3.8:27b-q8_0 \
  --out artifacts/clm-ollama-demo \
  --resume-existing
```

The first process uses the real Prime SDK and session journal. It stages an edit of a
historical assistant memo, captures the outgoing payload, checks source preservation,
reopens the session object, and resets its cache epoch without undoing the edit. The
second process proves a fresh-process resume. Reports record assertions, responses,
latency and usage. This is a client-side Ollama baseline, not a CUDA/SCR measurement.
Ollama requests deliberately omit SGLang metadata. The transcript contains synthetic
migration notes, not private user history.

The mode does not install itself into your running Prime binary. From a source checkout,
use the CLI flag or SDK option documented in
[`context-editing.md`](../../packages/coding-agent/docs/context-editing.md).

Check script types **without executing the demo**:

```sh
npx tsgo --noEmit -p scripts/clm/tsconfig.json
```

Do not import the demo entrypoint as a syntax check: it runs a live demo.

## Tests

Tests use only in-process mock HTTP/SSE servers. They make no provider, paid, or CUDA requests.

From `packages/coding-agent`:

```sh
npx tsx ../../node_modules/vitest/dist/cli.js --run test/clm-harness.test.ts test/clm-compare-reports.test.ts
```
