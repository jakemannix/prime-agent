import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { resolve } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import {
	assertComparableConfigs,
	type BenchmarkManifest,
	compareReplayReports,
	computeServingFingerprint,
	loadBenchmarkInputs,
	parseBenchmarkManifest,
	parseJsonlFixture,
	type ReplayConfig,
	type ReplayScenario,
	replayScenarios,
	sha256,
} from "../../../scripts/clm/replay.js";

const servers: Server[] = [];
afterEach(async () => {
	await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
});

const fixtureSha256 = "f".repeat(64);
const manifestSha256 = "e".repeat(64);
const sourceRevision = "a".repeat(40);
const manifestWithoutFingerprint: Omit<BenchmarkManifest, "servingFingerprint"> = {
	schemaVersion: 1,
	fixtureSha256,
	model: { id: "mock", revision: sourceRevision },
	tokenizer: { id: "mock-tokenizer", revision: sourceRevision, chatTemplateRevision: sourceRevision },
	software: { sglangVersion: "0.5.16", scrPatchRevision: "b".repeat(40) },
	serving: { contextLength: 4096, tensorParallel: 1, dtype: "bfloat16", cudaGraph: false },
	scr: { sideSessions: 1, sideTokens: 4096, maxBlocks: 1, ssmMode: "fork" },
	generation: { temperature: 0, topP: 1, seed: 0, stream: true },
	limits: {
		requestTimeoutMs: 1_000,
		maxInputTokens: 2048,
		maxOutputTokens: 128,
		maxResponseBytes: 64 * 1024,
		maxSerializedRequestBytes: 64 * 1024,
	},
};
const manifest: BenchmarkManifest = {
	...manifestWithoutFingerprint,
	servingFingerprint: computeServingFingerprint(manifestWithoutFingerprint),
};

const scenario: ReplayScenario = {
	id: "middle-edit",
	sessionId: "session",
	benchmarkClass: "semantic-quality",
	targetInputTokens: 128,
	minimumCommonPrefixTokens: 0,
	maxTokens: 16,
	originalMessages: [
		{ role: "system", content: "Answer with the current capital." },
		{ role: "user", content: "The current capital is Paris; London is obsolete." },
		{ role: "user", content: "Return the current capital and no obsolete value." },
	],
	editedMessages: [
		{ role: "system", content: "Answer with the current capital." },
		{ role: "user", content: "The current capital is Paris; Rome is obsolete." },
		{ role: "user", content: "Return the current capital and no obsolete value." },
	],
	expected: {
		original: { mustInclude: ["Paris"], mustNotInclude: ["London"], minCharacters: 5 },
		edited: { mustInclude: ["Paris"], mustNotInclude: ["Rome"], minCharacters: 5 },
		reset: { mustInclude: ["Paris"], mustNotInclude: ["Rome"], minCharacters: 5 },
	},
};

function config(baseUrl: string, mode: "exact" | "scr", overrides: Partial<ReplayConfig> = {}): ReplayConfig {
	return {
		mode,
		baseUrl,
		runId: "test-run",
		invocationNonce: "test-invocation",
		manifest,
		manifestSha256,
		fixtureSha256,
		servingFingerprint: manifest.servingFingerprint,
		...overrides,
	};
}

function endpoint(handler: (request: IncomingMessage, response: ServerResponse) => void): Promise<string> {
	const server = createServer(handler);
	servers.push(server);
	return new Promise((resolve) => {
		server.listen(0, "127.0.0.1", () => {
			const address = server.address();
			if (!address || typeof address === "string") throw new Error("mock server did not bind");
			resolve(`http://127.0.0.1:${address.port}/v1/`);
		});
	});
}

async function requestJson(request: IncomingMessage): Promise<Record<string, unknown>> {
	const chunks: Buffer[] = [];
	for await (const chunk of request) chunks.push(Buffer.from(chunk));
	return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
}

function sendSse(
	response: ServerResponse,
	text = "Paris is current.",
	tokenDelayMs = 0,
	usage = { prompt_tokens: 4, completion_tokens: 5, total_tokens: 9 },
): void {
	response.writeHead(200, { "content-type": "text/event-stream" });
	response.write(`: keepalive\r\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n`);
	setTimeout(() => {
		const thinking = Buffer.from(`data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\r\n\r\n`, "utf8");
		response.write(thinking.subarray(0, 17));
		response.write(thinking.subarray(17));
		response.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\r\n\r\n`);
		response.write(
			`data: ${JSON.stringify({
				choices: [],
				usage: { ...usage, prompt_tokens_details: { cached_tokens: 2 } },
			})}\r\n\r\n`,
		);
		response.end("data: [DONE]\r\n\r\n");
	}, tokenDelayMs);
}

describe("CLM replay harness", () => {
	test("uses real SSE token events for TTFT and injects stable salt with unique request IDs", async () => {
		const requests: Record<string, unknown>[] = [];
		const baseUrl = await endpoint(async (request, response) => {
			requests.push(await requestJson(request));
			sendSse(response, "Paris is current.", 15);
		});
		const report = await replayScenarios([scenario], config(baseUrl, "scr"));

		expect(report.allQualityPassed).toBe(true);
		expect(report.results).toHaveLength(3);
		expect(report.results[0]?.ttftMs).toBeGreaterThan(5);
		expect(report.results[0]?.wallLatencyMs).toBeGreaterThanOrEqual(report.results[0]?.ttftMs ?? 0);
		expect(report.results[0]?.thinking).toBe("think");
		expect(report.results[0]?.usage?.prompt_tokens_details).toEqual({ cached_tokens: 2 });
		expect(requests.map((request) => request.cache_salt)).toEqual([
			requests[0]?.cache_salt,
			requests[0]?.cache_salt,
			requests[2]?.cache_salt,
		]);
		expect(requests[2]?.cache_salt).not.toBe(requests[0]?.cache_salt);
		expect(new Set(requests.map((request) => request.rid)).size).toBe(3);
		expect(requests.every((request) => request.stream === true)).toBe(true);
		expect(requests.every((request) => !Object.hasOwn(request, "extra_body"))).toBe(true);
		expect(report.results[1]?.logicalInputSha256).toBe(report.results[2]?.logicalInputSha256);
		expect(report.results.every((result) => result.requestSha256 && result.cacheSalt && result.rid)).toBe(true);
		expect(report.cudaMeasured).toBe(false);
		expect(report.scrCapabilityVerified).toBe(false);
	});

	test("rotates salts and request IDs across invocations even when runId is reused", async () => {
		const baseUrl = await endpoint((_request, response) => sendSse(response));
		const first = await replayScenarios([scenario], config(baseUrl, "scr", { invocationNonce: "invocation-a" }));
		const second = await replayScenarios([scenario], config(baseUrl, "scr", { invocationNonce: "invocation-b" }));
		expect(second.results[0]?.cacheSalt).not.toBe(first.results[0]?.cacheSalt);
		const firstRids = new Set(first.results.map((result) => result.rid));
		expect(second.results.every((result) => result.rid && !firstRids.has(result.rid))).toBe(true);
	});

	test("reports null TTFT honestly when an endpoint returns nonstream JSON", async () => {
		const baseUrl = await endpoint((_request, response) => {
			response.writeHead(200, { "content-type": "application/json" });
			response.end(
				JSON.stringify({
					choices: [{ message: { content: "Paris is current." } }],
					usage: { prompt_tokens: 4, completion_tokens: 5, total_tokens: 9 },
				}),
			);
		});
		const report = await replayScenarios([scenario], config(baseUrl, "exact"));
		expect(report.results.map((result) => result.ttftMs)).toEqual([null, null, null]);
		expect(report.results.every((result) => result.status === "passed")).toBe(true);
	});

	test("keeps HTTP failures and aborted warm edits in the report, then executes the cold reset", async () => {
		let requests = 0;
		const baseUrl = await endpoint((_request, response) => {
			requests += 1;
			if (requests === 1) {
				response.writeHead(500, { "content-type": "text/plain" });
				response.end("synthetic failure");
				return;
			}
			sendSse(response);
		});
		const report = await replayScenarios([scenario], config(baseUrl, "scr"));
		expect(requests).toBe(2);
		expect(report.results.map((result) => result.status)).toEqual(["request_failed", "aborted", "passed"]);
		expect(report.results[0]?.httpStatus).toBe(500);
		expect(report.results[0]?.error).toContain("synthetic failure");
		expect(report.results[1]?.requestSha256).toBeNull();
		expect(report.results[2]?.cacheSalt).not.toBe(report.results[0]?.cacheSalt);
		expect(report.allQualityPassed).toBe(false);
	});

	test("fails content and structured quality even for nonempty output", async () => {
		const structuredScenario: ReplayScenario = {
			...scenario,
			expected: {
				original: { ...scenario.expected.original, jsonEquals: { capital: "Paris" } },
				edited: { ...scenario.expected.edited, jsonEquals: { capital: "Paris" } },
				reset: { ...scenario.expected.reset, jsonEquals: { capital: "Paris" } },
			},
		};
		const baseUrl = await endpoint((_request, response) => sendSse(response, '{"capital":"London"}'));
		const report = await replayScenarios([structuredScenario], config(baseUrl, "scr"));
		expect(report.allQualityPassed).toBe(false);
		expect(report.results[0]?.status).toBe("quality_failed");
		expect(report.results[0]?.quality?.missing).toEqual(["Paris"]);
		expect(report.results[0]?.quality?.forbidden).toEqual(["London"]);
		expect(report.results[0]?.quality?.jsonMismatches).toEqual(["capital"]);
	});

	test("rejects same-authority or mismatched-input comparisons before making requests", () => {
		const exact = config("http://127.0.0.1:30000/exact/", "exact");
		const scr = config("http://127.0.0.1:30000/scr/", "scr");
		const changedInputs = { ...manifest, model: { ...manifest.model, revision: "c".repeat(40) } };
		const changedManifest = {
			...changedInputs,
			servingFingerprint: computeServingFingerprint(changedInputs),
		};
		expect(() => assertComparableConfigs(exact, scr)).toThrow(/separate server authorities/);
		expect(() =>
			assertComparableConfigs(exact, {
				...config("http://127.0.0.1:30001/v1/", "scr"),
				servingFingerprint: "d".repeat(64),
			}),
		).toThrow(/Operator serving fingerprint/);
		expect(() =>
			assertComparableConfigs(exact, {
				...config("http://127.0.0.1:30001/v1/", "scr"),
				manifestSha256: "d".repeat(64),
			}),
		).toThrow(/same manifest revision/);
		expect(() =>
			assertComparableConfigs(exact, {
				...config("http://127.0.0.1:30001/v1/", "scr"),
				manifest: changedManifest,
				servingFingerprint: changedManifest.servingFingerprint,
			}),
		).toThrow(/identical model, tokenizer, serving, generation, and limit inputs/);
	});

	test("emits paired quality and wall-latency deltas without claiming SCR proof", async () => {
		const exactUrl = await endpoint((_request, response) => sendSse(response));
		const scrUrl = await endpoint((_request, response) => sendSse(response));
		const exact = await replayScenarios([scenario], config(exactUrl, "exact"));
		const scr = await replayScenarios([scenario], config(scrUrl, "scr"));
		const comparison = compareReplayReports(exact, scr);
		expect(comparison.deltas).toHaveLength(3);
		expect(comparison.deltas.every((delta) => delta.qualityPassDelta === 0)).toBe(true);
		expect(comparison.deltas.every((delta) => typeof delta.wallLatencyMsDelta === "number")).toBe(true);
		expect(comparison.summary.meanWallLatencyMsDelta).not.toBeNull();
		expect(comparison.cudaMeasured).toBe(false);
		expect(comparison.scrCapabilityVerified).toBe(false);
	});

	test("enforces manifest context budgets and returned prompt-token limits", async () => {
		const invalidManifest = {
			...manifest,
			serving: { ...manifest.serving, contextLength: 2_100 },
		};
		expect(() => parseBenchmarkManifest(JSON.stringify(invalidManifest))).toThrow(
			/exceed the serving context length/,
		);

		const baseUrl = await endpoint((_request, response) => {
			sendSse(response, "Paris is current.", 0, {
				prompt_tokens: 2_049,
				completion_tokens: 5,
				total_tokens: 2_054,
			});
		});
		const report = await replayScenarios([scenario], config(baseUrl, "scr"));
		expect(report.results.map((result) => result.status)).toEqual(["request_failed", "aborted", "request_failed"]);
		expect(report.results[0]?.error).toContain("exceeds pinned maxInputTokens");
		expect(report.results[2]?.error).toContain("exceeds pinned maxInputTokens");
	});

	test("loads the checked-in benchmark bytes and verifies their exact pins", async () => {
		const loaded = await loadBenchmarkInputs(
			resolve("../../scripts/clm/fixture.jsonl"),
			resolve("../../scripts/clm/benchmark-manifest.json"),
		);
		expect(loaded.fixtureSha256).toBe("d66af595c24fd033478b693a5482345fe2f5e33d9a896921b8911e3c2f054b58");
		expect(loaded.manifestSha256).toBe("1180940f1d5251135989b00afa325297fb962d0a62aa514043fc12d116c2e837");
		expect(loaded.manifest.fixtureSha256).toBe(loaded.fixtureSha256);
		expect(loaded.manifest.servingFingerprint).toBe(computeServingFingerprint(loaded.manifest));
		expect(loaded.scenarios.map((item) => item.id)).toEqual([
			"dossier-middle-edit",
			"dossier-4k",
			"dossier-16k",
			"dossier-28k",
		]);
		expect(loaded.scenarios.slice(1).every((item) => item.minimumCommonPrefixTokens >= 1024)).toBe(true);
	});

	test("strict fixture validation rejects unknown fields and fixtures without an edited middle", () => {
		const valid = {
			id: "x",
			session_id: "s",
			benchmark_class: "semantic-quality",
			target_input_tokens: 128,
			minimum_common_prefix_tokens: 0,
			max_tokens: 8,
			original_messages: scenario.originalMessages,
			edited_messages: scenario.originalMessages,
			expected: scenario.expected,
		};
		expect(() => parseJsonlFixture(`${JSON.stringify({ ...valid, surprise: true })}\n`)).toThrow(/keys mismatch/);
		expect(() => parseJsonlFixture(`${JSON.stringify(valid)}\n`)).toThrow(/change the edited middle/);
		expect(sha256("fixture")).toMatch(/^[a-f0-9]{64}$/);
	});
});
