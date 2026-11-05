import { createHash, randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	applySglangContextMetadata,
	validateSglangEndpoint,
} from "../../packages/coding-agent/src/core/context-editing/sglang-adapter.js";

export interface ReplayExpectation {
	mustInclude: string[];
	mustNotInclude?: string[];
	minCharacters: number;
	jsonEquals?: Record<string, string | number | boolean>;
}

export interface ReplayMessage {
	role: "system" | "user" | "assistant" | "tool";
	content: string;
}

export interface ReplayScenario {
	id: string;
	sessionId: string;
	benchmarkClass: "semantic-quality" | "scr-candidate";
	targetInputTokens: number;
	minimumCommonPrefixTokens: number;
	maxTokens: number;
	originalMessages: ReplayMessage[];
	editedMessages: ReplayMessage[];
	expected: {
		original: ReplayExpectation;
		edited: ReplayExpectation;
		reset: ReplayExpectation;
	};
}

export interface BenchmarkManifest {
	schemaVersion: 1;
	fixtureSha256: string;
	servingFingerprint: string;
	model: { id: string; revision: string };
	tokenizer: { id: string; revision: string; chatTemplateRevision: string };
	software: { sglangVersion: string; scrPatchRevision: string };
	serving: {
		contextLength: number;
		tensorParallel: number;
		dtype: string;
		cudaGraph: boolean;
	};
	scr: {
		sideSessions: number;
		sideTokens: number;
		maxBlocks: number;
		ssmMode: string;
	};
	generation: { temperature: number; topP: number; seed: number; stream: boolean };
	limits: {
		requestTimeoutMs: number;
		maxInputTokens: number;
		maxOutputTokens: number;
		maxResponseBytes: number;
		maxSerializedRequestBytes: number;
	};
}

export interface ReplayConfig {
	/** Operator-declared server role. This is not a detected capability. */
	mode: "exact" | "scr";
	baseUrl: string;
	endpointPath?: string;
	runId: string;
	/** Fresh per process invocation, even if an operator accidentally reuses runId. */
	invocationNonce: string;
	manifest: BenchmarkManifest;
	manifestSha256: string;
	fixtureSha256: string;
	/** Explicit operator declaration; must equal the fingerprint pinned in the manifest. */
	servingFingerprint: string;
	allowedHosts?: readonly string[];
}

export interface ReplayUsage extends Record<string, unknown> {
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
}

export interface ReplayQuality {
	passed: boolean;
	missing: string[];
	forbidden: string[];
	tooShort: boolean;
	invalidJson: boolean;
	jsonMismatches: string[];
}

export type ReplayPhase = "original" | "edited" | "reset";
export type ReplayResultStatus = "passed" | "quality_failed" | "request_failed" | "aborted";

export interface ScrEligibility {
	benchmarkClass: "semantic-quality" | "scr-candidate";
	targetInputTokens: number;
	minimumCommonPrefixTokens: number;
	serverCommonPrefixTokens: number | null;
	serverPlanVerified: false;
	eligibleForScrPerformance: false;
	reason: string;
}

export interface ReplayResult {
	id: string;
	scenarioId: string;
	phase: ReplayPhase;
	status: ReplayResultStatus;
	httpStatus: number | null;
	wallLatencyMs: number | null;
	/** First nonempty content or thinking delta. Null for nonstream responses and failures before a token. */
	ttftMs: number | null;
	usage: ReplayUsage | null;
	text: string;
	thinking: string;
	quality: ReplayQuality | null;
	error: string | null;
	cacheSalt: string | null;
	rid: string | null;
	logicalInputSha256: string;
	requestSha256: string | null;
	scrEligibility: ScrEligibility;
}

export interface ReplayReport {
	schemaVersion: 1;
	config: ReplayConfig;
	pinnedInputsSha256: string;
	results: ReplayResult[];
	allQualityPassed: boolean;
	cudaMeasured: false;
	scrCapabilityVerified: false;
	notes: string[];
}

export interface ComparisonDelta {
	id: string;
	qualityPassDelta: -1 | 0 | 1;
	exactPassed: boolean;
	scrPassed: boolean;
	wallLatencyMsDelta: number | null;
	ttftMsDelta: number | null;
	eligibleForScrPerformance: false;
}

export interface ReplayComparisonReport {
	schemaVersion: 1;
	exact: ReplayReport;
	scr: ReplayReport;
	deltas: ComparisonDelta[];
	summary: {
		qualityPassRateDelta: number;
		meanWallLatencyMsDelta: number | null;
		meanTtftMsDelta: number | null;
	};
	cudaMeasured: false;
	scrCapabilityVerified: false;
	notes: string[];
}

interface ParsedResponse {
	text: string;
	thinking: string;
	ttftMs: number | null;
	usage: ReplayUsage;
}

const SHA256 = /^[a-f0-9]{64}$/;
const SOURCE_REVISION = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/;
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const PHASES: readonly ReplayPhase[] = ["original", "edited", "reset"];

export function sha256(text: string): string {
	return createHash("sha256").update(text, "utf8").digest("hex");
}

function canonicalJson(value: unknown): string {
	if (value === undefined) return "null";
	if (value === null || typeof value !== "object") {
		const serialized = JSON.stringify(value);
		if (serialized === undefined) throw new Error("Canonical JSON value is not serializable");
		return serialized;
	}
	if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
	const object = value as Record<string, unknown>;
	return `{${Object.keys(object)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`)
		.join(",")}}`;
}

function assertObject(value: unknown, name: string): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} must be an object`);
	return value as Record<string, unknown>;
}

function assertExactKeys(object: Record<string, unknown>, keys: readonly string[], name: string): void {
	const expected = new Set(keys);
	const unexpected = Object.keys(object).filter((key) => !expected.has(key));
	const missing = keys.filter((key) => !Object.hasOwn(object, key));
	if (missing.length || unexpected.length) {
		throw new Error(`${name} keys mismatch; missing=[${missing.join(",")}], unexpected=[${unexpected.join(",")}]`);
	}
}

function asString(value: unknown, name: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`${name} must be a nonempty string`);
	return value;
}

function asIdentifier(value: unknown, name: string): string {
	const text = asString(value, name);
	if (!SAFE_ID.test(text)) throw new Error(`${name} must be a short opaque identifier`);
	return text;
}

function asInteger(value: unknown, name: string, minimum = 0): number {
	if (!Number.isInteger(value) || (value as number) < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
	return value as number;
}

function asFiniteNumber(value: unknown, name: string): number {
	if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be finite`);
	return value;
}

function asBoolean(value: unknown, name: string): boolean {
	if (typeof value !== "boolean") throw new Error(`${name} must be boolean`);
	return value;
}

function parseExpectation(value: unknown, name: string): ReplayExpectation {
	const object = assertObject(value, name);
	const allowed = ["mustInclude", "mustNotInclude", "minCharacters", "jsonEquals"] as const;
	const required = ["mustInclude", "minCharacters"] as const;
	const unexpected = Object.keys(object).filter((key) => !allowed.includes(key as (typeof allowed)[number]));
	const missing = required.filter((key) => !Object.hasOwn(object, key));
	if (missing.length || unexpected.length) {
		throw new Error(`${name} keys mismatch; missing=[${missing.join(",")}], unexpected=[${unexpected.join(",")}]`);
	}
	if (!Array.isArray(object.mustInclude) || object.mustInclude.length === 0) {
		throw new Error(`${name}.mustInclude must be a nonempty string array`);
	}
	const mustInclude = object.mustInclude.map((item, index) => asString(item, `${name}.mustInclude[${index}]`));
	let mustNotInclude: string[] | undefined;
	if (object.mustNotInclude !== undefined) {
		if (!Array.isArray(object.mustNotInclude)) throw new Error(`${name}.mustNotInclude must be a string array`);
		mustNotInclude = object.mustNotInclude.map((item, index) => asString(item, `${name}.mustNotInclude[${index}]`));
	}
	const forbiddenNormalized = new Set((mustNotInclude ?? []).map((item) => item.toLocaleLowerCase()));
	if (mustInclude.some((item) => forbiddenNormalized.has(item.toLocaleLowerCase()))) {
		throw new Error(`${name} cannot require and forbid the same text`);
	}
	let jsonEquals: Record<string, string | number | boolean> | undefined;
	if (object.jsonEquals !== undefined) {
		const expectedJson = assertObject(object.jsonEquals, `${name}.jsonEquals`);
		if (Object.keys(expectedJson).length === 0) throw new Error(`${name}.jsonEquals must not be empty`);
		jsonEquals = {};
		for (const [key, item] of Object.entries(expectedJson)) {
			if (!key || (typeof item !== "string" && typeof item !== "number" && typeof item !== "boolean")) {
				throw new Error(`${name}.jsonEquals must contain only primitive values`);
			}
			if (typeof item === "number" && !Number.isFinite(item)) throw new Error(`${name}.jsonEquals.${key} must be finite`);
			jsonEquals[key] = item;
		}
	}
	return {
		mustInclude,
		mustNotInclude,
		minCharacters: asInteger(object.minCharacters, `${name}.minCharacters`, 1),
		jsonEquals,
	};
}

function parseMessages(value: unknown, name: string): ReplayMessage[] {
	if (!Array.isArray(value) || value.length < 3) throw new Error(`${name} must contain at least three messages`);
	return value.map((item, index) => {
		const object = assertObject(item, `${name}[${index}]`);
		assertExactKeys(object, ["role", "content"], `${name}[${index}]`);
		const role = asString(object.role, `${name}[${index}].role`);
		if (role !== "system" && role !== "user" && role !== "assistant" && role !== "tool") {
			throw new Error(`${name}[${index}].role is invalid`);
		}
		return { role, content: asString(object.content, `${name}[${index}].content`) };
	});
}

function commonSuffixLength(left: readonly ReplayMessage[], right: readonly ReplayMessage[]): number {
	let length = 0;
	while (
		length < left.length &&
		length < right.length &&
		canonicalJson(left[left.length - length - 1]) === canonicalJson(right[right.length - length - 1])
	) length += 1;
	return length;
}

function commonPrefixLength(left: readonly ReplayMessage[], right: readonly ReplayMessage[]): number {
	let length = 0;
	while (length < left.length && length < right.length && canonicalJson(left[length]) === canonicalJson(right[length])) {
		length += 1;
	}
	return length;
}

function parseScenario(line: string, lineNumber: number): ReplayScenario {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch (error) {
		throw new Error(`Invalid fixture JSON at line ${lineNumber}: ${String(error)}`);
	}
	const object = assertObject(value, `Fixture line ${lineNumber}`);
	assertExactKeys(
		object,
		[
			"id",
			"session_id",
			"benchmark_class",
			"target_input_tokens",
			"minimum_common_prefix_tokens",
			"max_tokens",
			"original_messages",
			"edited_messages",
			"expected",
		],
		`Fixture line ${lineNumber}`,
	);
	const originalMessages = parseMessages(object.original_messages, `Fixture line ${lineNumber}.original_messages`);
	const editedMessages = parseMessages(object.edited_messages, `Fixture line ${lineNumber}.edited_messages`);
	if (canonicalJson(originalMessages) === canonicalJson(editedMessages)) {
		throw new Error(`Fixture line ${lineNumber} must change the edited middle`);
	}
	const retainedSuffixMessages = commonSuffixLength(originalMessages, editedMessages);
	if (retainedSuffixMessages === 0) {
		throw new Error(`Fixture line ${lineNumber} must retain at least one identical suffix message`);
	}
	const retainedPrefixMessages = commonPrefixLength(originalMessages, editedMessages);
	const expected = assertObject(object.expected, `Fixture line ${lineNumber}.expected`);
	assertExactKeys(expected, PHASES, `Fixture line ${lineNumber}.expected`);
	const benchmarkClass = asString(object.benchmark_class, `Fixture line ${lineNumber}.benchmark_class`);
	if (benchmarkClass !== "semantic-quality" && benchmarkClass !== "scr-candidate") {
		throw new Error(`Fixture line ${lineNumber}.benchmark_class is invalid`);
	}
	const targetInputTokens = asInteger(object.target_input_tokens, `Fixture line ${lineNumber}.target_input_tokens`, 1);
	const minimumCommonPrefixTokens = asInteger(
		object.minimum_common_prefix_tokens,
		`Fixture line ${lineNumber}.minimum_common_prefix_tokens`,
	);
	if (minimumCommonPrefixTokens > targetInputTokens) {
		throw new Error(`Fixture line ${lineNumber} minimum common prefix exceeds its target input size`);
	}
	if (benchmarkClass === "scr-candidate" && minimumCommonPrefixTokens < 1024) {
		throw new Error(`Fixture line ${lineNumber} SCR candidates require a >=1024-token server prefix`);
	}
	if (benchmarkClass === "scr-candidate") {
		const retainedPrefixCharacters = originalMessages
			.slice(0, retainedPrefixMessages)
			.reduce((total, message) => total + message.content.length, 0);
		const retainedSuffixCharacters = originalMessages
			.slice(originalMessages.length - retainedSuffixMessages)
			.reduce((total, message) => total + message.content.length, 0);
		if (retainedPrefixCharacters < minimumCommonPrefixTokens * 8) {
			throw new Error(`Fixture line ${lineNumber} SCR candidate prefix construction is too short`);
		}
		if (retainedSuffixCharacters < 2_048) {
			throw new Error(`Fixture line ${lineNumber} SCR candidate retained suffix is too short`);
		}
	}
	return {
		id: asIdentifier(object.id, `Fixture line ${lineNumber}.id`),
		sessionId: asIdentifier(object.session_id, `Fixture line ${lineNumber}.session_id`),
		benchmarkClass,
		targetInputTokens,
		minimumCommonPrefixTokens,
		maxTokens: asInteger(object.max_tokens, `Fixture line ${lineNumber}.max_tokens`, 1),
		originalMessages,
		editedMessages,
		expected: {
			original: parseExpectation(expected.original, `Fixture line ${lineNumber}.expected.original`),
			edited: parseExpectation(expected.edited, `Fixture line ${lineNumber}.expected.edited`),
			reset: parseExpectation(expected.reset, `Fixture line ${lineNumber}.expected.reset`),
		},
	};
}

export function parseJsonlFixture(text: string): ReplayScenario[] {
	const scenarios = text
		.split(/\r?\n/)
		.map((line, index) => ({ line, index }))
		.filter(({ line }) => line.trim() && !line.trimStart().startsWith("#"))
		.map(({ line, index }) => parseScenario(line, index + 1));
	if (scenarios.length === 0) throw new Error("Fixture must contain at least one scenario");
	const ids = new Set<string>();
	for (const scenario of scenarios) {
		if (ids.has(scenario.id)) throw new Error(`Duplicate fixture scenario id '${scenario.id}'`);
		ids.add(scenario.id);
	}
	return scenarios;
}

export function computeServingFingerprint(
	manifest: Pick<BenchmarkManifest, "model" | "tokenizer" | "software" | "serving" | "scr">,
): string {
	return sha256(
		canonicalJson({
			model: manifest.model,
			tokenizer: manifest.tokenizer,
			software: manifest.software,
			serving: manifest.serving,
			scr: manifest.scr,
		}),
	);
}

export function parseBenchmarkManifest(text: string): BenchmarkManifest {
	let value: unknown;
	try {
		value = JSON.parse(text);
	} catch (error) {
		throw new Error(`Invalid benchmark manifest JSON: ${String(error)}`);
	}
	const root = assertObject(value, "Benchmark manifest");
	assertExactKeys(
		root,
		[
			"schemaVersion",
			"fixtureSha256",
			"servingFingerprint",
			"model",
			"tokenizer",
			"software",
			"serving",
			"scr",
			"generation",
			"limits",
		],
		"Benchmark manifest",
	);
	if (root.schemaVersion !== 1) throw new Error("Benchmark manifest schemaVersion must be 1");
	const fixtureSha256 = asString(root.fixtureSha256, "Benchmark manifest.fixtureSha256");
	if (!SHA256.test(fixtureSha256)) throw new Error("Benchmark manifest.fixtureSha256 must be a lowercase SHA-256");
	const model = assertObject(root.model, "Benchmark manifest.model");
	assertExactKeys(model, ["id", "revision"], "Benchmark manifest.model");
	const tokenizer = assertObject(root.tokenizer, "Benchmark manifest.tokenizer");
	assertExactKeys(tokenizer, ["id", "revision", "chatTemplateRevision"], "Benchmark manifest.tokenizer");
	const software = assertObject(root.software, "Benchmark manifest.software");
	assertExactKeys(software, ["sglangVersion", "scrPatchRevision"], "Benchmark manifest.software");
	const serving = assertObject(root.serving, "Benchmark manifest.serving");
	assertExactKeys(serving, ["contextLength", "tensorParallel", "dtype", "cudaGraph"], "Benchmark manifest.serving");
	const scr = assertObject(root.scr, "Benchmark manifest.scr");
	assertExactKeys(scr, ["sideSessions", "sideTokens", "maxBlocks", "ssmMode"], "Benchmark manifest.scr");
	const generation = assertObject(root.generation, "Benchmark manifest.generation");
	assertExactKeys(generation, ["temperature", "topP", "seed", "stream"], "Benchmark manifest.generation");
	const limits = assertObject(root.limits, "Benchmark manifest.limits");
	assertExactKeys(
		limits,
		["requestTimeoutMs", "maxInputTokens", "maxOutputTokens", "maxResponseBytes", "maxSerializedRequestBytes"],
		"Benchmark manifest.limits",
	);
	const manifest: BenchmarkManifest = {
		schemaVersion: 1,
		fixtureSha256,
		servingFingerprint: asString(root.servingFingerprint, "Benchmark manifest.servingFingerprint"),
		model: {
			id: asIdentifier(model.id, "Benchmark manifest.model.id"),
			revision: asString(model.revision, "Benchmark manifest.model.revision"),
		},
		tokenizer: {
			id: asString(tokenizer.id, "Benchmark manifest.tokenizer.id"),
			revision: asString(tokenizer.revision, "Benchmark manifest.tokenizer.revision"),
			chatTemplateRevision: asString(tokenizer.chatTemplateRevision, "Benchmark manifest.tokenizer.chatTemplateRevision"),
		},
		software: {
			sglangVersion: asString(software.sglangVersion, "Benchmark manifest.software.sglangVersion"),
			scrPatchRevision: asString(software.scrPatchRevision, "Benchmark manifest.software.scrPatchRevision"),
		},
		serving: {
			contextLength: asInteger(serving.contextLength, "Benchmark manifest.serving.contextLength", 1),
			tensorParallel: asInteger(serving.tensorParallel, "Benchmark manifest.serving.tensorParallel", 1),
			dtype: asString(serving.dtype, "Benchmark manifest.serving.dtype"),
			cudaGraph: asBoolean(serving.cudaGraph, "Benchmark manifest.serving.cudaGraph"),
		},
		scr: {
			sideSessions: asInteger(scr.sideSessions, "Benchmark manifest.scr.sideSessions", 1),
			sideTokens: asInteger(scr.sideTokens, "Benchmark manifest.scr.sideTokens", 1),
			maxBlocks: asInteger(scr.maxBlocks, "Benchmark manifest.scr.maxBlocks", 1),
			ssmMode: asString(scr.ssmMode, "Benchmark manifest.scr.ssmMode"),
		},
		generation: {
			temperature: asFiniteNumber(generation.temperature, "Benchmark manifest.generation.temperature"),
			topP: asFiniteNumber(generation.topP, "Benchmark manifest.generation.topP"),
			seed: asInteger(generation.seed, "Benchmark manifest.generation.seed"),
			stream: asBoolean(generation.stream, "Benchmark manifest.generation.stream"),
		},
		limits: {
			requestTimeoutMs: asInteger(limits.requestTimeoutMs, "Benchmark manifest.limits.requestTimeoutMs", 100),
			maxInputTokens: asInteger(limits.maxInputTokens, "Benchmark manifest.limits.maxInputTokens", 1),
			maxOutputTokens: asInteger(limits.maxOutputTokens, "Benchmark manifest.limits.maxOutputTokens", 1),
			maxResponseBytes: asInteger(limits.maxResponseBytes, "Benchmark manifest.limits.maxResponseBytes", 1024),
			maxSerializedRequestBytes: asInteger(
				limits.maxSerializedRequestBytes,
				"Benchmark manifest.limits.maxSerializedRequestBytes",
				1024,
			),
		},
	};
	if (manifest.generation.topP <= 0 || manifest.generation.topP > 1) {
		throw new Error("Benchmark generation.topP must be > 0 and <= 1");
	}
	if (manifest.limits.requestTimeoutMs > 600_000) throw new Error("Benchmark request timeout must not exceed 600000ms");
	if (manifest.limits.maxInputTokens + manifest.limits.maxOutputTokens > manifest.serving.contextLength) {
		throw new Error("Benchmark input and output token limits exceed the serving context length");
	}
	for (const [name, revision] of [
		["model revision", manifest.model.revision],
		["tokenizer revision", manifest.tokenizer.revision],
		["chat-template revision", manifest.tokenizer.chatTemplateRevision],
		["SCR patch revision", manifest.software.scrPatchRevision],
	] as const) {
		if (!SOURCE_REVISION.test(revision)) throw new Error(`Benchmark ${name} must be a full source revision`);
	}
	if (!SHA256.test(manifest.servingFingerprint)) {
		throw new Error("Benchmark servingFingerprint must be a lowercase SHA-256");
	}
	if (manifest.servingFingerprint !== computeServingFingerprint(manifest)) {
		throw new Error("Benchmark servingFingerprint does not match the pinned serving inputs");
	}
	return manifest;
}

export function assertQuality(text: string, expected: ReplayExpectation): ReplayQuality {
	const normalized = text.toLocaleLowerCase();
	const missing = expected.mustInclude.filter((needle) => !normalized.includes(needle.toLocaleLowerCase()));
	const forbidden = (expected.mustNotInclude ?? []).filter((needle) => normalized.includes(needle.toLocaleLowerCase()));
	const tooShort = text.trim().length < expected.minCharacters;
	let invalidJson = false;
	const jsonMismatches: string[] = [];
	if (expected.jsonEquals) {
		let parsed: unknown;
		try {
			parsed = JSON.parse(text);
		} catch {
			invalidJson = true;
		}
		if (!invalidJson) {
			if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) invalidJson = true;
			else {
				const actual = parsed as Record<string, unknown>;
				for (const [key, expectedValue] of Object.entries(expected.jsonEquals)) {
					if (actual[key] !== expectedValue) jsonMismatches.push(key);
				}
			}
		}
	}
	return {
		passed: missing.length === 0 && forbidden.length === 0 && !tooShort && !invalidJson && jsonMismatches.length === 0,
		missing,
		forbidden,
		tooShort,
		invalidJson,
		jsonMismatches,
	};
}

function parseUsage(value: unknown): ReplayUsage {
	const object = assertObject(value, "Replay response usage");
	const promptTokens = asInteger(object.prompt_tokens, "Replay response usage.prompt_tokens");
	const completionTokens = asInteger(object.completion_tokens, "Replay response usage.completion_tokens");
	const totalTokens = asInteger(object.total_tokens, "Replay response usage.total_tokens");
	if (totalTokens !== promptTokens + completionTokens) {
		throw new Error("Replay response usage.total_tokens must equal prompt_tokens + completion_tokens");
	}
	return { ...structuredClone(object), prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: totalTokens };
}

function deltaText(delta: Record<string, unknown>, key: "content" | "reasoning_content" | "thinking"): string {
	const value = delta[key];
	return typeof value === "string" ? value : "";
}

async function readSseResponse(response: Response, startedAt: number, maxBytes: number): Promise<ParsedResponse> {
	if (!response.body) throw new Error("Streaming replay response has no body");
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let text = "";
	let thinking = "";
	let ttftMs: number | null = null;
	let usage: ReplayUsage | null = null;
	let bytesRead = 0;
	let sawDone = false;

	const consume = (event: string): void => {
		const data = event
			.split(/\r?\n/)
			.filter((line) => line.startsWith("data:"))
			.map((line) => line.slice(5).trimStart())
			.join("\n");
		if (!data) return;
		if (data === "[DONE]") {
			sawDone = true;
			return;
		}
		let parsed: unknown;
		try {
			parsed = JSON.parse(data);
		} catch (error) {
			throw new Error(`Invalid SSE JSON: ${String(error)}`);
		}
		const object = assertObject(parsed, "SSE event");
		if (object.usage !== undefined && object.usage !== null) usage = parseUsage(object.usage);
		if (!Array.isArray(object.choices) || !object.choices[0]) return;
		const choice = assertObject(object.choices[0], "SSE choice");
		if (choice.delta === undefined || choice.delta === null) return;
		const delta = assertObject(choice.delta, "SSE delta");
		const contentDelta = deltaText(delta, "content");
		const thinkingDelta =
			deltaText(delta, "reasoning_content") || deltaText(delta, "thinking") ||
			(typeof delta.reasoning === "string" ? delta.reasoning : "");
		if (ttftMs === null && (contentDelta.length > 0 || thinkingDelta.length > 0)) ttftMs = performance.now() - startedAt;
		text += contentDelta;
		thinking += thinkingDelta;
	};

	for (;;) {
		const part = await reader.read();
		if (part.done) break;
		bytesRead += part.value.byteLength;
		if (bytesRead > maxBytes) throw new Error(`Replay response exceeded ${maxBytes} bytes`);
		buffer += decoder.decode(part.value, { stream: true });
		for (;;) {
			const match = /\r?\n\r?\n/.exec(buffer);
			if (!match || match.index === undefined) break;
			const event = buffer.slice(0, match.index);
			buffer = buffer.slice(match.index + match[0].length);
			consume(event);
		}
	}
	buffer += decoder.decode();
	if (buffer.trim()) consume(buffer);
	if (!sawDone) throw new Error("Replay SSE response ended before [DONE]");
	if (!usage) throw new Error("Replay SSE response did not return usage");
	return { text, thinking, ttftMs, usage };
}

async function readBoundedText(response: Response, maxBytes: number): Promise<string> {
	if (!response.body) return "";
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let bytesRead = 0;
	let text = "";
	for (;;) {
		const part = await reader.read();
		if (part.done) break;
		bytesRead += part.value.byteLength;
		if (bytesRead > maxBytes) throw new Error(`Replay response exceeded ${maxBytes} bytes`);
		text += decoder.decode(part.value, { stream: true });
	}
	return text + decoder.decode();
}

async function readJsonResponse(response: Response, maxBytes: number): Promise<ParsedResponse> {
	const body = await readBoundedText(response, maxBytes);
	let parsed: unknown;
	try {
		parsed = JSON.parse(body);
	} catch (error) {
		throw new Error(`Invalid replay JSON response: ${String(error)}`);
	}
	const object = assertObject(parsed, "Replay response");
	if (!Array.isArray(object.choices) || !object.choices[0]) throw new Error("Replay response has no choice");
	const choice = assertObject(object.choices[0], "Replay response choice");
	let text = "";
	let thinking = "";
	if (choice.message !== undefined) {
		const message = assertObject(choice.message, "Replay response message");
		text = typeof message.content === "string" ? message.content : "";
		thinking = typeof message.reasoning_content === "string" ? message.reasoning_content : "";
	} else if (typeof choice.text === "string") text = choice.text;
	return { text, thinking, ttftMs: null, usage: parseUsage(object.usage) };
}

function endpointFor(config: ReplayConfig): string {
	return new URL(config.endpointPath ?? "chat/completions", config.baseUrl.endsWith("/") ? config.baseUrl : `${config.baseUrl}/`).toString();
}

function pinnedInputs(manifest: BenchmarkManifest): unknown {
	return {
		servingFingerprint: manifest.servingFingerprint,
		model: manifest.model,
		tokenizer: manifest.tokenizer,
		software: manifest.software,
		serving: manifest.serving,
		scr: manifest.scr,
		generation: manifest.generation,
		limits: manifest.limits,
	};
}

function validateConfig(config: ReplayConfig): void {
	asIdentifier(config.runId, "Replay runId");
	asIdentifier(config.invocationNonce, "Replay invocation nonce");
	if (!SHA256.test(config.manifestSha256)) throw new Error("Replay manifestSha256 must be a lowercase SHA-256");
	if (!SHA256.test(config.fixtureSha256)) throw new Error("Replay fixtureSha256 must be a lowercase SHA-256");
	if (config.fixtureSha256 !== config.manifest.fixtureSha256) throw new Error("Fixture SHA-256 does not match the benchmark manifest");
	if (!SHA256.test(config.servingFingerprint) || config.servingFingerprint !== config.manifest.servingFingerprint) {
		throw new Error("Operator serving fingerprint must match the benchmark manifest pin");
	}
	if (computeServingFingerprint(config.manifest) !== config.manifest.servingFingerprint) {
		throw new Error("Benchmark serving fingerprint does not match the pinned serving inputs");
	}
	validateSglangEndpoint({
		baseUrl: config.baseUrl,
		model: config.manifest.model.id,
		mode: config.mode,
		servingFingerprint: config.servingFingerprint,
		allowedHosts: config.allowedHosts,
	});
}

function stepFor(scenario: ReplayScenario, phase: ReplayPhase): { messages: ReplayMessage[]; expected: ReplayExpectation } {
	return phase === "original"
		? { messages: scenario.originalMessages, expected: scenario.expected.original }
		: { messages: scenario.editedMessages, expected: scenario.expected[phase] };
}

function eligibilityFor(scenario: ReplayScenario): ScrEligibility {
	return {
		benchmarkClass: scenario.benchmarkClass,
		targetInputTokens: scenario.targetInputTokens,
		minimumCommonPrefixTokens: scenario.minimumCommonPrefixTokens,
		serverCommonPrefixTokens: null,
		serverPlanVerified: false,
		eligibleForScrPerformance: false,
		reason: "No server common-prefix count or SCR plan evidence is attached to this client-only result",
	};
}

function failedResult(
	id: string,
	scenario: ReplayScenario,
	phase: ReplayPhase,
	logicalInputSha256: string,
	requestSha256: string | null,
	cacheSalt: string | null,
	rid: string | null,
	error: string,
	httpStatus: number | null,
	wallLatencyMs: number | null,
): ReplayResult {
	return {
		id,
		scenarioId: scenario.id,
		phase,
		status: "request_failed",
		httpStatus,
		wallLatencyMs,
		ttftMs: null,
		usage: null,
		text: "",
		thinking: "",
		quality: null,
		error,
		cacheSalt,
		rid,
		logicalInputSha256,
		requestSha256,
		scrEligibility: eligibilityFor(scenario),
	};
}

export async function replayScenarios(
	scenarios: ReplayScenario[],
	config: ReplayConfig,
	fetchImpl: typeof fetch = fetch,
): Promise<ReplayReport> {
	validateConfig(config);
	if (scenarios.length === 0) throw new Error("Replay requires at least one scenario");
	const endpoint = endpointFor(config);
	const results: ReplayResult[] = [];
	const seenRids = new Set<string>();
	for (const [scenarioIndex, scenario] of scenarios.entries()) {
		if (scenario.targetInputTokens > config.manifest.limits.maxInputTokens) {
			throw new Error(`Scenario '${scenario.id}' target_input_tokens exceeds the pinned input limit`);
		}
		if (scenario.maxTokens > config.manifest.limits.maxOutputTokens) {
			throw new Error(`Scenario '${scenario.id}' max_tokens exceeds the pinned output limit`);
		}
		let warmLineageAvailable = true;
		for (const [phaseIndex, phase] of PHASES.entries()) {
			const id = `${scenario.id}/${phase}`;
			const step = stepFor(scenario, phase);
			const logicalPayload = {
				model: config.manifest.model.id,
				messages: step.messages,
				max_tokens: scenario.maxTokens,
				temperature: config.manifest.generation.temperature,
				top_p: config.manifest.generation.topP,
				seed: config.manifest.generation.seed,
				stream: config.manifest.generation.stream,
				stream_options: config.manifest.generation.stream ? { include_usage: true } : undefined,
			};
			const logicalInputSha256 = sha256(canonicalJson(logicalPayload));
			if (phase === "edited" && !warmLineageAvailable) {
				results.push({
					...failedResult(
						id,
						scenario,
						phase,
						logicalInputSha256,
						null,
						null,
						null,
						"Original warm request failed; edited request was not sent",
						null,
						null,
					),
					status: "aborted",
				});
				continue;
			}
			const epoch = `${config.runId}-${config.invocationNonce}-${scenarioIndex}-${phase === "reset" ? 1 : 0}`;
			const requestId = `${config.runId}-${config.invocationNonce}-${scenarioIndex}-${phaseIndex}`;
			const payload = applySglangContextMetadata(
				logicalPayload,
				{ sessionId: scenario.sessionId, cacheEpoch: epoch, revision: phaseIndex, requestId },
				{
					baseUrl: config.baseUrl,
					model: config.manifest.model.id,
					mode: config.mode,
					servingFingerprint: config.servingFingerprint,
					allowedHosts: config.allowedHosts,
				},
			);
			const cacheSalt = String(payload.cache_salt);
			const rid = String(payload.rid);
			if (seenRids.has(rid)) throw new Error(`Generated duplicate SGLang rid '${rid}'`);
			seenRids.add(rid);
			const body = JSON.stringify(payload);
			if (Buffer.byteLength(body, "utf8") > config.manifest.limits.maxSerializedRequestBytes) {
				throw new Error(`Scenario '${scenario.id}' exceeds the pinned serialized request size`);
			}
			const requestSha256 = sha256(body);
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), config.manifest.limits.requestTimeoutMs);
			const startedAt = performance.now();
			try {
				const response = await fetchImpl(endpoint, {
					method: "POST",
					headers: { "content-type": "application/json" },
					body,
					signal: controller.signal,
				});
				if (!response.ok) {
					const errorBody = (await readBoundedText(response, config.manifest.limits.maxResponseBytes)).slice(0, 500);
					throw Object.assign(new Error(`HTTP ${response.status}: ${errorBody}`), { httpStatus: response.status });
				}
				const isSse = response.headers.get("content-type")?.toLowerCase().includes("text/event-stream") ?? false;
				const parsed = config.manifest.generation.stream && isSse
					? await readSseResponse(response, startedAt, config.manifest.limits.maxResponseBytes)
					: await readJsonResponse(response, config.manifest.limits.maxResponseBytes);
				if (parsed.usage.prompt_tokens > config.manifest.limits.maxInputTokens) {
					throw new Error(
						`Replay prompt_tokens ${parsed.usage.prompt_tokens} exceeds pinned maxInputTokens ${config.manifest.limits.maxInputTokens}`,
					);
				}
				if (parsed.usage.completion_tokens > scenario.maxTokens) {
					throw new Error(`Replay completion_tokens exceeds scenario max_tokens for '${scenario.id}'`);
				}
				if (parsed.usage.total_tokens > config.manifest.serving.contextLength) {
					throw new Error("Replay total_tokens exceeds the pinned serving context length");
				}
				const wallLatencyMs = performance.now() - startedAt;
				const quality = assertQuality(parsed.text, step.expected);
				results.push({
					id,
					scenarioId: scenario.id,
					phase,
					status: quality.passed ? "passed" : "quality_failed",
					httpStatus: response.status,
					wallLatencyMs,
					ttftMs: parsed.ttftMs,
					usage: parsed.usage,
					text: parsed.text,
					thinking: parsed.thinking,
					quality,
					error: null,
					cacheSalt,
					rid,
					logicalInputSha256,
					requestSha256,
					scrEligibility: eligibilityFor(scenario),
				});
			} catch (error) {
				const wallLatencyMs = performance.now() - startedAt;
				const httpStatus = typeof error === "object" && error !== null && "httpStatus" in error
					? Number((error as { httpStatus: unknown }).httpStatus)
					: null;
				results.push(
					failedResult(
						id,
						scenario,
						phase,
						logicalInputSha256,
						requestSha256,
						cacheSalt,
						rid,
						error instanceof Error ? error.message : String(error),
						httpStatus,
						wallLatencyMs,
					),
				);
				if (phase === "original") warmLineageAvailable = false;
			} finally {
				clearTimeout(timer);
			}
		}
	}
	return {
		schemaVersion: 1,
		config,
		pinnedInputsSha256: sha256(canonicalJson(pinnedInputs(config.manifest))),
		results,
		allQualityPassed: results.every((result) => result.status === "passed"),
		cudaMeasured: false,
		scrCapabilityVerified: false,
		notes: [
			`${config.mode} is an operator-declared endpoint role, not a detected server capability.`,
			"No CUDA execution, SCR relocation, cache hit, or GPU timing is proven by this client report.",
			"TTFT is the first nonempty content/thinking SSE delta; it is null for nonstream mode.",
			"SCR performance eligibility remains false without server common-prefix and plan telemetry.",
		],
	};
}

function endpointAuthority(config: ReplayConfig): string {
	const url = new URL(endpointFor(config));
	return url.origin.toLowerCase();
}

export function assertComparableConfigs(exact: ReplayConfig, scr: ReplayConfig): void {
	validateConfig(exact);
	validateConfig(scr);
	if (exact.mode !== "exact" || scr.mode !== "scr") throw new Error("Comparison requires exact and SCR configs in that order");
	if (endpointAuthority(exact) === endpointAuthority(scr)) {
		throw new Error("Exact and SCR comparison requires separate server authorities");
	}
	if (exact.manifestSha256 !== scr.manifestSha256) throw new Error("Comparison requires the same manifest revision");
	if (exact.fixtureSha256 !== scr.fixtureSha256) throw new Error("Comparison requires the same fixture revision");
	if (sha256(canonicalJson(pinnedInputs(exact.manifest))) !== sha256(canonicalJson(pinnedInputs(scr.manifest)))) {
		throw new Error("Comparison requires identical model, tokenizer, serving, generation, and limit inputs");
	}
}

export function assertSeparateModeEndpoints(configs: readonly ReplayConfig[]): void {
	const exact = configs.find((config) => config.mode === "exact");
	const scr = configs.find((config) => config.mode === "scr");
	if (exact && scr) assertComparableConfigs(exact, scr);
}

function mean(values: readonly number[]): number | null {
	return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function compareReplayReports(exact: ReplayReport, scr: ReplayReport): ReplayComparisonReport {
	assertComparableConfigs(exact.config, scr.config);
	if (exact.pinnedInputsSha256 !== scr.pinnedInputsSha256) throw new Error("Report pinned inputs differ");
	const scrById = new Map(scr.results.map((result) => [result.id, result]));
	if (exact.results.length !== scr.results.length) throw new Error("Comparison reports have different result counts");
	const deltas = exact.results.map((exactResult): ComparisonDelta => {
		const scrResult = scrById.get(exactResult.id);
		if (!scrResult) throw new Error(`SCR report is missing result '${exactResult.id}'`);
		if (exactResult.logicalInputSha256 !== scrResult.logicalInputSha256) {
			throw new Error(`Comparison semantic request differs for '${exactResult.id}'`);
		}
		const exactPassed = exactResult.status === "passed";
		const scrPassed = scrResult.status === "passed";
		return {
			id: exactResult.id,
			qualityPassDelta: (Number(scrPassed) - Number(exactPassed)) as -1 | 0 | 1,
			exactPassed,
			scrPassed,
			wallLatencyMsDelta:
				exactResult.wallLatencyMs === null || scrResult.wallLatencyMs === null
					? null
					: scrResult.wallLatencyMs - exactResult.wallLatencyMs,
			ttftMsDelta:
				exactResult.ttftMs === null || scrResult.ttftMs === null ? null : scrResult.ttftMs - exactResult.ttftMs,
			eligibleForScrPerformance: false,
		};
	});
	const wallDeltas = deltas.flatMap((delta) => delta.wallLatencyMsDelta === null ? [] : [delta.wallLatencyMsDelta]);
	const ttftDeltas = deltas.flatMap((delta) => delta.ttftMsDelta === null ? [] : [delta.ttftMsDelta]);
	return {
		schemaVersion: 1,
		exact,
		scr,
		deltas,
		summary: {
			qualityPassRateDelta: mean(deltas.map((delta) => delta.qualityPassDelta)) ?? 0,
			meanWallLatencyMsDelta: mean(wallDeltas),
			meanTtftMsDelta: mean(ttftDeltas),
		},
		cudaMeasured: false,
		scrCapabilityVerified: false,
		notes: [
			"Deltas are SCR minus exact for paired logical generations.",
			"The exact endpoint is operator-declared as a separate SCR-disabled server.",
			"These client measurements do not prove server-side SCR relocation or CUDA execution.",
		],
	};
}

export async function replayComparison(
	scenarios: ReplayScenario[],
	exactConfig: ReplayConfig,
	scrConfig: ReplayConfig,
	fetchImpl: typeof fetch = fetch,
): Promise<ReplayComparisonReport> {
	assertComparableConfigs(exactConfig, scrConfig);
	const exact = await replayScenarios(scenarios, exactConfig, fetchImpl);
	const scr = await replayScenarios(scenarios, scrConfig, fetchImpl);
	return compareReplayReports(exact, scr);
}

export async function loadBenchmarkInputs(
	fixturePath: string,
	manifestPath: string,
): Promise<{ scenarios: ReplayScenario[]; manifest: BenchmarkManifest; fixtureSha256: string; manifestSha256: string }> {
	const [fixtureText, manifestText] = await Promise.all([readFile(fixturePath, "utf8"), readFile(manifestPath, "utf8")]);
	const fixtureSha256 = sha256(fixtureText);
	const manifest = parseBenchmarkManifest(manifestText);
	if (fixtureSha256 !== manifest.fixtureSha256) {
		throw new Error(`Fixture SHA-256 ${fixtureSha256} does not match manifest pin ${manifest.fixtureSha256}`);
	}
	return { scenarios: parseJsonlFixture(fixtureText), manifest, fixtureSha256, manifestSha256: sha256(manifestText) };
}

function arg(args: string[], name: string): string | undefined {
	const index = args.indexOf(name);
	return index >= 0 ? args[index + 1] : undefined;
}

function args(args: string[], name: string): string[] {
	return args.flatMap((value, index) => value === name && args[index + 1] ? [args[index + 1] as string] : []);
}

async function main(argv: string[]): Promise<void> {
	const fixturePath = resolve(arg(argv, "--fixture") ?? "scripts/clm/fixture.jsonl");
	const manifestPath = resolve(arg(argv, "--manifest") ?? "scripts/clm/benchmark-manifest.json");
	const mode = arg(argv, "--mode");
	const runId = arg(argv, "--run-id");
	const servingFingerprint = arg(argv, "--serving-fingerprint");
	if ((mode !== "exact" && mode !== "scr" && mode !== "compare") || !runId || !servingFingerprint) {
		throw new Error(
			"Usage: replay.ts --mode exact|scr|compare --run-id ID --serving-fingerprint SHA256 " +
				"[--base-url URL | --exact-base-url URL --scr-base-url URL] [--allow-host HOST]",
		);
	}
	const loaded = await loadBenchmarkInputs(fixturePath, manifestPath);
	const allowedHosts = args(argv, "--allow-host");
	const common = {
		runId,
		invocationNonce: randomUUID(),
		manifest: loaded.manifest,
		manifestSha256: loaded.manifestSha256,
		fixtureSha256: loaded.fixtureSha256,
		servingFingerprint,
		allowedHosts,
	};
	let report: ReplayReport | ReplayComparisonReport;
	if (mode === "compare") {
		const exactBaseUrl = arg(argv, "--exact-base-url");
		const scrBaseUrl = arg(argv, "--scr-base-url");
		if (!exactBaseUrl || !scrBaseUrl) throw new Error("Comparison requires --exact-base-url and --scr-base-url");
		report = await replayComparison(
			loaded.scenarios,
			{ ...common, mode: "exact", baseUrl: exactBaseUrl },
			{ ...common, mode: "scr", baseUrl: scrBaseUrl },
		);
	} else {
		const baseUrl = arg(argv, "--base-url");
		if (!baseUrl) throw new Error("Single-mode replay requires --base-url");
		report = await replayScenarios(loaded.scenarios, { ...common, mode, baseUrl });
	}
	const serialized = `${JSON.stringify(report, null, 2)}\n`;
	const output = arg(argv, "--output");
	if (output) await writeFile(output, serialized);
	else process.stdout.write(serialized);
	const passed = "allQualityPassed" in report
		? report.allQualityPassed
		: report.exact.allQualityPassed && report.scr.allQualityPassed;
	if (!passed) process.exitCode = 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	void main(process.argv.slice(2)).catch((error: unknown) => {
		process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
		process.exitCode = 1;
	});
}
