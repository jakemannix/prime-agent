import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { compareSavedReportFiles, compareSavedReports } from "../../../scripts/clm/compare-reports.js";
import { type BenchmarkManifest, computeServingFingerprint, type ReplayReport } from "../../../scripts/clm/replay.js";

const temporaryDirectories: string[] = [];
afterEach(async () => {
	await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

const revision = "a".repeat(40);
const manifestWithoutFingerprint: Omit<BenchmarkManifest, "servingFingerprint"> = {
	schemaVersion: 1,
	fixtureSha256: "f".repeat(64),
	model: { id: "mock", revision },
	tokenizer: { id: "mock-tokenizer", revision, chatTemplateRevision: revision },
	software: { sglangVersion: "0.5.16", scrPatchRevision: "b".repeat(40) },
	serving: { contextLength: 4096, tensorParallel: 1, dtype: "bfloat16", cudaGraph: false },
	scr: { sideSessions: 1, sideTokens: 4096, maxBlocks: 1, ssmMode: "fork" },
	generation: { temperature: 0, topP: 1, seed: 0, stream: true },
	limits: {
		requestTimeoutMs: 1000,
		maxInputTokens: 2048,
		maxOutputTokens: 128,
		maxResponseBytes: 65536,
		maxSerializedRequestBytes: 65536,
	},
};
const manifest: BenchmarkManifest = {
	...manifestWithoutFingerprint,
	servingFingerprint: computeServingFingerprint(manifestWithoutFingerprint),
};

function report(mode: "exact" | "scr", port: number, wallLatencyMs: number, ttftMs: number): ReplayReport {
	return {
		schemaVersion: 1,
		config: {
			mode,
			baseUrl: `http://127.0.0.1:${port}/v1/`,
			runId: "offline-pair",
			invocationNonce: "saved-invocation",
			manifest,
			manifestSha256: "e".repeat(64),
			fixtureSha256: manifest.fixtureSha256,
			servingFingerprint: manifest.servingFingerprint,
		},
		pinnedInputsSha256: "c".repeat(64),
		results: [
			{
				id: "scenario/original",
				scenarioId: "scenario",
				phase: "original",
				status: "passed",
				httpStatus: 200,
				wallLatencyMs,
				ttftMs,
				usage: { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105 },
				text: "correct",
				thinking: "",
				quality: {
					passed: true,
					missing: [],
					forbidden: [],
					tooShort: false,
					invalidJson: false,
					jsonMismatches: [],
				},
				error: null,
				cacheSalt: `${mode}-salt`,
				rid: `${mode}-rid`,
				logicalInputSha256: "d".repeat(64),
				requestSha256: mode === "exact" ? "1".repeat(64) : "2".repeat(64),
				scrEligibility: {
					benchmarkClass: "scr-candidate",
					targetInputTokens: 4096,
					minimumCommonPrefixTokens: 1024,
					serverCommonPrefixTokens: null,
					serverPlanVerified: false,
					eligibleForScrPerformance: false,
					reason: "no server evidence",
				},
			},
		],
		allQualityPassed: true,
		cudaMeasured: false,
		scrCapabilityVerified: false,
		notes: [],
	};
}

describe("offline CLM report comparison", () => {
	test("compares saved reports without a serving endpoint", () => {
		const comparison = compareSavedReports(report("exact", 30001, 120, 80), report("scr", 30000, 90, 50));
		expect(comparison.deltas[0]).toMatchObject({
			qualityPassDelta: 0,
			wallLatencyMsDelta: -30,
			ttftMsDelta: -30,
			eligibleForScrPerformance: false,
		});
		expect(comparison.cudaMeasured).toBe(false);
		expect(comparison.scrCapabilityVerified).toBe(false);
	});

	test("reads and writes report files and rejects swapped modes", async () => {
		const directory = await mkdtemp(join(tmpdir(), "clm-offline-"));
		temporaryDirectories.push(directory);
		const exactPath = join(directory, "exact.json");
		const scrPath = join(directory, "scr.json");
		const outputPath = join(directory, "comparison.json");
		await Promise.all([
			writeFile(exactPath, JSON.stringify(report("exact", 30001, 120, 80))),
			writeFile(scrPath, JSON.stringify(report("scr", 30000, 90, 50))),
		]);

		const comparison = await compareSavedReportFiles(exactPath, scrPath, outputPath);
		expect(comparison.summary.meanWallLatencyMsDelta).toBe(-30);
		expect(JSON.parse(await readFile(outputPath, "utf8"))).toEqual(comparison);
		await expect(compareSavedReportFiles(scrPath, exactPath)).rejects.toThrow(/Saved exact replay report has mode/);
	});
});
