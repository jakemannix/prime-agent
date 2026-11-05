import { describe, expect, test } from "vitest";
import type { SglangAdapterConfig } from "../src/core/context-editing/sglang-adapter.js";
import { applySglangContextMetadata, validateSglangEndpoint } from "../src/core/context-editing/sglang-adapter.js";

const config = {
	baseUrl: "http://127.0.0.1:30000/v1",
	model: "qwen36-27b",
	mode: "scr" as const,
	servingFingerprint: "scr-sglang0.5.16-qwen36-27b",
};
const identity = { sessionId: "root", cacheEpoch: "epoch-1", revision: 3, requestId: "turn-4" };

describe("SGLang context metadata adapter", () => {
	test("clones the final payload and writes only top-level metadata", () => {
		const payload = { model: "qwen36-27b", messages: [{ role: "user", content: "hello" }], temperature: 0 };
		const result = applySglangContextMetadata(payload, identity, config);
		expect(result).not.toBe(payload);
		expect(result).toMatchObject({ model: payload.model, messages: payload.messages, temperature: 0 });
		expect(typeof result.cache_salt).toBe("string");
		expect(typeof result.rid).toBe("string");
		expect(payload).not.toHaveProperty("cache_salt");
		expect(payload).not.toHaveProperty("rid");
		expect(result).not.toHaveProperty("extra_body");
	});

	test("keeps salt stable for ordinary revisions and namespaces epochs and modes", () => {
		const first = applySglangContextMetadata({ model: config.model }, identity, config);
		const ordinary = applySglangContextMetadata({ model: config.model }, { ...identity, revision: 99 }, config);
		const reset = applySglangContextMetadata({ model: config.model }, { ...identity, cacheEpoch: "epoch-2" }, config);
		const exact = applySglangContextMetadata({ model: config.model }, identity, { ...config, mode: "exact" });
		const changedServing = applySglangContextMetadata({ model: config.model }, identity, {
			...config,
			servingFingerprint: "different-serving",
		});
		expect(ordinary.cache_salt).toBe(first.cache_salt);
		expect(ordinary.rid).not.toBe(first.rid);
		expect(reset.cache_salt).not.toBe(first.cache_salt);
		expect(exact.cache_salt).not.toBe(first.cache_salt);
		expect(changedServing.cache_salt).not.toBe(first.cache_salt);
	});

	test("rejects nested adapter metadata and model mismatches", () => {
		expect(() =>
			applySglangContextMetadata({ model: config.model, extra_body: { cache_salt: "bad" } }, identity, config),
		).toThrow(/top-level/);
		expect(() => applySglangContextMetadata({ model: "other" }, identity, config)).toThrow(/does not match/);
		expect(() => applySglangContextMetadata({}, identity, config)).toThrow(/does not match/);
		expect(() => applySglangContextMetadata({ model: config.model, rid: "owned" }, identity, config)).toThrow(
			/second cache owner/,
		);
	});

	test("validates bounded, explicitly allowlisted endpoints without probing SCR", () => {
		expect(validateSglangEndpoint(config).mode).toBe("scr");
		expect(() => validateSglangEndpoint({ ...config, baseUrl: "https://example.com/v1" })).toThrow(/allowlisted/);
		expect(() => validateSglangEndpoint({ ...config, baseUrl: "http://192.168.1.20:30000/v1" })).toThrow(
			/allowlisted/,
		);
		expect(() => validateSglangEndpoint({ ...config, baseUrl: "http://user:pass@127.0.0.1:30000/v1" })).toThrow(
			/credentials/,
		);
		expect(
			validateSglangEndpoint({ ...config, baseUrl: "https://scr.internal/v1", allowedHosts: ["scr.internal"] }).url
				.hostname,
		).toBe("scr.internal");
		expect(() =>
			validateSglangEndpoint({
				baseUrl: config.baseUrl,
				mode: config.mode,
				servingFingerprint: config.servingFingerprint,
			} as unknown as SglangAdapterConfig),
		).toThrow(/explicit model/);
		expect(() => validateSglangEndpoint({ ...config, servingFingerprint: "" })).toThrow(/servingFingerprint/);
	});
});
