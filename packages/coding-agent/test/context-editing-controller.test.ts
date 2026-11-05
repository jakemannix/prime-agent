import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
	ContextEditingController,
	ContextEditingError,
	type ContextEditOperation,
	type ContextEditProposal,
	type SourceEntry,
} from "../src/core/context-editing/index.js";

const user = (text: string): AgentMessage =>
	({ role: "user", content: [{ type: "text", text }], timestamp: 1 }) as AgentMessage;
const assistant = (text: string): AgentMessage =>
	({
		role: "assistant",
		content: [{ type: "text", text }],
		api: "test",
		provider: "test",
		model: "test",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 2,
	}) as AgentMessage;
const toolAssistant = (id = "call-1"): AgentMessage =>
	({ ...assistant("working"), content: [{ type: "toolCall", id, name: "read", arguments: {} }] }) as AgentMessage;
const toolResult = (id = "call-1"): AgentMessage =>
	({
		role: "toolResult",
		toolCallId: id,
		toolName: "read",
		content: [{ type: "text", text: "result" }],
		isError: false,
		timestamp: 3,
	}) as AgentMessage;
const multiToolAssistant = (): AgentMessage =>
	({
		...assistant("working"),
		content: [
			{ type: "toolCall", id: "call-1", name: "read", arguments: {} },
			{ type: "toolCall", id: "call-2", name: "read", arguments: {} },
		],
	}) as AgentMessage;
const dir = () => mkdtempSync(join(tmpdir(), "clm-controller-"));

function proposal(
	controller: ContextEditingController,
	edits: readonly ContextEditOperation[],
	extra: Record<string, unknown> = {},
) {
	const status = controller.status;
	return controller.stageProposal({
		schemaVersion: 1,
		sessionId: status.sessionId,
		baseRevision: status.revision,
		baseDigest: status.effectiveDigest,
		edits,
		...extra,
	} as ContextEditProposal);
}

describe("durable context editing controller", () => {
	it("keeps stable source IDs, applies a staged edit, and resumes cold", () => {
		const directory = dir();
		const entries: SourceEntry[] = [
			{ id: "u-1", message: user("keep this instruction") },
			{ id: "a-1", message: assistant("old answer") },
		];
		const controller = new ContextEditingController({ sessionId: "s-1", directory });
		const first = controller.prepare(entries);
		const receipt = proposal(controller, [{ sourceId: "a-1", op: "replace_text", text: "new answer" }]);
		expect(receipt.accepted).toBe(true);
		const edited = controller.prepare(entries);
		expect(JSON.stringify(edited.messages)).toContain("new answer");
		expect(JSON.stringify(edited.messages)).not.toContain("old answer");
		expect(edited.revision).toBeGreaterThan(first.revision);
		expect(readFileSync(join(directory, "source.jsonl"), "utf8")).toContain('"id":"u-1"');
		controller.close();
		const reopened = new ContextEditingController({ sessionId: "s-1", directory });
		const resumed = reopened.prepare(entries);
		expect(resumed.cacheEpoch).not.toBe(edited.cacheEpoch);
		expect(JSON.stringify(resumed.messages)).toContain("new answer");
	});

	it("rejects divergence, stale proposals, role mutation, and unsupported multimodal source", () => {
		const directory = dir();
		const entries: SourceEntry[] = [
			{ id: "u-1", message: user("instruction") },
			{ id: "a-1", message: assistant("answer") },
		];
		const controller = new ContextEditingController({ sessionId: "s-2", directory });
		controller.prepare(entries);
		expect(() => controller.prepare([{ ...entries[0]!, message: user("changed") }, entries[1]!])).toThrowError(
			ContextEditingError,
		);
		const stale = controller.stageProposal({
			sessionId: "s-2",
			baseRevision: 0,
			baseDigest: "bad",
			edits: [],
		} as ContextEditProposal);
		expect(stale.accepted).toBe(false);
		const roleChange = proposal(controller, [{ sourceId: "a-1", op: "replace", message: user("not assistant") }]);
		expect(roleChange.accepted).toBe(true);
		const snapshot = controller.prepare(entries);
		expect(JSON.stringify(snapshot.messages)).toContain("answer");
		const image: SourceEntry[] = [
			{
				id: "image",
				message: { ...user("x"), content: [{ type: "image", data: "x", mimeType: "image/png" }] } as AgentMessage,
			},
		];
		expect(() => controller.prepare(image)).toThrowError(/multimodal/);
	});

	it("rejects reordered, deleted, and duplicate source frontiers and protects every user record", () => {
		const directory = dir();
		const entries: SourceEntry[] = [
			{ id: "u-1", message: user("first instruction") },
			{ id: "a-1", message: assistant("answer") },
			{ id: "u-2", message: user("correction instruction") },
		];
		const controller = new ContextEditingController({ sessionId: "s-frontier", directory });
		controller.prepare(entries);
		expect(() => controller.prepare(entries.slice(0, 2))).toThrowError(/backwards/);
		expect(() => controller.prepare([entries[1]!, entries[0]!, entries[2]!])).toThrowError(/diverges/);
		expect(() => controller.prepare([...entries, entries[2]!])).toThrowError(/duplicate/);
		const userEdit = proposal(controller, [{ sourceId: "u-1", op: "replace_text", text: "removed instruction" }]);
		expect(userEdit.accepted).toBe(true);
		const unchanged = controller.prepare(entries);
		expect(JSON.stringify(unchanged.messages)).toContain("first instruction");
	});

	it("protects a complete live multi-tool batch as one unit", () => {
		const directory = dir();
		const batch: SourceEntry[] = [
			{ id: "u", message: user("do both") },
			{ id: "call", message: multiToolAssistant() },
			{ id: "result-1", message: toolResult("call-1") },
			{ id: "result-2", message: toolResult("call-2") },
		];
		const controller = new ContextEditingController({ sessionId: "s-multi", directory });
		controller.prepare(batch);
		proposal(controller, [{ sourceIds: ["call", "result-1", "result-2"], op: "summary", text: "both completed" }]);
		const stillLive = controller.prepare(batch);
		expect(JSON.stringify(stillLive.messages)).toContain("call-2");
	});

	it("never splits a tool batch, but permits an old complete batch summary", () => {
		const directory = dir();
		const batch: SourceEntry[] = [
			{ id: "u", message: user("do it") },
			{ id: "call", message: toolAssistant() },
			{ id: "result", message: toolResult() },
		];
		const controller = new ContextEditingController({ sessionId: "s-3", directory });
		controller.prepare(batch);
		const rejected = proposal(controller, [{ sourceId: "call", op: "delete" }]);
		expect(rejected.accepted).toBe(true);
		const stillLive = controller.prepare(batch);
		expect(JSON.stringify(stillLive.messages)).toContain("toolCall");
		const withNext = [...batch, { id: "u-2", message: user("next") }];
		const summary = proposal(controller, [
			{ sourceIds: ["call", "result"], op: "summary", text: "Tool exchange completed." },
		]);
		expect(summary.accepted).toBe(true);
		const compacted = controller.prepare(withNext);
		expect(JSON.stringify(compacted.messages)).toContain("Tool exchange completed.");
		expect(JSON.stringify(compacted.messages)).not.toContain("toolCall");
	});

	it("rotates epoch without restoring deleted context and enforces conservative budget", () => {
		const directory = dir();
		const entries: SourceEntry[] = [
			{ id: "u", message: user("keep") },
			{ id: "a", message: assistant("remove") },
		];
		const controller = new ContextEditingController({
			sessionId: "s-4",
			directory,
			maxPromptTokens: 10,
			maxOutputTokens: 2,
		});
		expect(() => controller.prepare(entries)).toThrowError(/exceeds budget/);
		controller.close();
		const roomy = new ContextEditingController({ sessionId: "s-4", directory, maxPromptTokens: 1000 });
		roomy.prepare(entries);
		proposal(roomy, [{ sourceId: "a", op: "delete" }]);
		const removed = roomy.prepare(entries);
		const epoch = removed.cacheEpoch;
		const reset = roomy.hardReset("test reset");
		expect(reset.cacheEpoch).not.toBe(epoch);
		expect(JSON.stringify(reset.messages)).not.toContain("remove");
	});
	it("fails closed on a corrupt manifest and rejects a staged hard reset only at the boundary", () => {
		const directory = dir();
		const entries: SourceEntry[] = [
			{ id: "u", message: user("keep") },
			{ id: "a", message: assistant("answer") },
		];
		const controller = new ContextEditingController({ sessionId: "s-5", directory });
		controller.prepare(entries);
		const before = controller.status.cacheEpoch;
		const staged = proposal(controller, [], { hardReset: true, resetReason: "new lineage" });
		expect(staged.accepted).toBe(true);
		const reset = controller.prepare(entries);
		expect(reset.cacheEpoch).not.toBe(before);
		expect(JSON.stringify(reset.messages)).toContain("answer");
		controller.close();
		writeFileSync(join(directory, "manifest.json"), "broken\n");
		expect(() => new ContextEditingController({ sessionId: "s-5", directory })).toThrowError(
			/authoritative|manifest|JSON/,
		);
		const ownedDirectory = dir();
		const owner = new ContextEditingController({ sessionId: "owner", directory: ownedDirectory });
		expect(() => new ContextEditingController({ sessionId: "owner", directory: ownedDirectory })).toThrowError(
			/already owned/,
		);
		owner.close();
		expect(() => new ContextEditingController({ sessionId: "other", directory: ownedDirectory })).toThrowError(
			/different session|mismatch/,
		);
	});

	it("fails closed for incomplete tool batches and signed assistant text edits", () => {
		const directory = dir();
		const incomplete: SourceEntry[] = [
			{ id: "u", message: user("run") },
			{ id: "call", message: toolAssistant() },
		];
		const controller = new ContextEditingController({ sessionId: "s-incomplete", directory });
		expect(() => controller.prepare(incomplete)).toThrowError(/incomplete tool-call/);
		const signed = assistant("signed");
		(signed as { content: unknown[] }).content = [{ type: "text", text: "signed", textSignature: "opaque" }];
		const signedEntries: SourceEntry[] = [
			{ id: "u", message: user("keep") },
			{ id: "a", message: signed },
		];
		const signedController = new ContextEditingController({ sessionId: "s-signed", directory: dir() });
		signedController.prepare(signedEntries);
		const receipt = proposal(signedController, [{ sourceId: "a", op: "replace_text", text: "tamper" }]);
		expect(receipt.accepted).toBe(true);
		const snapshot = signedController.prepare(signedEntries);
		expect(JSON.stringify(snapshot.messages)).toContain("signed");
		expect(JSON.stringify(snapshot.messages)).not.toContain("tamper");
	});

	it("initializes and then enforces the serving fingerprint and validates numeric budget", () => {
		const directory = dir();
		const entries: SourceEntry[] = [{ id: "u", message: user("hello") }];
		const controller = new ContextEditingController({ sessionId: "s-fingerprint", directory });
		controller.prepare(entries, { servingFingerprint: "fp-1", fixedPromptTokens: 10, maxPromptTokens: 1000 });
		expect(() => controller.prepare(entries, { servingFingerprint: "fp-2" })).toThrowError(/fingerprint/);
		controller.hardReset("change", "fp-2");
		expect(controller.prepare(entries, { servingFingerprint: "fp-2" }).cacheEpoch).toBe(controller.status.cacheEpoch);
		expect(
			() => new ContextEditingController({ sessionId: "bad", directory: dir(), maxOutputTokens: Number.NaN }),
		).toThrowError(/finite/);
		expect(
			() => new ContextEditingController({ sessionId: "bad2", directory: dir(), maxPromptTokens: -1 }),
		).toThrowError(/non-negative/);
	});
});
