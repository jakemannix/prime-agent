import { existsSync, mkdtempSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
	ContextEditingController,
	type ContextEditProposal,
	type ContextStoreFaultPoint,
	type SourceEntry,
} from "../src/core/context-editing/index.js";

const directory = () => mkdtempSync(join(tmpdir(), "clm-durability-"));
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

function stageReplace(controller: ContextEditingController, sourceId: string, text: string): ContextEditProposal {
	const status = controller.status;
	const proposal: ContextEditProposal = {
		schemaVersion: 1,
		sessionId: status.sessionId,
		baseRevision: status.revision,
		baseDigest: status.effectiveDigest,
		edits: [{ op: "replace_text", sourceId, text }],
	};
	expect(controller.stageProposal(proposal)).toMatchObject({ accepted: true, phase: "staged" });
	return proposal;
}

function manifestGeneration(path: string): string {
	return (JSON.parse(readFileSync(join(path, "manifest.json"), "utf8")) as { generationId: string }).generationId;
}

describe("context editing durable transactions", () => {
	it("recovers a verified source tail and an unconsumed proposal after failure before pointer swap", () => {
		const path = directory();
		let armed = false;
		const injector = (point: ContextStoreFaultPoint) => {
			if (armed && point === "before_manifest_swap") throw new Error("before pointer");
		};
		const initial: SourceEntry[] = [
			{ id: "u-1", message: user("instruction") },
			{ id: "a-1", message: assistant("old answer") },
		];
		const appended = [...initial, { id: "u-2", message: user("next question") }];
		const controller = new ContextEditingController({
			sessionId: "before",
			directory: path,
			storeFaultInjector: injector,
		});
		controller.prepare(initial);
		const committedGeneration = manifestGeneration(path);
		stageReplace(controller, "a-1", "durable replacement");
		armed = true;
		expect(() => controller.prepare(appended)).toThrowError(/before pointer/);
		expect(manifestGeneration(path)).toBe(committedGeneration);
		expect(readFileSync(join(path, "source.jsonl"), "utf8")).toContain('"id":"u-2"');
		expect(existsSync(join(path, "proposal.json"))).toBe(true);
		expect(() => controller.prepare(appended)).toThrowError(/poisoned/);
		controller.close();

		const reopened = new ContextEditingController({ sessionId: "before", directory: path });
		expect(() => reopened.prepare(initial)).toThrowError(/backwards|behind/);
		const recovered = reopened.prepare(appended);
		expect(JSON.stringify(recovered.messages)).toContain("durable replacement");
		expect(JSON.stringify(recovered.messages).match(/next question/g)).toHaveLength(1);
		expect(reopened.status.stagingReceipt).toMatchObject({ accepted: true, phase: "committed" });
		expect(existsSync(join(path, "proposal.json"))).toBe(false);
	});

	it("uses a committed pointer after failure immediately after pointer swap without applying an edit twice", () => {
		const path = directory();
		let armed = false;
		const injector = (point: ContextStoreFaultPoint) => {
			if (armed && point === "after_manifest_swap") throw new Error("after pointer");
		};
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message: assistant("old") },
		];
		const controller = new ContextEditingController({
			sessionId: "after",
			directory: path,
			storeFaultInjector: injector,
		});
		controller.prepare(entries);
		const before = manifestGeneration(path);
		stageReplace(controller, "a", "committed once");
		armed = true;
		expect(() => controller.prepare(entries)).toThrowError(/after pointer/);
		expect(manifestGeneration(path)).not.toBe(before);
		expect(existsSync(join(path, "proposal.json"))).toBe(true);
		controller.close();

		const reopened = new ContextEditingController({ sessionId: "after", directory: path });
		const revision = reopened.status.revision;
		const recovered = reopened.prepare(entries);
		expect(recovered.revision).toBe(revision);
		expect(JSON.stringify(recovered.messages).match(/committed once/g)).toHaveLength(1);
		expect(reopened.status.stagingReceipt).toMatchObject({
			accepted: true,
			phase: "committed",
			revision,
		});
		expect(existsSync(join(path, "proposal.json"))).toBe(false);
	});

	it("does not consume a pending proposal on budget failure or hard reset", () => {
		const path = directory();
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message: assistant("old") },
		];
		const controller = new ContextEditingController({ sessionId: "budget", directory: path });
		controller.prepare(entries);
		stageReplace(controller, "a", "pending replacement");
		expect(() => controller.prepare(entries, { maxPromptTokens: 1 })).toThrowError(/exceeds budget/);
		expect(existsSync(join(path, "proposal.json"))).toBe(true);
		expect(controller.status.stagingReceipt).toMatchObject({ phase: "staged" });
		const epoch = controller.hardReset("budget recovery").cacheEpoch;
		expect(existsSync(join(path, "proposal.json"))).toBe(true);
		const committed = controller.prepare(entries, { maxPromptTokens: 10_000 });
		expect(committed.cacheEpoch).toBe(epoch);
		expect(JSON.stringify(committed.messages)).toContain("pending replacement");
		expect(controller.status.stagingReceipt).toMatchObject({ phase: "committed", revision: committed.revision });
	});

	it("records zero-op staged reset at the actual revision and keeps the guide static", () => {
		const path = directory();
		const controller = new ContextEditingController({ sessionId: "zero", directory: path });
		expect(existsSync(controller.status.protocolPath)).toBe(true);
		const guide = readFileSync(controller.status.protocolPath, "utf8");
		const entries: SourceEntry[] = [{ id: "u", message: user("instruction") }];
		const first = controller.prepare(entries);
		const status = controller.status;
		controller.stageProposal({
			schemaVersion: 1,
			sessionId: status.sessionId,
			baseRevision: status.revision,
			baseDigest: status.effectiveDigest,
			edits: [],
			hardReset: true,
			resetReason: "new cache lineage",
		});
		const reset = controller.prepare(entries);
		expect(reset.revision).toBe(first.revision);
		expect(reset.digest).toBe(first.digest);
		expect(reset.cacheEpoch).not.toBe(first.cacheEpoch);
		const firstReceipt = controller.status.stagingReceipt;
		expect(firstReceipt).toMatchObject({ phase: "committed", revision: first.revision });
		expect(firstReceipt?.proposalId).toMatch(/^proposal-/);

		const sameBase = controller.status;
		controller.stageProposal({
			schemaVersion: 1,
			sessionId: sameBase.sessionId,
			baseRevision: sameBase.revision,
			baseDigest: sameBase.effectiveDigest,
			edits: [],
			hardReset: true,
			resetReason: "new cache lineage",
		});
		const secondReset = controller.prepare(entries);
		expect(secondReset.revision).toBe(reset.revision);
		expect(secondReset.digest).toBe(reset.digest);
		expect(secondReset.cacheEpoch).not.toBe(reset.cacheEpoch);
		expect(controller.status.stagingReceipt?.proposalId).not.toBe(firstReceipt?.proposalId);
		expect(readFileSync(controller.status.protocolPath, "utf8")).toBe(guide);
	});

	it("excludes concurrent owners, safely ignores replacement locks on close, and recovers a dead lock", () => {
		const path = directory();
		const controller = new ContextEditingController({ sessionId: "locks", directory: path });
		expect(() => new ContextEditingController({ sessionId: "locks", directory: path })).toThrowError(/already owned/);
		unlinkSync(join(path, "owner.lock"));
		const replacement = `${JSON.stringify({ pid: process.pid, sessionId: "locks", token: "replacement" })}\n`;
		writeFileSync(join(path, "owner.lock"), replacement);
		const lockStatus = controller.status;
		expect(() =>
			controller.stageProposal({
				schemaVersion: 1,
				sessionId: lockStatus.sessionId,
				baseRevision: lockStatus.revision,
				baseDigest: lockStatus.effectiveDigest,
				edits: [],
			}),
		).toThrowError(/ownership lock token/);
		expect(() => controller.prepare([])).toThrowError(/poisoned/);
		controller.close();
		expect(() => controller.prepare([])).toThrowError(/closed/);
		expect(readFileSync(join(path, "owner.lock"), "utf8")).toBe(replacement);
		unlinkSync(join(path, "owner.lock"));
		writeFileSync(
			join(path, "owner.lock"),
			`${JSON.stringify({ pid: 2_147_483_647, sessionId: "locks", token: "dead" })}\n`,
		);
		const recovered = new ContextEditingController({ sessionId: "locks", directory: path });
		expect(recovered.status.sessionId).toBe("locks");
		recovered.close();
	});

	it("fails closed for corrupt commit pointers and immutable generation digests while ignoring informational state", () => {
		const path = directory();
		const controller = new ContextEditingController({ sessionId: "corrupt", directory: path });
		controller.prepare([{ id: "u", message: user("instruction") }]);
		controller.close();
		writeFileSync(join(path, "state.json"), "not authoritative\n");
		const ignoresState = new ContextEditingController({ sessionId: "corrupt", directory: path });
		ignoresState.close();
		const manifestPath = join(path, "manifest.json");
		const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { generationDigest: string };
		writeFileSync(manifestPath, `${JSON.stringify({ ...manifest, generationDigest: "bad" })}\n`);
		expect(() => new ContextEditingController({ sessionId: "corrupt", directory: path })).toThrowError(/digest/);
	});

	it("uses JSON round-trip semantics for undefined message properties", () => {
		const path = directory();
		const message = assistant("answer") as AgentMessage & { errorMessage?: string; responseId?: string };
		message.errorMessage = undefined;
		message.responseId = undefined;
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message },
		];
		const controller = new ContextEditingController({ sessionId: "undefined", directory: path });
		controller.prepare(entries);
		expect(() => controller.prepare(entries)).not.toThrow();
		controller.close();
		const reopened = new ContextEditingController({ sessionId: "undefined", directory: path });
		expect(() => reopened.prepare(entries)).not.toThrow();
	});

	it("never clears a newer proposal written after snapshot pinning", () => {
		const path = directory();
		let armed = false;
		let nextProposal = "";
		const injector = (point: ContextStoreFaultPoint) => {
			if (armed && point === "before_manifest_swap") {
				writeFileSync(join(path, "proposal.json"), nextProposal);
				armed = false;
			}
		};
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message: assistant("old") },
		];
		const controller = new ContextEditingController({
			sessionId: "race",
			directory: path,
			storeFaultInjector: injector,
		});
		controller.prepare(entries);
		const base = controller.status;
		stageReplace(controller, "a", "first replacement");
		nextProposal = `${JSON.stringify({
			schemaVersion: 1,
			sessionId: base.sessionId,
			baseRevision: base.revision,
			baseDigest: base.effectiveDigest,
			edits: [{ op: "replace_text", sourceId: "a", text: "newer replacement" }],
		})}\n`;
		armed = true;
		const first = controller.prepare(entries);
		expect(JSON.stringify(first.messages)).toContain("first replacement");
		expect(existsSync(join(path, "proposal.json"))).toBe(true);
		const next = controller.prepare(entries);
		expect(JSON.stringify(next.messages)).toContain("first replacement");
		expect(JSON.stringify(next.messages)).not.toContain("newer replacement");
		expect(controller.status.stagingReceipt).toMatchObject({ accepted: false, phase: "rejected" });
		expect(existsSync(join(path, "proposal.json"))).toBe(false);
	});

	it("durably rejects malformed summaries and broad signed-text mutation", () => {
		const path = directory();
		const signed = assistant("signed") as AgentMessage & { content: Array<Record<string, unknown>> };
		signed.content = [{ type: "text", text: "signed", textSignature: "opaque" }];
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message: signed },
		];
		const controller = new ContextEditingController({ sessionId: "malformed", directory: path });
		controller.prepare(entries);
		const base = controller.status;
		writeFileSync(
			join(path, "proposal.json"),
			`${JSON.stringify({
				schemaVersion: 1,
				sessionId: base.sessionId,
				baseRevision: base.revision,
				baseDigest: base.effectiveDigest,
				edits: [{ op: "summary", sourceIds: ["a"], text: { malformed: true } }],
			})}\n`,
		);
		const malformed = controller.prepare(entries);
		expect(JSON.stringify(malformed.messages)).toContain("signed");
		expect(controller.status.stagingReceipt).toMatchObject({ accepted: false, phase: "rejected" });

		const replacement = assistant("tampered") as AgentMessage & { content: Array<Record<string, unknown>> };
		replacement.content = [{ type: "text", text: "tampered", textSignature: "forged" }];
		const status = controller.status;
		controller.stageProposal({
			schemaVersion: 1,
			sessionId: status.sessionId,
			baseRevision: status.revision,
			baseDigest: status.effectiveDigest,
			edits: [{ op: "replace", sourceId: "a", message: replacement }],
		});
		const protectedSnapshot = controller.prepare(entries);
		expect(JSON.stringify(protectedSnapshot.messages)).toContain("opaque");
		expect(JSON.stringify(protectedSnapshot.messages)).not.toContain("forged");
		expect(controller.status.stagingReceipt).toMatchObject({ accepted: false, phase: "rejected" });
	});

	it("poisons and removes a proposal after an ambiguous staging rename failure", () => {
		const path = directory();
		let armed = false;
		const injector = (point: ContextStoreFaultPoint) => {
			if (armed && point === "after_proposal_rename") throw new Error("proposal rename ambiguity");
		};
		const entries: SourceEntry[] = [
			{ id: "u", message: user("instruction") },
			{ id: "a", message: assistant("old answer") },
		];
		const controller = new ContextEditingController({
			sessionId: "ambiguous-stage",
			directory: path,
			storeFaultInjector: injector,
		});
		controller.prepare(entries);
		const status = controller.status;
		armed = true;
		expect(() =>
			controller.stageProposal({
				schemaVersion: 1,
				sessionId: status.sessionId,
				baseRevision: status.revision,
				baseDigest: status.effectiveDigest,
				edits: [{ op: "replace_text", sourceId: "a", text: "must not apply" }],
			}),
		).toThrowError(/proposal rename ambiguity/);
		expect(existsSync(join(path, "proposal.json"))).toBe(false);
		expect(existsSync(join(path, "proposal.write-in-progress.json"))).toBe(false);
		expect(() => controller.prepare(entries)).toThrowError(/poisoned/);
		controller.close();

		const reopened = new ContextEditingController({ sessionId: "ambiguous-stage", directory: path });
		const snapshot = reopened.prepare(entries);
		expect(JSON.stringify(snapshot.messages)).toContain("old answer");
		expect(JSON.stringify(snapshot.messages)).not.toContain("must not apply");
	});

	it("makes close sticky after a replacement controller acquires ownership", () => {
		const path = directory();
		const entries: SourceEntry[] = [{ id: "u", message: user("instruction") }];
		const first = new ContextEditingController({ sessionId: "sticky-close", directory: path });
		first.prepare(entries);
		first.close();
		const second = new ContextEditingController({ sessionId: "sticky-close", directory: path });
		const pointer = readFileSync(join(path, "manifest.json"), "utf8");
		const staleProposal: ContextEditProposal = {
			schemaVersion: 1,
			sessionId: "sticky-close",
			baseRevision: first.status.revision,
			baseDigest: first.status.effectiveDigest,
			edits: [],
		};
		expect(() => first.prepare(entries)).toThrowError(/closed/);
		expect(() => first.hardReset("stale reset")).toThrowError(/closed/);
		expect(() => first.stageProposal(staleProposal)).toThrowError(/closed/);
		expect(readFileSync(join(path, "manifest.json"), "utf8")).toBe(pointer);
		expect(existsSync(join(path, "proposal.json"))).toBe(false);
		second.close();
	});
});
