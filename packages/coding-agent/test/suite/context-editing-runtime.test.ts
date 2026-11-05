import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	fauxAssistantMessage,
	fauxToolCall,
	type Model,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import type { AgentSession } from "../../src/core/agent-session.js";
import { AuthStorage } from "../../src/core/auth-storage.js";
import type { ExtensionFactory } from "../../src/core/extensions/index.js";
import { ModelRegistry } from "../../src/core/model-registry.js";
import { createAgentSession } from "../../src/core/sdk.js";
import { SessionManager } from "../../src/core/session-manager.js";
import { SettingsManager } from "../../src/core/settings-manager.js";
import { startSideQuestion } from "../../src/core/side-question.js";
import { createTestExtensionsResult, createTestResourceLoader } from "../utilities.js";
import { getMessageText } from "./harness.js";

interface RuntimeFixture {
	session: AgentSession;
	sessionManager: SessionManager;
	faux: FauxProviderRegistration;
	tempDir: string;
}

const fixtures: RuntimeFixture[] = [];

async function createFixture(
	options: {
		tempDir?: string;
		sessionFile?: string;
		contextEditing?: boolean;
		contextWindow?: number;
		modelMaxTokens?: number;
		contextMaxOutputTokens?: number;
		systemPrompt?: string;
		sglang?: { baseUrl: string; model: string; mode: "exact" | "scr"; servingFingerprint: string };
		extensionFactories?: ExtensionFactory[];
	} = {},
): Promise<RuntimeFixture> {
	const tempDir =
		options.tempDir ?? join(tmpdir(), `prime-context-runtime-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	mkdirSync(tempDir, { recursive: true });
	const faux = registerFauxProvider({
		models: [
			{
				id: "context-faux",
				contextWindow: options.contextWindow,
				maxTokens: options.modelMaxTokens,
			},
		],
	});
	const model = faux.getModel() as Model<string>;
	const authStorage = AuthStorage.inMemory();
	authStorage.setRuntimeApiKey(model.provider, "faux-key");
	const modelRegistry = ModelRegistry.inMemory(authStorage);
	modelRegistry.registerProvider(model.provider, {
		api: faux.api,
		baseUrl: model.baseUrl,
		apiKey: "faux-key",
		models: [model],
	});
	const sessionManager = options.sessionFile
		? SessionManager.open(options.sessionFile, join(tempDir, "sessions"))
		: SessionManager.create(tempDir, join(tempDir, "sessions"));
	const extensionsResult = options.extensionFactories
		? await createTestExtensionsResult(options.extensionFactories, tempDir)
		: undefined;
	const baseResourceLoader = createTestResourceLoader(extensionsResult ? { extensionsResult } : undefined);
	const resourceLoader = options.systemPrompt
		? { ...baseResourceLoader, getSystemPrompt: () => options.systemPrompt }
		: baseResourceLoader;
	let created: Awaited<ReturnType<typeof createAgentSession>>;
	try {
		created = await createAgentSession({
			cwd: tempDir,
			model,
			authStorage,
			modelRegistry,
			settingsManager: SettingsManager.inMemory(),
			sessionManager,
			resourceLoader,
			contextEditing: options.contextEditing
				? {
						directory: join(tempDir, "context"),
						fixedPromptTokens: 64,
						maxOutputTokens: options.contextMaxOutputTokens ?? 64,
						sglang: options.sglang,
					}
				: undefined,
		});
	} catch (error) {
		faux.unregister();
		rmSync(tempDir, { recursive: true, force: true });
		throw error;
	}
	const fixture = { session: created.session, sessionManager, faux, tempDir };
	fixtures.push(fixture);
	return fixture;
}

afterEach(async () => {
	while (fixtures.length > 0) {
		const fixture = fixtures.pop()!;
		await fixture.session.disposeAsync();
		fixture.faux.unregister();
		rmSync(fixture.tempDir, { recursive: true, force: true });
	}
});

describe("context editing runtime integration", () => {
	it("leaves disabled SDK sessions unchanged and creates no controller artifacts", async () => {
		const fixture = await createFixture();
		let providerTexts: string[] = [];
		fixture.faux.setResponses([
			(context) => {
				providerTexts = context.messages.map(getMessageText);
				return fauxAssistantMessage("ordinary response");
			},
		]);

		await fixture.session.prompt("ordinary input");

		expect(providerTexts).toEqual(["ordinary input"]);
		expect(fixture.session.contextEditingEnabled).toBe(false);
		expect(fixture.session.getContextEditingStatus()).toBeUndefined();
		expect(existsSync(join(fixture.tempDir, "context"))).toBe(false);
	});

	it("drains queued message persistence before starting the provider request", async () => {
		let releaseMessageEnd = () => {};
		const messageEndGate = new Promise<void>((resolve) => {
			releaseMessageEnd = resolve;
		});
		let markMessageEndReached = () => {};
		const messageEndReached = new Promise<void>((resolve) => {
			markMessageEndReached = resolve;
		});
		let held = false;
		const fixture = await createFixture({
			contextEditing: true,
			extensionFactories: [
				(pi) => {
					pi.on("message_end", async (event) => {
						if (!held && event.message.role === "user") {
							held = true;
							markMessageEndReached();
							await messageEndGate;
						}
					});
				},
			],
		});
		let journalAtProvider = "";
		fixture.faux.setResponses([
			() => {
				journalAtProvider = readFileSync(fixture.sessionManager.getSessionFile()!, "utf8");
				return fauxAssistantMessage("after barrier");
			},
		]);

		const prompt = fixture.session.prompt("durable before dispatch");
		await messageEndReached;
		await new Promise<void>((resolve) => setImmediate(resolve));
		expect(fixture.faux.state.callCount).toBe(0);
		expect(
			fixture.sessionManager
				.getBranch()
				.filter((entry) => entry.type === "message")
				.map((entry) => (entry.type === "message" ? getMessageText(entry.message) : "")),
		).toEqual([]);

		releaseMessageEnd();
		await prompt;
		expect(fixture.faux.state.callCount).toBe(1);
		expect(journalAtProvider).toContain("durable before dispatch");
		expect(fixture.session.getContextEditingStatus()?.sourceCount).toBe(1);
	});

	it("sticks a user persistence failure even when later agent events recover the queue", async () => {
		const fixture = await createFixture({ contextEditing: true });
		const sessionManager = fixture.sessionManager as SessionManager & {
			appendMessage: SessionManager["appendMessage"];
		};
		const appendMessage = sessionManager.appendMessage.bind(sessionManager);
		let rejectedUser = false;
		sessionManager.appendMessage = ((message: Parameters<SessionManager["appendMessage"]>[0]) => {
			if (!rejectedUser && message.role === "user") {
				rejectedUser = true;
				throw new Error("induced user journal failure");
			}
			return appendMessage(message);
		}) as SessionManager["appendMessage"];
		fixture.faux.setResponses([fauxAssistantMessage("must not dispatch")]);

		await fixture.session.prompt("must be durable before dispatch").catch(() => {});

		expect(rejectedUser).toBe(true);
		expect(fixture.faux.state.callCount).toBe(0);
		expect(fixture.session.agent.state.errorMessage).toContain(
			"Context editing blocked because durable session persistence failed",
		);
	});

	it("waits for a complete multi-tool batch before applying an edit staged by a live tool", async () => {
		let fixture: RuntimeFixture;
		let oldAssistantId = "";
		let releaseSlow = () => {};
		const slowGate = new Promise<void>((resolve) => {
			releaseSlow = resolve;
		});
		let markSlowStarted = () => {};
		const slowStarted = new Promise<void>((resolve) => {
			markSlowStarted = resolve;
		});
		let markProposalStaged = () => {};
		const proposalStaged = new Promise<void>((resolve) => {
			markProposalStaged = resolve;
		});
		fixture = await createFixture({
			contextEditing: true,
			extensionFactories: [
				(pi) => {
					pi.registerTool({
						name: "stage_old_edit",
						label: "Stage old edit",
						description: "Stage an edit of old eligible history",
						parameters: Type.Object({}),
						execute: async () => {
							const status = fixture.session.getContextEditingStatus()!;
							writeFileSync(
								status.stagingPath,
								`${JSON.stringify({
									schemaVersion: 1,
									sessionId: status.sessionId,
									baseRevision: status.revision,
									baseDigest: status.effectiveDigest,
									edits: [{ op: "delete", sourceId: oldAssistantId }],
								})}\n`,
							);
							markProposalStaged();
							return { content: [{ type: "text", text: "staged" }], details: {} };
						},
					});
					pi.registerTool({
						name: "slow_live_tail",
						label: "Slow live tail",
						description: "Delay the second result",
						parameters: Type.Object({}),
						execute: async () => {
							markSlowStarted();
							await slowGate;
							return { content: [{ type: "text", text: "slow complete" }], details: {} };
						},
					});
				},
			],
		});
		fixture.faux.setResponses([fauxAssistantMessage("historical assistant to remove")]);
		await fixture.session.prompt("historical protected user");
		const oldAssistant = fixture.sessionManager
			.getBranch()
			.find(
				(entry) =>
					entry.type === "message" &&
					entry.message.role === "assistant" &&
					getMessageText(entry.message) === "historical assistant to remove",
			);
		expect(oldAssistant?.type).toBe("message");
		oldAssistantId = oldAssistant!.id;

		const stageCall = fauxToolCall("stage_old_edit", {});
		const slowCall = fauxToolCall("slow_live_tail", {});
		let continuationMessages: unknown[] = [];
		fixture.faux.setResponses([
			fauxAssistantMessage([stageCall, slowCall], { stopReason: "toolUse" }),
			(context) => {
				continuationMessages = context.messages;
				return fauxAssistantMessage("continuation complete");
			},
		]);
		const turn = fixture.session.prompt("run both live tools");
		await Promise.all([proposalStaged, slowStarted]);
		await new Promise<void>((resolve) => setImmediate(resolve));
		// The baseline and tool-call requests have dispatched, but continuation is blocked on the full result batch.
		expect(fixture.faux.state.callCount).toBe(2);
		releaseSlow();
		await turn;

		const serialized = JSON.stringify(continuationMessages);
		expect(serialized).toContain("historical protected user");
		expect(serialized).not.toContain("historical assistant to remove");
		expect(serialized).toContain("run both live tools");
		expect(serialized).toContain(stageCall.id);
		expect(serialized).toContain(slowCall.id);
		expect(serialized).toContain("staged");
		expect(serialized).toContain("slow complete");
		expect(
			fixture.sessionManager
				.getBranch()
				.filter((entry) => entry.type === "message" && entry.message.role === "toolResult"),
		).toHaveLength(2);
	});

	it("commits an edit at the next boundary, reopens cold, and hard-resets without restoring removed text", async () => {
		const first = await createFixture({ contextEditing: true });
		first.faux.setResponses([fauxAssistantMessage("remove this history")]);
		await first.session.prompt("first user instruction");
		const status = first.session.getContextEditingStatus()!;
		const assistantEntry = first.sessionManager
			.getBranch()
			.find((entry) => entry.type === "message" && entry.message.role === "assistant");
		expect(assistantEntry?.type).toBe("message");
		writeFileSync(
			status.stagingPath,
			`${JSON.stringify({
				schemaVersion: 1,
				sessionId: status.sessionId,
				baseRevision: status.revision,
				baseDigest: status.effectiveDigest,
				edits: [{ op: "delete", sourceId: assistantEntry!.id }],
			})}\n`,
		);
		let secondContext = "";
		first.faux.setResponses([
			(context) => {
				secondContext = context.messages.map(getMessageText).join("|");
				return fauxAssistantMessage("second response");
			},
		]);
		await first.session.prompt("second user instruction");
		expect(secondContext).toContain("first user instruction");
		expect(secondContext).toContain("second user instruction");
		expect(secondContext).not.toContain("remove this history");

		const sessionFile = first.sessionManager.getSessionFile()!;
		const rawPrefix = readFileSync(sessionFile, "utf8");
		expect(rawPrefix).toContain("remove this history");
		const epochBeforeReopen = first.session.getContextEditingStatus()!.cacheEpoch;
		await first.session.disposeAsync();
		first.faux.unregister();
		fixtures.splice(fixtures.indexOf(first), 1);

		const reopened = await createFixture({ tempDir: first.tempDir, sessionFile, contextEditing: true });
		const epochAfterReopen = reopened.session.getContextEditingStatus()!.cacheEpoch;
		expect(epochAfterReopen).not.toBe(epochBeforeReopen);
		expect(readFileSync(sessionFile, "utf8").startsWith(rawPrefix)).toBe(true);
		const reset = await reopened.session.hardResetContextEditing("test reset");
		expect(reset.cacheEpoch).not.toBe(epochAfterReopen);
		let reopenedContext = "";
		reopened.faux.setResponses([
			(context) => {
				reopenedContext = context.messages.map(getMessageText).join("|");
				return fauxAssistantMessage("third response");
			},
		]);
		await reopened.session.prompt("third user instruction");
		expect(reopenedContext).not.toContain("remove this history");
		expect(readFileSync(sessionFile, "utf8").startsWith(rawPrefix)).toBe(true);
	});

	it("reports a rejected protected edit once, blocks that dispatch, then continues last-good context", async () => {
		const fixture = await createFixture({ contextEditing: true });
		fixture.faux.setResponses([fauxAssistantMessage("baseline assistant")]);
		await fixture.session.prompt("protected user instruction");
		const userEntry = fixture.sessionManager
			.getBranch()
			.find((entry) => entry.type === "message" && entry.message.role === "user");
		expect(userEntry?.type).toBe("message");
		const status = fixture.session.getContextEditingStatus()!;
		writeFileSync(
			status.stagingPath,
			`${JSON.stringify({
				schemaVersion: 1,
				sessionId: status.sessionId,
				baseRevision: status.revision,
				baseDigest: status.effectiveDigest,
				edits: [{ op: "delete", sourceId: userEntry!.id }],
			})}\n`,
		);
		fixture.faux.setResponses([fauxAssistantMessage("must not dispatch")]);

		await fixture.session.prompt("request that reports rejection");

		expect(fixture.faux.state.callCount).toBe(1);
		expect(fixture.session.agent.state.errorMessage).toContain("Context edit proposal rejected");
		expect(fixture.session.agent.state.errorMessage).toContain("last-good context was retained");

		let resumedContext = "";
		fixture.faux.setResponses([
			(context) => {
				resumedContext = context.messages.map(getMessageText).join("|");
				return fauxAssistantMessage("continued safely");
			},
		]);
		await fixture.session.prompt("clean continuation request");
		expect(fixture.faux.state.callCount).toBe(2);
		expect(resumedContext).toContain("protected user instruction");
		expect(resumedContext).toContain("clean continuation request");
	});

	it("releases controller ownership when a pre-AgentSession journal append fails", async () => {
		const tempDir = join(tmpdir(), `prime-context-construction-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
		const faux = registerFauxProvider({ models: [{ id: "construction-faux" }] });
		const model = faux.getModel() as Model<string>;
		const authStorage = AuthStorage.inMemory();
		authStorage.setRuntimeApiKey(model.provider, "faux-key");
		const modelRegistry = ModelRegistry.inMemory(authStorage);
		modelRegistry.registerProvider(model.provider, {
			api: faux.api,
			baseUrl: model.baseUrl,
			apiKey: "faux-key",
			models: [model],
		});
		const sessionManager = SessionManager.create(tempDir, join(tempDir, "sessions"));
		const settingsManager = SettingsManager.inMemory();
		const goodLoader = createTestResourceLoader();
		const common = {
			cwd: tempDir,
			model,
			authStorage,
			modelRegistry,
			settingsManager,
			sessionManager,
			contextEditing: { directory: join(tempDir, "context"), fixedPromptTokens: 64, maxOutputTokens: 64 },
		};

		const appendModelChange = sessionManager.appendModelChange.bind(sessionManager);
		sessionManager.appendModelChange = (() => {
			throw new Error("induced model journal failure");
		}) as SessionManager["appendModelChange"];
		await expect(createAgentSession({ ...common, resourceLoader: goodLoader })).rejects.toThrow(
			"induced model journal failure",
		);
		sessionManager.appendModelChange = appendModelChange;
		const reopened = await createAgentSession({ ...common, resourceLoader: goodLoader });
		fixtures.push({ session: reopened.session, sessionManager, faux, tempDir });
		expect(reopened.session.contextEditingEnabled).toBe(true);
	});

	it("rejects SGLang configuration before a provider without payload hooks can dispatch", async () => {
		await expect(
			createFixture({
				contextEditing: true,
				sglang: {
					baseUrl: "http://localhost:0",
					model: "context-faux",
					mode: "exact",
					servingFingerprint: "test-server-v1",
				},
			}),
		).rejects.toThrow("requires the openai-completions API");
	});

	it("measures fixed prompt overhead and bounds actual output tokens", async () => {
		const oversized = await createFixture({
			contextEditing: true,
			contextWindow: 2_000,
			modelMaxTokens: 512,
			contextMaxOutputTokens: 32,
			systemPrompt: "x".repeat(4_000),
		});
		oversized.faux.setResponses([fauxAssistantMessage("must not dispatch")]);
		await oversized.session.prompt("budget check");
		expect(oversized.faux.state.callCount).toBe(0);
		expect(oversized.session.agent.state.errorMessage).toContain("exceeds budget");

		const bounded = await createFixture({
			contextEditing: true,
			contextMaxOutputTokens: 37,
			modelMaxTokens: 512,
		});
		let observedMaxTokens: number | undefined;
		bounded.faux.setResponses([
			(_context, streamOptions) => {
				observedMaxTokens = streamOptions?.maxTokens;
				return fauxAssistantMessage("bounded");
			},
		]);
		await bounded.session.prompt("output bound");
		expect(observedMaxTokens).toBe(37);
	});

	it("blocks fail-open in-place payload mutation and clears the rejected request snapshot", async () => {
		let mutateFirstPayload = true;
		const fixture = await createFixture({
			contextEditing: true,
			extensionFactories: [
				(pi) => {
					pi.on("before_provider_request", (event) => {
						if (!mutateFirstPayload) return undefined;
						mutateFirstPayload = false;
						(event.payload as Record<string, unknown>).messages = [];
						throw new Error("extension failure after mutation");
					});
				},
			],
		});
		const agentHooks = fixture.session.agent as unknown as {
			transformContext(messages: typeof fixture.session.messages): Promise<unknown>;
			onPayload(payload: unknown, model: Model<string>): Promise<unknown>;
		};
		await agentHooks.transformContext(fixture.session.messages);
		await expect(
			agentHooks.onPayload({ model: "context-faux", messages: [{ role: "user" }] }, fixture.session.model!),
		).rejects.toThrow("mutated the pinned context-editing payload");

		await agentHooks.transformContext(fixture.session.messages);
		await expect(
			agentHooks.onPayload({ model: "context-faux", messages: [{ role: "user" }] }, fixture.session.model!),
		).resolves.toEqual({ model: "context-faux", messages: [{ role: "user" }] });
	});

	it("guards incompatible auxiliary and history-mutating paths without changing settings", async () => {
		const fixture = await createFixture({ contextEditing: true });
		const compactionSetting = fixture.session.settingsManager.getCompactionEnabled();
		await expect(fixture.session.compact()).rejects.toThrow("unavailable while context editing");
		await expect(fixture.session.refine()).rejects.toThrow("unavailable while context editing");
		await expect(fixture.session.navigateTree("missing")).rejects.toThrow("unavailable while context editing");
		expect(() => startSideQuestion(fixture.session.agent, "side", "question", () => {})).toThrow(
			"unavailable while context editing",
		);
		expect(fixture.session.settingsManager.getCompactionEnabled()).toBe(compactionSetting);
	});
});
