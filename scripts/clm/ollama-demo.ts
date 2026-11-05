import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { join, resolve } from "node:path";
import type { Model } from "../../packages/ai/src/types.js";
import { AuthStorage } from "../../packages/coding-agent/src/core/auth-storage.js";
import { createAgentSession } from "../../packages/coding-agent/src/core/sdk.js";
import type { AgentSession } from "../../packages/coding-agent/src/core/agent-session.js";
import { ModelRegistry } from "../../packages/coding-agent/src/core/model-registry.js";
import { DefaultResourceLoader } from "../../packages/coding-agent/src/core/resource-loader.js";
import { SessionManager } from "../../packages/coding-agent/src/core/session-manager.js";
import { SettingsManager } from "../../packages/coding-agent/src/core/settings-manager.js";

interface CapturedPayload { messages?: unknown; [key: string]: unknown }
interface TurnMetric { label: string; latencyMs: number; response: string; usage?: unknown }

function option(name: string, fallback: string): string {
	const index = process.argv.indexOf(name);
	return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function messageText(message: unknown): string {
	if (!message || typeof message !== "object") return "";
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => {
			if (!block || typeof block !== "object") return "";
			const item = block as { type?: unknown; text?: unknown; thinking?: unknown };
			return item.type === "thinking" ? String(item.thinking ?? "") : String(item.text ?? "");
		})
		.join("");
}

function branchMessages(session: { sessionManager: SessionManager }): Array<{ id: string; role: string; text: string; usage?: unknown; errorMessage?: string; stopReason?: string }> {
	return session.sessionManager
		.getBranch()
		.filter((entry) => entry.type === "message")
		.map((entry) => {
			const message = entry.message as { usage?: unknown; errorMessage?: string; stopReason?: string };
			return { id: entry.id, role: entry.message.role, text: messageText(entry.message), usage: message.usage, errorMessage: message.errorMessage, stopReason: message.stopReason };
		});
}

async function runTurn(session: AgentSession, label: string, prompt: string, metrics: TurnMetric[]): Promise<void> {
	const started = performance.now();
	await session.promptAndWait(prompt);
	const assistant = [...branchMessages(session)].reverse().find((entry) => entry.role === "assistant");
	if (!assistant) throw new Error(`SDK turn '${label}' returned no assistant response`);
	if (assistant.errorMessage || assistant.stopReason === "error" || assistant.stopReason === "aborted") {
		throw new Error(`SDK turn '${label}' failed: ${assistant.errorMessage ?? assistant.stopReason}`);
	}
	if (!assistant.text.trim()) throw new Error(`SDK turn '${label}' returned empty assistant response`);
	metrics.push({ label, latencyMs: performance.now() - started, response: assistant.text, usage: assistant.usage });
}

function payloadMessages(payload: CapturedPayload): unknown[] {
	return Array.isArray(payload.messages) ? payload.messages : [];
}

async function writeJsonl(path: string, values: readonly unknown[]): Promise<void> {
	await writeFile(path, values.map((value) => JSON.stringify(value)).join("\n") + (values.length ? "\n" : ""));
}

async function assertReady(baseUrl: string, model: string): Promise<void> {
	const response = await fetch(new URL("/api/tags", baseUrl));
	if (!response.ok) throw new Error(`Ollama is not ready (HTTP ${response.status}); start it before running this explicit demo`);
	const body = (await response.json()) as { models?: Array<{ name?: unknown }> };
	if (!body.models?.some((entry) => entry.name === model)) {
		throw new Error(`Ollama model '${model}' is not installed; pass --model for an installed local model`);
	}
}

async function makeLoader(cwd: string, agentDir: string, settingsManager: SettingsManager, captures: CapturedPayload[]): Promise<DefaultResourceLoader> {
	const loader = new DefaultResourceLoader({
		cwd,
		agentDir,
		settingsManager,
		noExtensions: true,
		noSkills: true,
		noContextFiles: true,
		extensionFactories: [
			(api) => {
				api.on("before_provider_request", (event) => {
					if (event.payload && typeof event.payload === "object" && !Array.isArray(event.payload)) {
						captures.push(structuredClone(event.payload) as CapturedPayload);
					}
				});
			},
		],
	});
	await loader.reload();
	return loader;
}

async function runResumeMode(
	model: Model<"openai-completions">,
	outDir: string,
	cwd: string,
	agentDir: string,
	contextOptions: { directory: string; maxPromptTokens: number; fixedPromptTokens: number; maxOutputTokens: number },
): Promise<void> {
	const prior = JSON.parse(await readFile(join(outDir, "report.json"), "utf8")) as {
		sessionFile?: string;
		contextDirectory?: string;
		acceptedEdit?: { replacement?: string; sourceId?: string; originalTextSha256?: string };
		hardReset?: { epoch?: string };
	};
	if (!prior.sessionFile || !prior.contextDirectory || !prior.acceptedEdit?.replacement || !prior.acceptedEdit.sourceId || !prior.acceptedEdit.originalTextSha256 || !prior.hardReset?.epoch) {
		throw new Error("--resume-existing requires a completed report.json from the main demo");
	}
	const captures: CapturedPayload[] = [];
	const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
	const modelRegistry = ModelRegistry.create(authStorage, join(agentDir, "models.json"));
	modelRegistry.registerProvider("ollama", { apiKey: "ollama" });
	const settings = SettingsManager.create(cwd, agentDir);
	const loader = await makeLoader(cwd, agentDir, settings, captures);
	const manager = SessionManager.open(prior.sessionFile, join(outDir, "sessions"), cwd);
	const reopened = (await createAgentSession({
		cwd,
		agentDir,
		authStorage,
		modelRegistry,
		settingsManager: settings,
		resourceLoader: loader,
		sessionManager: manager,
		model,
		thinkingLevel: "off",
		noTools: "all",
		includeGoals: false,
		contextEditing: { ...contextOptions, directory: prior.contextDirectory },
	})).session;
	try {
		const status = reopened.getContextEditingStatus();
		if (!status || status.cacheEpoch === prior.hardReset.epoch) throw new Error("Fresh-process resume did not rotate cache epoch");
		const source = (await readFile(join(prior.contextDirectory, "source.jsonl"), "utf8"))
			.split("\n")
			.filter(Boolean)
			.map((line) => JSON.parse(line) as { id?: string; message?: unknown });
		const oldText = messageText(source.find((entry) => entry.id === prior.acceptedEdit!.sourceId)?.message);
		if (!oldText || createHash("sha256").update(oldText).digest("hex") !== prior.acceptedEdit.originalTextSha256) {
			throw new Error("Fresh-process resume could not verify the original edited source entry");
		}
		const captureStart = captures.length;
		const metrics: TurnMetric[] = [];
		await runTurn(reopened, "fresh-process-resume", "Confirm that the accepted summary survives a fresh process resume.", metrics);
		const payload = captures[captureStart];
		if (!payload) throw new Error("Fresh-process resume emitted no provider payload");
		const messages = payloadMessages(payload);
		if (!messages.some((message) => messageText(message) === prior.acceptedEdit!.replacement)) throw new Error("Fresh-process resume lost the committed replacement");
		if (messages.some((message) => messageText(message) === oldText)) throw new Error("Fresh-process resume restored removed assistant text");
		const payloadPath = join(outDir, "process-resume-payloads.jsonl");
		await writeJsonl(payloadPath, captures);
		const resumeReport = { previousEpoch: prior.hardReset.epoch, resumedEpoch: status.cacheEpoch, retainedEdit: true, originalAssistantAbsentFromEffectiveRequest: true, originalTextSha256: createHash("sha256").update(oldText).digest("hex"), payloadEvidence: payloadPath, turnMetrics: metrics, notes: ["This report was produced by a separate process using SessionManager.open and createAgentSession."] };
		await writeFile(join(outDir, "process-resume-report.json"), `${JSON.stringify(resumeReport, null, 2)}\n`);
		process.stdout.write(`${JSON.stringify(resumeReport, null, 2)}\n`);
	} finally {
		reopened.dispose();
	}
}

const modelId = option("--model", "qwen3.8:27b-q8_0");
const ollamaUrl = option("--ollama-url", "http://127.0.0.1:11434");
const outDir = resolve(option("--out", "artifacts/clm-ollama-demo"));
const cwd = resolve(join(outDir, "project"));
const agentDir = resolve(join(outDir, "agent"));
const contextDir = resolve(join(outDir, "context-editing"));
await mkdir(cwd, { recursive: true });
await mkdir(agentDir, { recursive: true });
await assertReady(ollamaUrl, modelId);

const model: Model<"openai-completions"> = {
	id: modelId,
	name: modelId,
	api: "openai-completions",
	provider: "ollama",
	baseUrl: `${ollamaUrl.replace(/\/$/, "")}/v1`,
	reasoning: true,
	thinkingLevelMap: { off: "none", minimal: "none", low: "none", medium: "none", high: "none", xhigh: "none", max: "none" },
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 32768,
	maxTokens: 256,
	compat: { supportsReasoningEffort: true, supportsUsageInStreaming: true, supportsDeveloperRole: false, maxTokensField: "max_tokens" },
};

const contextOptions = {
	directory: contextDir,
	maxPromptTokens: 32768,
	fixedPromptTokens: 1024,
	maxOutputTokens: 256,
};
if (process.argv.includes("--resume-existing")) {
	await runResumeMode(model, outDir, cwd, agentDir, contextOptions);
	process.exit(0);
}
const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
const modelRegistry = ModelRegistry.create(authStorage, join(agentDir, "models.json"));
modelRegistry.registerProvider("ollama", { apiKey: "ollama" });
const settings = SettingsManager.create(cwd, agentDir);
const captures: CapturedPayload[] = [];
const turnMetrics: TurnMetric[] = [];
const saveFailure = async (reason: unknown): Promise<void> => {
	const failure = { error: reason instanceof Error ? reason.message : String(reason), turnMetrics, captures: captures.length, at: new Date().toISOString() };
	await writeFile(join(outDir, "failure-report.json"), `${JSON.stringify(failure, null, 2)}\n`);
};
process.on("unhandledRejection", (reason) => {
	saveFailure(reason).finally(() => { process.exitCode = 1; });
});
process.on("uncaughtException", (error) => {
	saveFailure(error).finally(() => { process.exitCode = 1; });
});
const sessionManager = SessionManager.create(cwd, join(outDir, "sessions"));
const loader = await makeLoader(cwd, agentDir, settings, captures);
const created = await createAgentSession({
	cwd,
	agentDir,
	authStorage,
	modelRegistry,
	settingsManager: settings,
	resourceLoader: loader,
	sessionManager,
	model,
	thinkingLevel: "off",
	noTools: "all",
	includeGoals: false,
	contextEditing: contextOptions,
});
const session = created.session;
await runTurn(session, "initial-memo", "Write a detailed historical memo of about 120 words describing a fictional migration project. Include concrete but disposable implementation details; do not mention this request in the memo.", turnMetrics);
await runTurn(session, "historical-risk", "Now state one concise risk from that memo.", turnMetrics);
const beforeEditBranch = branchMessages(session);
const oldAssistant = beforeEditBranch.find((entry) => entry.role === "assistant" && entry.text.trim());
if (!oldAssistant) throw new Error("The SDK produced no assistant history to edit");
const oldTranscriptPath = session.sessionManager.getSessionFile();
if (!oldTranscriptPath) throw new Error("Expected a durable SDK session file");
const transcriptBefore = await readFile(oldTranscriptPath);
await writeFile(join(outDir, "transcript-before-edit.jsonl"), transcriptBefore);
const statusBefore = session.getContextEditingStatus();
if (!statusBefore) throw new Error("Context editing was not enabled");
const replacement = "Accepted context edit summary: archive the fictional migration memo; retain only its project risk and implementation trade-offs.";
await writeFile(
	statusBefore.stagingPath,
	`${JSON.stringify({ schemaVersion: 1, sessionId: statusBefore.sessionId, baseRevision: statusBefore.revision, baseDigest: statusBefore.effectiveDigest, edits: [{ sourceId: oldAssistant.id, op: "replace_text", text: replacement }] }, null, 2)}\n`,
);
const captureBeforeEdit = captures.length;
await runTurn(session, "accepted-edit", "Confirm the accepted context edit with one concise sentence.", turnMetrics);
const editedStatus = session.getContextEditingStatus();
if (!editedStatus || editedStatus.revision <= statusBefore.revision) throw new Error("The staged SDK context edit was not committed");
const editedPayload = captures[captureBeforeEdit];
if (!editedPayload) throw new Error("No provider payload was captured for the edited request");
const editedMessages = payloadMessages(editedPayload);
const editedWire = JSON.stringify(editedMessages);
if (editedMessages.some((message) => messageText(message) === oldAssistant.text)) throw new Error("Edited outgoing prompt still contains the removed assistant history");
for (const userMessage of beforeEditBranch.filter((entry) => entry.role === "user")) {
	if (userMessage.text && !editedWire.includes(userMessage.text)) throw new Error(`Protected user instruction disappeared: ${userMessage.id}`);
}
if (!editedWire.includes(replacement)) throw new Error("Edited outgoing prompt does not contain the accepted replacement");
const previousPayload = captures[captureBeforeEdit - 1];
let previousContextChars: number | undefined;
let editedContextChars: number | undefined;
if (previousPayload) {
	const previousContext = JSON.stringify(payloadMessages(previousPayload).slice(0, -1));
	const editedContext = JSON.stringify(editedMessages.slice(0, -1));
	previousContextChars = previousContext.length;
	editedContextChars = editedContext.length;
	if (editedContext.length >= previousContext.length) throw new Error("Accepted edit did not reduce the effective wire context");
}
const transcriptAfterEdit = await readFile(oldTranscriptPath);
if (!transcriptAfterEdit.subarray(0, transcriptBefore.length).equals(transcriptBefore)) throw new Error("SDK transcript was not append-preserved");
const rawContext = await readFile(join(contextDir, "source.jsonl"), "utf8");
const rawEntries = rawContext.split("\n").filter(Boolean).map((line) => JSON.parse(line) as { message?: unknown });
if (!rawEntries.some((entry) => messageText(entry.message) === oldAssistant.text)) throw new Error("Durable raw source no longer contains the original assistant history");
const epochBeforeReopen = editedStatus.cacheEpoch;
session.dispose();

const reopenedManager = SessionManager.open(oldTranscriptPath, join(outDir, "sessions"), cwd);
const reopenedCaptures: CapturedPayload[] = [];
const reopenedLoader = await makeLoader(cwd, agentDir, settings, reopenedCaptures);
const reopened = (await createAgentSession({ cwd, agentDir, authStorage, modelRegistry, settingsManager: settings, resourceLoader: reopenedLoader, sessionManager: reopenedManager, model, thinkingLevel: "off", noTools: "all", includeGoals: false, contextEditing: contextOptions })).session;
const reopenedStatus = reopened.getContextEditingStatus();
if (!reopenedStatus || reopenedStatus.cacheEpoch === epochBeforeReopen) throw new Error("Cold reopen did not rotate the cache epoch");
const reopenedCaptureStart = reopenedCaptures.length;
await runTurn(reopened, "cold-reopen", "Confirm after cold reopen that the accepted summary remains the effective context.", turnMetrics);
const reopenedPayload = reopenedCaptures[reopenedCaptureStart];
if (!reopenedPayload || !JSON.stringify(payloadMessages(reopenedPayload)).includes(replacement)) throw new Error("Cold reopen lost the accepted context edit");
const beforeResetStatus = reopened.getContextEditingStatus();
if (!beforeResetStatus) throw new Error("Context editing status disappeared before hard reset");
const epochBeforeReset = beforeResetStatus.cacheEpoch;
const digestBeforeReset = beforeResetStatus.effectiveDigest;
const resetSnapshot = await reopened.hardResetContextEditing("demo explicit hard reset");
const resetStatus = reopened.getContextEditingStatus();
if (!resetStatus || resetStatus.cacheEpoch === epochBeforeReset) throw new Error("Explicit hard reset did not change the cache epoch");
if (resetStatus.effectiveDigest !== digestBeforeReset) throw new Error("Hard reset changed effective context instead of only rotating epoch");
if (!JSON.stringify(resetSnapshot.messages).includes(replacement)) throw new Error("Hard reset unexpectedly restored removed assistant history");
const resetCaptureStart = reopenedCaptures.length;
await runTurn(reopened, "after-hard-reset", "Confirm after the explicit hard reset that the accepted summary remains effective.", turnMetrics);
const resetPayload = reopenedCaptures[resetCaptureStart];
if (!resetPayload || !JSON.stringify(payloadMessages(resetPayload)).includes(replacement)) throw new Error("Post-reset request lost the accepted summary");
reopened.dispose();
const mainPayloadPath = join(outDir, "payloads-main.jsonl");
const reopenResetPayloadPath = join(outDir, "payloads-reopen-reset.jsonl");
await writeJsonl(mainPayloadPath, captures);
await writeJsonl(reopenResetPayloadPath, reopenedCaptures);

const report = {
	model: modelId,
	ollamaUrl,
	sessionFile: oldTranscriptPath,
	contextDirectory: contextDir,
	originalTranscriptPreserved: true,
	acceptedEdit: { revision: editedStatus.revision, sourceId: oldAssistant.id, removedAssistantCharacters: oldAssistant.text.length, originalTextSha256: createHash("sha256").update(oldAssistant.text).digest("hex"), replacement },
	coldReopen: { retainedEdit: true, previousEpoch: epochBeforeReopen, reopenedEpoch: reopenedStatus.cacheEpoch, processRestart: false },
	hardReset: { previousEpoch: epochBeforeReset, epoch: resetStatus.cacheEpoch, previousEffectiveDigest: digestBeforeReset, retainedEffectiveDigest: resetStatus.effectiveDigest },
	providerPayloadChecks: { editedPromptRemovedOriginalAssistant: true, preservedUserInstructions: true },
	smallerInput: { beforeContextChars: previousContextChars, afterContextChars: editedContextChars },
	payloadEvidence: { transcriptBeforeEdit: join(outDir, "transcript-before-edit.jsonl"), main: mainPayloadPath, reopenReset: reopenResetPayloadPath },
	turnMetrics,
	notes: ["This is a real Prime createAgentSession run against local Ollama.", "Ollama calls intentionally omit SGLang cache_salt/rid metadata; no SCR or CUDA capability is claimed."],
};
await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
