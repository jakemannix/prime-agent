import { createHash, randomUUID } from "node:crypto";
import {
	closeSync,
	existsSync,
	fsyncSync,
	mkdirSync,
	openSync,
	readFileSync,
	renameSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	CONTEXT_EDITING_SCHEMA_VERSION,
	CONTEXT_EDITING_SERIALIZER,
	ContextEditingError,
	type ContextEditProposal,
	type ContextEditReceipt,
	type ContextStoreFaultInjector,
	type EffectiveEntry,
	type SourceEntry,
} from "./types.js";

export interface DurableResetRecord {
	reason: string;
	at: string;
	previousEpoch: string;
	newEpoch: string;
}

export interface DurableContextState {
	schemaVersion: number;
	serializer: string;
	sessionId: string;
	revision: number;
	cacheEpoch: string;
	sourceEntries: SourceEntry[];
	effectiveEntries: EffectiveEntry[];
	sourceDigest: string;
	sourceFrontier?: string;
	effectiveDigest: string;
	servingFingerprint?: string;
	lastReceipt?: ContextEditReceipt;
	handledProposalIdentity?: string;
	handledProposalContentDigest?: string;
	lastReset?: DurableResetRecord;
}

interface DurableGeneration {
	schemaVersion: number;
	generationId: string;
	state: DurableContextState;
}

interface DurableManifest {
	schemaVersion: number;
	sessionId: string;
	generationId: string;
	generationDigest: string;
}

export interface ProposalRecord {
	readonly value: unknown;
	readonly fileDigest: string;
}

export function canonicalJson(value: unknown): string {
	if (value === null || typeof value !== "object") {
		const encoded = JSON.stringify(value);
		return encoded === undefined ? "null" : encoded;
	}
	if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
	const record = value as Record<string, unknown>;
	return `{${Object.keys(record)
		.filter((key) => JSON.stringify(record[key]) !== undefined)
		.sort()
		.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
		.join(",")}}`;
}

export function digest(value: unknown): string {
	return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

export function sourceDigest(entries: readonly SourceEntry[]): string {
	return digest({ schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION, serializer: CONTEXT_EDITING_SERIALIZER, entries });
}

export function effectiveDigest(entries: readonly EffectiveEntry[]): string {
	return digest({ schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION, serializer: CONTEXT_EDITING_SERIALIZER, entries });
}

function clone<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

function durableError(message: string): ContextEditingError {
	return new ContextEditingError("corrupt_state", message);
}

function syncDirectory(path: string): void {
	const fd = openSync(path, "r");
	try {
		fsyncSync(fd);
	} finally {
		closeSync(fd);
	}
}

function atomicWrite(path: string, contents: string, afterRename?: () => void): void {
	mkdirSync(dirname(path), { recursive: true });
	const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
	const fd = openSync(temporary, "wx", 0o600);
	try {
		writeFileSync(fd, contents, "utf8");
		fsyncSync(fd);
	} finally {
		closeSync(fd);
	}
	renameSync(temporary, path);
	afterRename?.();
	syncDirectory(dirname(path));
}

function jsonWrite(path: string, value: unknown): void {
	atomicWrite(
		path,
		`${JSON.stringify(value, null, 2)}
`,
	);
}

function parseJson(path: string): unknown {
	try {
		return JSON.parse(readFileSync(path, "utf8")) as unknown;
	} catch (error) {
		throw durableError(
			`Cannot read authoritative context state ${path}: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertString(value: unknown, label: string): asserts value is string {
	if (typeof value !== "string" || value.length === 0) throw durableError(`Invalid ${label} in authoritative state`);
}

function assertToolProtocol(messages: readonly AgentMessage[], label: string): void {
	const calls = new Map<string, number>();
	const results = new Map<string, number[]>();
	for (const [index, message] of messages.entries()) {
		const record = message as unknown as Record<string, unknown>;
		if (Array.isArray(record.content)) {
			for (const block of record.content) {
				if (!isRecord(block) || block.type !== "toolCall") continue;
				if (typeof block.id !== "string" || !block.id || calls.has(block.id))
					throw durableError(`${label} has a duplicate or invalid tool call ID`);
				calls.set(block.id, index);
			}
		}
		if (record.role === "toolResult") {
			if (typeof record.toolCallId !== "string" || !record.toolCallId)
				throw durableError(`${label} has a tool result without an ID`);
			results.set(record.toolCallId, [...(results.get(record.toolCallId) ?? []), index]);
		}
	}
	for (const [id, callIndex] of calls) {
		const matches = results.get(id) ?? [];
		if (matches.length !== 1 || matches[0]! <= callIndex)
			throw durableError(`${label} does not preserve a tool call/result bijection for ${id}`);
	}
	for (const id of results.keys()) {
		if (!calls.has(id)) throw durableError(`${label} has an orphan tool result for ${id}`);
	}
}

function assertState(value: unknown): asserts value is DurableContextState {
	if (!isRecord(value)) throw durableError("Authoritative context state is not an object");
	if (value.schemaVersion !== CONTEXT_EDITING_SCHEMA_VERSION)
		throw durableError("Unsupported context state schema version");
	if (value.serializer !== CONTEXT_EDITING_SERIALIZER) throw durableError("Unsupported context serializer");
	assertString(value.sessionId, "sessionId");
	assertString(value.cacheEpoch, "cacheEpoch");
	assertString(value.sourceDigest, "sourceDigest");
	assertString(value.effectiveDigest, "effectiveDigest");
	if (!Number.isInteger(value.revision) || (value.revision as number) < 0)
		throw durableError("Invalid context revision");
	if (!Array.isArray(value.sourceEntries) || !Array.isArray(value.effectiveEntries))
		throw durableError("Invalid context entries");
	const sourceEntries = value.sourceEntries as unknown as SourceEntry[];
	const effectiveEntries = value.effectiveEntries as unknown as EffectiveEntry[];
	for (const entry of sourceEntries) {
		if (!isRecord(entry) || typeof entry.id !== "string" || !entry.id || !isRecord(entry.message))
			throw durableError("Invalid durable source entry");
	}
	for (const entry of effectiveEntries) {
		if (
			!isRecord(entry) ||
			typeof entry.id !== "string" ||
			!entry.id ||
			!Array.isArray(entry.sourceIds) ||
			entry.sourceIds.length === 0 ||
			entry.sourceIds.some((id) => typeof id !== "string" || !id) ||
			!isRecord(entry.message)
		) {
			throw durableError("Invalid durable effective entry");
		}
	}
	const sourceById = new Map<string, SourceEntry>();
	for (const entry of sourceEntries) {
		if (sourceById.has(entry.id)) throw durableError(`Duplicate committed source ID ${entry.id}`);
		sourceById.set(entry.id, entry);
	}
	const effectiveIds = new Set<string>();
	const representedSourceIds = new Set<string>();
	for (const entry of effectiveEntries) {
		if (effectiveIds.has(entry.id)) throw durableError(`Duplicate effective entry ID ${entry.id}`);
		effectiveIds.add(entry.id);
		const positions = entry.sourceIds.map((id) => sourceEntries.findIndex((source) => source.id === id));
		if (positions.some((position) => position < 0))
			throw durableError("Effective entry has unknown source provenance");
		if (positions.some((position, index) => index > 0 && position !== positions[index - 1]! + 1))
			throw durableError("Effective entry provenance is not a contiguous ordered source range");
		for (const id of entry.sourceIds) {
			if (representedSourceIds.has(id)) throw durableError(`Source ID ${id} is represented more than once`);
			representedSourceIds.add(id);
		}
	}
	for (const source of sourceEntries) {
		const role = (source.message as unknown as { role?: unknown }).role;
		if (role !== "user") continue;
		const retained = effectiveEntries.find(
			(entry) => entry.sourceIds.length === 1 && entry.sourceIds[0] === source.id,
		);
		if (!retained || canonicalJson(retained.message) !== canonicalJson(source.message))
			throw durableError(`Protected user source ${source.id} is not retained exactly`);
	}
	assertToolProtocol(
		sourceEntries.map((entry) => entry.message),
		"Committed source",
	);
	assertToolProtocol(
		effectiveEntries.map((entry) => entry.message),
		"Committed effective context",
	);
}

function sameEntry(left: SourceEntry, right: SourceEntry): boolean {
	return left.id === right.id && canonicalJson(left.message) === canonicalJson(right.message);
}

export class DurableContextStore {
	readonly directory: string;
	readonly stagingPath: string;
	readonly protocolPath: string;
	private readonly statePath: string;
	private readonly proposalWriteMarkerPath: string;
	private readonly manifestPath: string;
	private readonly sourcePath: string;
	private readonly generationsPath: string;
	private readonly ownerPath: string;
	private readonly lockPath: string;
	private readonly faultInjector?: ContextStoreFaultInjector;
	private lockFd: number | undefined;
	private lockToken: string | undefined;
	private committedGenerationId: string | undefined;
	private committedManifestDigest: string | undefined;
	private closed = false;

	private constructor(directory: string, faultInjector?: ContextStoreFaultInjector) {
		this.directory = resolve(directory);
		this.stagingPath = join(this.directory, "proposal.json");
		this.protocolPath = join(this.directory, "protocol.md");
		this.statePath = join(this.directory, "state.json");
		this.proposalWriteMarkerPath = join(this.directory, "proposal.write-in-progress.json");
		this.manifestPath = join(this.directory, "manifest.json");
		this.sourcePath = join(this.directory, "source.jsonl");
		this.generationsPath = join(this.directory, "generations");
		this.ownerPath = join(this.directory, "owner.json");
		this.lockPath = join(this.directory, "owner.lock");
		this.faultInjector = faultInjector;
	}

	private claimLock(sessionId: string): void {
		for (let attempt = 0; attempt < 3; attempt++) {
			const token = randomUUID();
			try {
				const fd = openSync(this.lockPath, "wx", 0o600);
				try {
					writeFileSync(
						fd,
						`${JSON.stringify({ pid: process.pid, sessionId, token, createdAt: new Date().toISOString() })}
`,
						"utf8",
					);
					fsyncSync(fd);
					syncDirectory(this.directory);
				} catch (error) {
					closeSync(fd);
					try {
						unlinkSync(this.lockPath);
					} catch {
						// The original error is authoritative.
					}
					throw error;
				}
				this.lockFd = fd;
				this.lockToken = token;
				return;
			} catch (error) {
				const code = (error as NodeJS.ErrnoException).code;
				if (code !== "EEXIST") {
					throw new ContextEditingError(
						"session_mismatch",
						`Unable to claim context ownership lock: ${String(error)}`,
					);
				}
			}

			let observed: string;
			let ownerPid: number;
			try {
				observed = readFileSync(this.lockPath, "utf8");
				const owner = JSON.parse(observed) as { pid?: unknown; token?: unknown };
				if (
					typeof owner.pid !== "number" ||
					!Number.isSafeInteger(owner.pid) ||
					owner.pid <= 0 ||
					typeof owner.token !== "string" ||
					!owner.token
				)
					throw new Error("invalid lock record");
				ownerPid = owner.pid;
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
				throw new ContextEditingError("corrupt_state", "Context ownership lock is corrupt");
			}
			if (ownerPid === process.pid)
				throw new ContextEditingError("session_mismatch", "Context directory is already owned by this process");
			try {
				process.kill(ownerPid, 0);
				throw new ContextEditingError("session_mismatch", `Context directory is owned by process ${ownerPid}`);
			} catch (error) {
				if (error instanceof ContextEditingError) throw error;
				if ((error as NodeJS.ErrnoException).code !== "ESRCH")
					throw new ContextEditingError("session_mismatch", `Cannot prove ownership lock ${ownerPid} is stale`);
			}
			const stalePath = `${this.lockPath}.stale-${process.pid}-${randomUUID()}`;
			try {
				renameSync(this.lockPath, stalePath);
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
				throw new ContextEditingError("session_mismatch", "Unable to quarantine stale ownership lock");
			}
			if (readFileSync(stalePath, "utf8") !== observed) {
				if (!existsSync(this.lockPath)) renameSync(stalePath, this.lockPath);
				throw new ContextEditingError("session_mismatch", "Ownership lock changed during stale-lock recovery");
			}
			unlinkSync(stalePath);
			syncDirectory(this.directory);
		}
		throw new ContextEditingError("session_mismatch", "Unable to claim context ownership lock");
	}

	private ensureProtocolGuide(): void {
		const contents = [
			"# Context editing protocol",
			"",
			"`manifest.json` is the sole authoritative commit pointer. Follow its `generationId` to",
			"`generations/<generationId>.json`. The generation state contains the current revision, base digest,",
			"stable source IDs, committed effective entries, source frontier, cache epoch, and last receipt.",
			"A legacy `state.json`, if present, is informational only. Never use it as a commit pointer.",
			"",
			"To propose an edit, atomically replace `proposal.json` with one JSON object:",
			"",
			"```json",
			'{"schemaVersion":1,"proposalId":"<fresh UUID>","sessionId":"<owner sessionId>","baseRevision":0,"baseDigest":"<generation state effectiveDigest>","edits":[{"op":"replace_text","sourceId":"<stable assistant source ID>","text":"replacement"}],"hardReset":false,"resetReason":"optional"}',
			"```",
			"",
			"`proposalId` identifies one intended operation. Reuse it only when retrying that operation; use a fresh UUID",
			"for every new proposal, including a byte-identical hard reset. Legacy proposals without it remain accepted",
			"but cannot distinguish an exact retry from a new byte-identical operation.",
			"",
			"Supported edit operations are `replace_text`, `replace`, `delete`, and whole closed tool-group `summary`.",
			"All user entries, signed content, native metadata, and live tool groups are protected.",
			"A proposal is only committed at a request boundary after validation, budget checking, and durable commit.",
			"Set `hardReset` to rotate cache lineage while preserving the resulting effective context.",
			"",
		].join("\n");
		if (existsSync(this.protocolPath)) {
			if (readFileSync(this.protocolPath, "utf8") !== contents)
				throw durableError("Context editing protocol guide was modified");
			return;
		}
		atomicWrite(this.protocolPath, contents);
	}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		if (this.lockFd !== undefined) {
			closeSync(this.lockFd);
			this.lockFd = undefined;
		}
		const token = this.lockToken;
		this.lockToken = undefined;
		if (!token) return;
		try {
			const owner = JSON.parse(readFileSync(this.lockPath, "utf8")) as { token?: unknown };
			if (owner.token === token) {
				unlinkSync(this.lockPath);
				syncDirectory(this.directory);
			}
		} catch {
			// Never unlink a lock unless its unique token still belongs to this instance.
		}
	}

	private assertOwned(): void {
		if (this.closed || this.lockFd === undefined || this.lockToken === undefined)
			throw new ContextEditingError("protocol_violation", "Context store is closed or unowned");
		try {
			const owner = JSON.parse(readFileSync(this.lockPath, "utf8")) as { token?: unknown };
			if (owner.token !== this.lockToken)
				throw new ContextEditingError("session_mismatch", "Context ownership lock token changed");
		} catch (error) {
			if (error instanceof ContextEditingError) throw error;
			throw new ContextEditingError("session_mismatch", "Context ownership lock is missing or corrupt");
		}
	}

	static open(
		directory: string,
		sessionId: string,
		faultInjector?: ContextStoreFaultInjector,
	): { store: DurableContextStore; state: DurableContextState; reopened: boolean } {
		if (!directory) throw new ContextEditingError("invalid_config", "Context editing requires a durable directory");
		const store = new DurableContextStore(directory, faultInjector);
		mkdirSync(store.directory, { recursive: true, mode: 0o700 });
		store.claimLock(sessionId);
		try {
			if (existsSync(store.ownerPath)) {
				const owner = parseJson(store.ownerPath);
				if (!isRecord(owner) || owner.sessionId !== sessionId)
					throw new ContextEditingError("session_mismatch", "Context directory belongs to a different session");
			} else {
				if (existsSync(store.statePath) || existsSync(store.manifestPath) || existsSync(store.sourcePath))
					throw durableError("Context directory has state but no ownership record");
				jsonWrite(store.ownerPath, { schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION, sessionId });
			}
			store.ensureProtocolGuide();

			if (!existsSync(store.manifestPath)) {
				if (existsSync(store.statePath) || existsSync(store.sourcePath))
					throw durableError("Context commit pointer is missing");
				const initial: DurableContextState = {
					schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION,
					serializer: CONTEXT_EDITING_SERIALIZER,
					sessionId,
					revision: 0,
					cacheEpoch: `epoch-${randomUUID()}`,
					sourceEntries: [],
					effectiveEntries: [],
					sourceDigest: sourceDigest([]),
					effectiveDigest: effectiveDigest([]),
				};
				store.commitState(initial);
				return { store, state: initial, reopened: false };
			}

			const manifestValue = parseJson(store.manifestPath);
			if (
				!isRecord(manifestValue) ||
				manifestValue.schemaVersion !== CONTEXT_EDITING_SCHEMA_VERSION ||
				manifestValue.sessionId !== sessionId ||
				typeof manifestValue.generationId !== "string" ||
				!/^[a-zA-Z0-9-]+$/.test(manifestValue.generationId) ||
				typeof manifestValue.generationDigest !== "string"
			) {
				throw durableError("Context manifest pointer is invalid");
			}
			const manifest = manifestValue as unknown as DurableManifest;
			const generationPath = join(store.generationsPath, `${manifest.generationId}.json`);
			if (!existsSync(generationPath)) throw durableError("Committed context generation is missing");
			const generationValue = parseJson(generationPath);
			if (
				!isRecord(generationValue) ||
				generationValue.schemaVersion !== CONTEXT_EDITING_SCHEMA_VERSION ||
				generationValue.generationId !== manifest.generationId ||
				digest(generationValue) !== manifest.generationDigest
			) {
				throw durableError("Committed context generation digest is invalid");
			}
			const generation = generationValue as unknown as DurableGeneration;
			assertState(generation.state);
			const state = clone(generation.state);
			if (state.sessionId !== sessionId)
				throw new ContextEditingError("session_mismatch", "Context generation session ID mismatch");
			if (sourceDigest(state.sourceEntries) !== state.sourceDigest)
				throw durableError("Authoritative source digest does not match its contents");
			if (effectiveDigest(state.effectiveEntries) !== state.effectiveDigest)
				throw durableError("Authoritative effective digest does not match its contents");
			if (state.sourceFrontier !== state.sourceEntries[state.sourceEntries.length - 1]?.id)
				throw durableError("Authoritative source frontier ID is invalid");
			const raw = store.readRawSource();
			if (raw.length < state.sourceEntries.length)
				throw durableError("Raw source is behind the committed context generation");
			for (let i = 0; i < state.sourceEntries.length; i++) {
				if (!sameEntry(raw[i]!, state.sourceEntries[i]!))
					throw durableError("Raw source prefix diverges from the committed context generation");
			}
			const ids = new Set<string>();
			for (const entry of raw) {
				if (ids.has(entry.id)) throw durableError(`Raw source contains duplicate ID ${entry.id}`);
				ids.add(entry.id);
			}
			store.committedGenerationId = manifest.generationId;
			store.committedManifestDigest = digest(manifest);
			return { store, state, reopened: true };
		} catch (error) {
			store.close();
			throw error;
		}
	}

	private readRawSource(): SourceEntry[] {
		if (!existsSync(this.sourcePath)) return [];
		const text = readFileSync(this.sourcePath, "utf8");
		if (!text) return [];
		if (!text.endsWith("\n")) throw durableError("Raw source has an incomplete trailing record");
		return text
			.split("\n")
			.filter(Boolean)
			.map((line, index) => {
				try {
					const parsed = JSON.parse(line) as unknown;
					if (!isRecord(parsed) || typeof parsed.id !== "string" || !parsed.id || !isRecord(parsed.message))
						throw new Error("invalid entry");
					return { id: parsed.id, message: parsed.message as unknown as AgentMessage };
				} catch (error) {
					throw durableError(
						`Raw source record ${index + 1} is corrupt: ${error instanceof Error ? error.message : String(error)}`,
					);
				}
			});
	}

	verifySourceInput(entries: readonly SourceEntry[]): void {
		this.assertOwned();
		const raw = this.readRawSource();
		if (entries.length < raw.length)
			throw new ContextEditingError(
				"source_divergence",
				"Source frontier moved backwards behind the verified durable source tail",
			);
		for (let i = 0; i < raw.length; i++) {
			if (!sameEntry(raw[i]!, entries[i]!))
				throw new ContextEditingError(
					"source_divergence",
					`Supplied Prime journal diverges from durable source record ${raw[i]!.id}`,
				);
		}
	}

	private appendSourceForState(state: DurableContextState): void {
		const raw = this.readRawSource();
		if (state.sourceEntries.length < raw.length) throw durableError("Commit would truncate durable source");
		for (let i = 0; i < raw.length; i++) {
			if (!sameEntry(raw[i]!, state.sourceEntries[i]!))
				throw durableError("Commit source prefix diverges from raw source");
		}
		const appended = state.sourceEntries.slice(raw.length);
		if (appended.length === 0) return;
		const fd = openSync(this.sourcePath, "a", 0o600);
		try {
			writeFileSync(
				fd,
				appended
					.map(
						(entry) => `${JSON.stringify(entry)}
`,
					)
					.join(""),
				"utf8",
			);
			fsyncSync(fd);
		} finally {
			closeSync(fd);
		}
		syncDirectory(this.directory);
	}

	private assertCurrentPointer(): void {
		this.assertOwned();
		if (this.committedGenerationId === undefined) {
			if (existsSync(this.manifestPath)) throw durableError("Context commit pointer appeared concurrently");
			return;
		}
		const manifest = parseJson(this.manifestPath);
		if (
			!isRecord(manifest) ||
			manifest.generationId !== this.committedGenerationId ||
			digest(manifest) !== this.committedManifestDigest
		) {
			throw durableError("Context commit pointer changed outside its exclusive owner");
		}
	}

	private commitState(state: DurableContextState): void {
		this.assertCurrentPointer();
		const copy = clone(state);
		if (
			sourceDigest(copy.sourceEntries) !== copy.sourceDigest ||
			effectiveDigest(copy.effectiveEntries) !== copy.effectiveDigest
		)
			throw durableError("Refusing to commit a state with invalid digests");
		mkdirSync(this.generationsPath, { recursive: true, mode: 0o700 });
		const generationId = `generation-${randomUUID()}`;
		const generation: DurableGeneration = {
			schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION,
			generationId,
			state: copy,
		};
		const generationPath = join(this.generationsPath, `${generationId}.json`);
		jsonWrite(generationPath, generation);
		this.faultInjector?.("before_manifest_swap");
		const manifest = {
			schemaVersion: CONTEXT_EDITING_SCHEMA_VERSION,
			sessionId: copy.sessionId,
			generationId,
			generationDigest: digest(generation),
		} satisfies DurableManifest;
		jsonWrite(this.manifestPath, manifest);
		this.faultInjector?.("after_manifest_swap");
		this.committedGenerationId = generationId;
		this.committedManifestDigest = digest(manifest);
	}

	commit(state: DurableContextState): void {
		this.assertCurrentPointer();
		this.appendSourceForState(state);
		this.commitState(state);
	}

	commitStateForController(state: DurableContextState): void {
		this.commitState(state);
	}

	writeProposal(proposal: ContextEditProposal): void {
		this.assertOwned();
		if (existsSync(this.proposalWriteMarkerPath))
			throw durableError("An interrupted proposal staging write requires recovery");
		const contents = `${JSON.stringify(proposal, null, 2)}\n`;
		const fileDigest = digest(contents);
		jsonWrite(this.proposalWriteMarkerPath, { fileDigest });
		try {
			atomicWrite(this.stagingPath, contents, () => this.faultInjector?.("after_proposal_rename"));
			unlinkSync(this.proposalWriteMarkerPath);
			syncDirectory(this.directory);
		} catch (error) {
			try {
				if (existsSync(this.stagingPath)) {
					const staged = readFileSync(this.stagingPath, "utf8");
					if (digest(staged) !== fileDigest) throw durableError("Proposal changed during failed staging cleanup");
					unlinkSync(this.stagingPath);
					syncDirectory(this.directory);
				}
				if (existsSync(this.proposalWriteMarkerPath)) {
					unlinkSync(this.proposalWriteMarkerPath);
					syncDirectory(this.directory);
				}
			} catch {
				// Leave the durable marker in place so reopen fails closed.
			}
			throw error;
		}
	}

	readProposal(): ProposalRecord | undefined {
		this.assertOwned();
		if (existsSync(this.proposalWriteMarkerPath))
			throw durableError("An interrupted proposal staging write requires recovery");
		if (!existsSync(this.stagingPath)) return undefined;
		const text = readFileSync(this.stagingPath, "utf8");
		try {
			return { value: JSON.parse(text) as unknown, fileDigest: digest(text) };
		} catch {
			return { value: null, fileDigest: digest(text) };
		}
	}

	consumeProposal(expectedDigest: string): boolean {
		this.assertOwned();
		if (!existsSync(this.stagingPath)) return true;
		const text = readFileSync(this.stagingPath, "utf8");
		if (digest(text) !== expectedDigest) return false;
		unlinkSync(this.stagingPath);
		syncDirectory(this.directory);
		return true;
	}

	writeMarkdownProtocol(proposal: ContextEditProposal): void {
		this.assertOwned();
		atomicWrite(
			join(this.directory, "proposal.md"),
			[
				"# Context editing proposal",
				"",
				"Edit `proposal.json`; this Markdown file is explanatory and is not authoritative.",
				"",
				"```json",
				JSON.stringify(proposal, null, 2),
				"```",
				"",
			].join("\n"),
		);
	}
}
