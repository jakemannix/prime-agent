import { randomUUID } from "node:crypto";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	canonicalJson,
	type DurableContextState,
	DurableContextStore,
	digest,
	effectiveDigest,
	sourceDigest,
} from "./store.js";
import {
	type ContextBudgetOverride,
	type ContextEditConfig,
	type ContextEditingControllerOptions,
	ContextEditingError,
	type ContextEditingStatus,
	type ContextEditProposal,
	type ContextEditReceipt,
	type EffectiveEntry,
	type RequestSnapshot,
	type SourceEntry,
} from "./types.js";

interface ToolGroup {
	ids: string[];
	complete: boolean;
}
interface SourceAnalysis {
	groups: ToolGroup[];
	incompleteIds: Set<string>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
function clone<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}
function deepFreeze<T>(value: T): T {
	if (value && typeof value === "object" && !Object.isFrozen(value)) {
		Object.freeze(value);
		for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
	}
	return value;
}
function messageRole(message: AgentMessage): string {
	return (message as unknown as { role?: string }).role ?? "";
}
function contentOf(message: AgentMessage): unknown {
	return (message as unknown as { content?: unknown }).content;
}
function assertTextOnlyMessage(message: AgentMessage, label: string): void {
	if (!isRecord(message) || typeof message.role !== "string")
		throw new ContextEditingError("invalid_source", `${label} has no valid role`);
	const content = (message as Record<string, unknown>).content;
	if (typeof content === "string") return;
	if (!Array.isArray(content)) throw new ContextEditingError("invalid_source", `${label} has invalid content`);
	const role = messageRole(message);
	for (const block of content) {
		if (!isRecord(block) || (block.type !== "text" && block.type !== "thinking" && block.type !== "toolCall")) {
			throw new ContextEditingError("protocol_violation", `${label} contains unsupported multimodal content`);
		}
		if ((role === "user" || role === "toolResult") && block.type !== "text") {
			throw new ContextEditingError("protocol_violation", `${label} contains non-text content for ${role}`);
		}
		if (
			(block.type === "text" || block.type === "thinking") &&
			typeof block[block.type === "text" ? "text" : "thinking"] !== "string"
		) {
			throw new ContextEditingError("invalid_source", `${label} contains malformed content`);
		}
	}
}
function toolCalls(message: AgentMessage): Array<{ id: string }> {
	const content = contentOf(message);
	if (!Array.isArray(content)) return [];
	return content.filter(
		(block): block is { type: "toolCall"; id: string } =>
			isRecord(block) && block.type === "toolCall" && typeof block.id === "string",
	);
}
function toolResultId(message: AgentMessage): string | undefined {
	return messageRole(message) === "toolResult" && isRecord(message) && typeof message.toolCallId === "string"
		? message.toolCallId
		: undefined;
}
function hasProtectedAssistantContent(message: AgentMessage): boolean {
	const content = contentOf(message);
	if (!Array.isArray(content)) return false;
	return content.some((block) => !isRecord(block) || block.type !== "text" || block.textSignature !== undefined);
}
function validateSourceEntries(entries: readonly SourceEntry[]): SourceAnalysis {
	const seen = new Set<string>();
	for (const [index, entry] of entries.entries()) {
		if (!isRecord(entry) || typeof entry.id !== "string" || !entry.id || seen.has(entry.id)) {
			throw new ContextEditingError(
				"invalid_source",
				`Source entry ${index + 1} has a duplicate or invalid stable ID`,
			);
		}
		seen.add(entry.id);
		assertTextOnlyMessage(entry.message, `Source entry ${entry.id}`);
	}
	const groups: ToolGroup[] = [];
	const incompleteIds = new Set<string>();
	const results = new Map<string, string[]>();
	const callLocations = new Map<string, number>();
	for (const [index, entry] of entries.entries()) {
		for (const call of toolCalls(entry.message)) {
			if (callLocations.has(call.id))
				throw new ContextEditingError("protocol_violation", `Duplicate tool call ID ${call.id}`);
			callLocations.set(call.id, index);
		}
		const id = toolResultId(entry.message);
		if (id) results.set(id, [...(results.get(id) ?? []), entry.id]);
	}
	for (const [toolCallId, ids] of results) {
		if (!callLocations.has(toolCallId))
			throw new ContextEditingError("protocol_violation", `Orphan tool result for ${toolCallId}`);
		if (ids.length !== 1)
			throw new ContextEditingError("protocol_violation", `Tool call ${toolCallId} has ${ids.length} results`);
		const resultIndex = entries.findIndex((entry) => entry.id === ids[0]);
		if (resultIndex <= callLocations.get(toolCallId)!)
			throw new ContextEditingError("protocol_violation", `Tool result for ${toolCallId} precedes its call`);
	}
	for (let i = 0; i < entries.length; i++) {
		const calls = toolCalls(entries[i]!.message);
		if (!calls.length) continue;
		const ids = [entries[i]!.id];
		let complete = true;
		const expected = new Set(calls.map((call) => call.id));
		let j = i + 1;
		while (j < entries.length && expected.size > 0 && messageRole(entries[j]!.message) === "toolResult") {
			ids.push(entries[j]!.id);
			const resultId = toolResultId(entries[j]!.message);
			if (!resultId || !expected.delete(resultId)) complete = false;
			j++;
		}
		if (expected.size > 0) complete = false;
		if (j < entries.length && expected.size > 0) complete = false;
		const group = { ids, complete };
		groups.push(group);
		if (!complete) for (const id of ids) incompleteIds.add(id);
		i = j - 1;
	}
	return { groups, incompleteIds };
}
function sameProtectedFields(before: AgentMessage, after: AgentMessage): boolean {
	if (messageRole(before) !== messageRole(after)) return false;
	if (messageRole(before) === "assistant") {
		const left = clone(before as unknown as Record<string, unknown>);
		const right = clone(after as unknown as Record<string, unknown>);
		delete left.content;
		delete right.content;
		if (canonicalJson(left) !== canonicalJson(right)) return false;
		const beforeContent = contentOf(before);
		const afterContent = contentOf(after);
		if (typeof beforeContent === "string" || typeof afterContent === "string")
			return typeof beforeContent === "string" && typeof afterContent === "string";
		if (!Array.isArray(beforeContent) || !Array.isArray(afterContent) || beforeContent.length !== afterContent.length)
			return false;
		return beforeContent.every((block, index) => {
			const replacement = afterContent[index];
			if (!isRecord(block) || !isRecord(replacement) || block.type !== replacement.type) return false;
			if (block.type !== "text") return canonicalJson(block) === canonicalJson(replacement);
			if (block.textSignature !== undefined) return canonicalJson(block) === canonicalJson(replacement);
			const protectedBefore = clone(block);
			const protectedAfter = clone(replacement);
			delete protectedBefore.text;
			delete protectedAfter.text;
			return canonicalJson(protectedBefore) === canonicalJson(protectedAfter);
		});
	}
	return false;
}
function replaceText(message: AgentMessage, text: string): AgentMessage {
	if (!text.trim()) throw new ContextEditingError("invalid_proposal", "Replacement text cannot be empty");
	const copy = clone(message as unknown as Record<string, unknown>);
	if (typeof copy.content === "string") copy.content = text;
	else if (Array.isArray(copy.content)) {
		const indexes = copy.content
			.map((block, index) => (isRecord(block) && block.type === "text" ? index : -1))
			.filter((index) => index >= 0);
		if (indexes.length !== 1)
			throw new ContextEditingError("protocol_violation", "Only one text block can be edited at a time");
		const target = copy.content[indexes[0]!] as Record<string, unknown>;
		if (target.textSignature !== undefined)
			throw new ContextEditingError("protocol_violation", "Signed assistant text cannot be edited");
		target.text = text;
	} else throw new ContextEditingError("invalid_proposal", "Message has no editable text");
	return copy as unknown as AgentMessage;
}
function assertExactKeys(record: Record<string, unknown>, allowed: readonly string[], label: string): void {
	const allowedSet = new Set(allowed);
	const unknown = Object.keys(record).find((key) => !allowedSet.has(key));
	if (unknown) throw new ContextEditingError("invalid_proposal", `${label} contains unknown field ${unknown}`);
}

function assertProposal(
	value: unknown,
	sessionId: string,
	revision: number,
	effectiveDigest: string,
): ContextEditProposal {
	if (
		!isRecord(value) ||
		(value.schemaVersion !== undefined && value.schemaVersion !== 1) ||
		(value.proposalId !== undefined &&
			(typeof value.proposalId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.proposalId))) ||
		value.sessionId !== sessionId ||
		!Number.isInteger(value.baseRevision) ||
		typeof value.baseDigest !== "string" ||
		!Array.isArray(value.edits) ||
		(value.hardReset !== undefined && typeof value.hardReset !== "boolean") ||
		(value.resetReason !== undefined && (typeof value.resetReason !== "string" || !value.resetReason.trim()))
	) {
		throw new ContextEditingError(
			"invalid_proposal",
			"Proposal must contain schemaVersion, sessionId, baseRevision, baseDigest, and edits",
		);
	}
	assertExactKeys(
		value,
		["schemaVersion", "proposalId", "sessionId", "baseRevision", "baseDigest", "edits", "hardReset", "resetReason"],
		"Proposal",
	);
	for (const [index, edit] of value.edits.entries()) {
		if (!isRecord(edit) || typeof edit.op !== "string")
			throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} is malformed`);
		if (edit.op === "replace") {
			assertExactKeys(edit, ["op", "sourceId", "message"], `Edit ${index + 1}`);
			if (typeof edit.sourceId !== "string" || !edit.sourceId || !isRecord(edit.message))
				throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} replacement is malformed`);
		} else if (edit.op === "replace_text") {
			assertExactKeys(edit, ["op", "sourceId", "text"], `Edit ${index + 1}`);
			if (typeof edit.sourceId !== "string" || !edit.sourceId || typeof edit.text !== "string" || !edit.text.trim())
				throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} text replacement is malformed`);
		} else if (edit.op === "delete") {
			assertExactKeys(edit, ["op", "sourceId"], `Edit ${index + 1}`);
			if (typeof edit.sourceId !== "string" || !edit.sourceId)
				throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} deletion is malformed`);
		} else if (edit.op === "summary") {
			assertExactKeys(edit, ["op", "sourceIds", "text"], `Edit ${index + 1}`);
			if (
				!Array.isArray(edit.sourceIds) ||
				edit.sourceIds.length === 0 ||
				edit.sourceIds.some((id) => typeof id !== "string" || !id) ||
				new Set(edit.sourceIds).size !== edit.sourceIds.length ||
				typeof edit.text !== "string" ||
				!edit.text.trim()
			) {
				throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} summary is malformed`);
			}
		} else {
			throw new ContextEditingError("invalid_proposal", `Edit ${index + 1} has unsupported operation ${edit.op}`);
		}
	}
	if (value.baseRevision !== revision || value.baseDigest !== effectiveDigest)
		throw new ContextEditingError(
			"stale_proposal",
			`Proposal is stale (base ${value.baseRevision}/${value.baseDigest}; current ${revision}/${effectiveDigest})`,
		);
	return value as unknown as ContextEditProposal;
}

function proposalIdentity(value: unknown, fileDigest: string): string {
	if (
		isRecord(value) &&
		typeof value.sessionId === "string" &&
		typeof value.proposalId === "string" &&
		/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.proposalId)
	) {
		return `proposal-id:${digest({ sessionId: value.sessionId, proposalId: value.proposalId })}`;
	}
	return `legacy-file:${fileDigest}`;
}

export class ContextEditingController {
	private readonly config: ContextEditConfig;
	private readonly store: DurableContextStore;
	private state: DurableContextState;
	private stagingReceipt: ContextEditReceipt | undefined;
	private poisoned = false;
	private closed = false;

	constructor(config: ContextEditingControllerOptions) {
		const directory = config.directory ?? config.storageDir;
		if (!config.sessionId || !directory)
			throw new ContextEditingError("invalid_config", "Context editing requires sessionId and durable directory");
		this.config = { ...config, directory };
		validateBudget(this.config);
		const opened = DurableContextStore.open(directory, config.sessionId, config.storeFaultInjector);
		this.store = opened.store;
		this.state = opened.state;
		try {
			if (
				config.servingFingerprint !== undefined &&
				this.state.servingFingerprint !== undefined &&
				config.servingFingerprint !== this.state.servingFingerprint
			) {
				throw new ContextEditingError(
					"protocol_violation",
					"Serving fingerprint differs from durable context state; perform an explicit hard reset",
				);
			}
			const fingerprint = config.servingFingerprint ?? this.state.servingFingerprint;
			if (opened.reopened || fingerprint !== this.state.servingFingerprint) {
				const previousEpoch = this.state.cacheEpoch;
				const nextEpoch = opened.reopened ? `epoch-${randomUUID()}` : previousEpoch;
				const nextState: DurableContextState = {
					...this.state,
					cacheEpoch: nextEpoch,
					servingFingerprint: fingerprint,
					lastReset: opened.reopened
						? {
								reason: "cold resume",
								at: new Date().toISOString(),
								previousEpoch,
								newEpoch: nextEpoch,
							}
						: this.state.lastReset,
				};
				this.store.commitStateForController(nextState);
				this.state = nextState;
			}
		} catch (error) {
			this.store.close();
			throw error;
		}
	}

	get stagingPath(): string {
		return this.store.stagingPath;
	}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.store.close();
	}

	dispose(): void {
		this.close();
	}

	getStatus(): ContextEditingStatus {
		return this.status;
	}

	getStagingPath(): string {
		return this.store.stagingPath;
	}

	get status(): ContextEditingStatus {
		return {
			sessionId: this.state.sessionId,
			revision: this.state.revision,
			cacheEpoch: this.state.cacheEpoch,
			sourceCount: this.state.sourceEntries.length,
			sourceFrontier: this.state.sourceFrontier,
			sourceDigest: this.state.sourceDigest,
			effectiveDigest: this.state.effectiveDigest,
			stagingPath: this.store.stagingPath,
			protocolPath: this.store.protocolPath,
			stagingReceipt: this.stagingReceipt ?? this.state.lastReceipt,
		};
	}

	stageProposal(proposal: ContextEditProposal): ContextEditReceipt {
		this.assertUsable();
		const base = {
			id: `receipt-${randomUUID()}`,
			at: new Date().toISOString(),
			revision: this.state.revision,
		};
		const identified = proposal.proposalId ? proposal : { ...proposal, proposalId: `proposal-${randomUUID()}` };
		let valid: ContextEditProposal;
		try {
			valid = assertProposal(identified, this.state.sessionId, this.state.revision, this.state.effectiveDigest);
		} catch (error) {
			const receipt: ContextEditReceipt = {
				...base,
				accepted: false,
				phase: "rejected",
				error: error instanceof Error ? error.message : String(error),
			};
			this.stagingReceipt = receipt;
			return receipt;
		}
		try {
			this.store.writeMarkdownProtocol(valid);
			this.store.writeProposal(valid);
		} catch (error) {
			this.poisoned = true;
			throw error;
		}
		const receipt: ContextEditReceipt = {
			...base,
			accepted: true,
			phase: "staged",
			proposalId: valid.proposalId,
			baseRevision: valid.baseRevision,
			operationCount: valid.edits.length,
		};
		this.stagingReceipt = receipt;
		return receipt;
	}

	prepare(sourceEntries: readonly SourceEntry[], override: ContextBudgetOverride = {}): RequestSnapshot {
		this.assertUsable();
		validateBudget(override);
		if (override.servingFingerprint !== undefined && override.servingFingerprint !== this.state.servingFingerprint) {
			if (this.state.servingFingerprint !== undefined) {
				throw new ContextEditingError(
					"protocol_violation",
					"Serving fingerprint changed; perform a hard reset before preparing context",
				);
			}
		}
		const nextFingerprint = override.servingFingerprint ?? this.state.servingFingerprint;
		const budget = { ...this.config, ...override };
		const analysis = validateSourceEntries(sourceEntries);
		if (analysis.incompleteIds.size > 0)
			throw new ContextEditingError(
				"protocol_violation",
				"Cannot prepare context with an incomplete tool-call/result batch",
			);
		this.store.verifySourceInput(sourceEntries);
		const current = this.state.sourceEntries;
		if (sourceEntries.length < current.length)
			throw new ContextEditingError("source_divergence", "Source frontier moved backwards");
		for (let i = 0; i < current.length; i++) {
			const supplied = sourceEntries[i];
			const committed = current[i];
			if (
				!supplied ||
				!committed ||
				supplied.id !== committed.id ||
				canonicalJson(supplied.message) !== canonicalJson(committed.message)
			) {
				throw new ContextEditingError(
					"source_divergence",
					`Source entry ${supplied?.id ?? i} diverges from durable raw source`,
				);
			}
		}

		const appended = sourceEntries.slice(current.length).map(clone);
		const fullSource = sourceEntries.map(clone);
		let effective = this.state.effectiveEntries.map(clone);
		for (const entry of appended)
			effective.push({ id: entry.id, sourceIds: [entry.id], message: clone(entry.message) });

		let proposalRecord = this.store.readProposal();
		let pendingProposalIdentity = proposalRecord
			? proposalIdentity(proposalRecord.value, proposalRecord.fileDigest)
			: undefined;
		if (proposalRecord && pendingProposalIdentity === this.state.handledProposalIdentity) {
			if (digest(proposalRecord.value) !== this.state.handledProposalContentDigest)
				throw new ContextEditingError("invalid_proposal", "Proposal ID was reused with different proposal content");
			this.consumeCommittedProposal(proposalRecord.fileDigest);
			proposalRecord = undefined;
			pendingProposalIdentity = undefined;
		}
		let proposal: ContextEditProposal | undefined;
		let proposalError: string | undefined;
		let stagedHardReset = false;
		if (proposalRecord) {
			try {
				proposal = assertProposal(
					proposalRecord.value,
					this.state.sessionId,
					this.state.revision,
					this.state.effectiveDigest,
				);
				const finalSourceId = fullSource[fullSource.length - 1]?.id;
				const liveGroup = finalSourceId
					? analysis.groups.find((group) => group.ids[group.ids.length - 1] === finalSourceId)
					: undefined;
				effective = this.applyProposal(effective, fullSource, analysis, proposal, new Set(liveGroup?.ids ?? []));
				stagedHardReset = proposal.hardReset === true;
			} catch (error) {
				proposalError = error instanceof Error ? error.message : String(error);
				effective = this.state.effectiveEntries.map(clone);
				for (const entry of appended)
					effective.push({ id: entry.id, sourceIds: [entry.id], message: clone(entry.message) });
			}
		}

		const estimatedTokens = estimateTokens(
			effective.map((entry) => entry.message),
			budget.fixedPromptTokens ?? 0,
		);
		const reservedOutputTokens = budget.maxOutputTokens ?? 0;
		if (budget.maxPromptTokens !== undefined && estimatedTokens + reservedOutputTokens > budget.maxPromptTokens) {
			throw new ContextEditingError(
				"budget_exceeded",
				`Effective context estimate ${estimatedTokens} plus output reserve ${reservedOutputTokens} exceeds budget ${budget.maxPromptTokens}`,
			);
		}

		const nextRevision =
			canonicalJson(effective) === canonicalJson(this.state.effectiveEntries)
				? this.state.revision
				: this.state.revision + 1;
		let receipt: ContextEditReceipt | undefined;
		if (proposalRecord) {
			receipt = {
				id: `receipt-${randomUUID()}`,
				at: new Date().toISOString(),
				accepted: proposalError === undefined,
				phase: proposalError === undefined ? "committed" : "rejected",
				proposalId: proposal?.proposalId,
				baseRevision: proposal?.baseRevision,
				revision: nextRevision,
				operationCount: proposal?.edits.length,
				error: proposalError,
			};
		}
		const previousEpoch = this.state.cacheEpoch;
		const nextEpoch = stagedHardReset ? `epoch-${randomUUID()}` : previousEpoch;
		const nextState: DurableContextState = {
			...this.state,
			revision: nextRevision,
			cacheEpoch: nextEpoch,
			sourceEntries: clone(fullSource),
			effectiveEntries: clone(effective),
			sourceDigest: sourceDigest(fullSource),
			sourceFrontier: fullSource[fullSource.length - 1]?.id,
			effectiveDigest: effectiveDigest(effective),
			servingFingerprint: nextFingerprint,
			lastReceipt: receipt ?? this.state.lastReceipt,
			handledProposalIdentity: pendingProposalIdentity ?? this.state.handledProposalIdentity,
			handledProposalContentDigest: proposalRecord
				? digest(proposalRecord.value)
				: this.state.handledProposalContentDigest,
			lastReset: stagedHardReset
				? {
						reason: proposal?.resetReason ?? "staged hard reset",
						at: new Date().toISOString(),
						previousEpoch,
						newEpoch: nextEpoch,
					}
				: this.state.lastReset,
		};
		const changed =
			appended.length > 0 ||
			canonicalJson(effective) !== canonicalJson(this.state.effectiveEntries) ||
			nextEpoch !== previousEpoch ||
			nextFingerprint !== this.state.servingFingerprint ||
			proposalRecord !== undefined;
		if (changed) this.commit(nextState);
		if (receipt) this.stagingReceipt = receipt;
		if (proposalRecord) this.consumeCommittedProposal(proposalRecord.fileDigest);
		return this.buildSnapshot(this.state.effectiveEntries, estimatedTokens, reservedOutputTokens);
	}

	private assertUsable(): void {
		if (this.closed) throw new ContextEditingError("protocol_violation", "Context controller is closed");
		if (this.poisoned)
			throw new ContextEditingError(
				"corrupt_state",
				"Context controller is poisoned after an ambiguous durable write; close and reopen it",
			);
	}

	private commit(nextState: DurableContextState): void {
		try {
			this.store.commit(nextState);
			this.state = nextState;
		} catch (error) {
			this.poisoned = true;
			throw error;
		}
	}

	private consumeCommittedProposal(proposalDigest: string): void {
		try {
			this.store.consumeProposal(proposalDigest);
		} catch (error) {
			this.poisoned = true;
			throw error;
		}
	}

	private buildSnapshot(
		effective: readonly EffectiveEntry[],
		estimatedTokens = estimateTokens(
			effective.map((entry) => entry.message),
			this.config.fixedPromptTokens ?? 0,
		),
		reservedOutputTokens = this.config.maxOutputTokens ?? 0,
	): RequestSnapshot {
		const notices = [
			"Token estimate is a conservative UTF-8 byte and structure upper bound; it is not a tokenizer count.",
		];
		const receipt = this.stagingReceipt ?? this.state.lastReceipt;
		if (receipt?.accepted === false && receipt.error) notices.push(`Rejected staged edit: ${receipt.error}`);
		return deepFreeze({
			messages: clone(effective).map((entry) => entry.message),
			revision: this.state.revision,
			cacheEpoch: this.state.cacheEpoch,
			digest: this.state.effectiveDigest,
			sourceFrontier: this.state.sourceFrontier,
			sourceDigest: this.state.sourceDigest,
			estimatedTokens,
			reservedOutputTokens,
			notices,
		});
	}

	private applyProposal(
		effective: EffectiveEntry[],
		source: SourceEntry[],
		analysis: SourceAnalysis,
		proposal: ContextEditProposal,
		liveIds: Set<string>,
	): EffectiveEntry[] {
		const current = effective.map(clone);
		const sourceMap = new Map(source.map((entry) => [entry.id, entry]));
		const groups = analysis.groups;
		for (const edit of proposal.edits) {
			const ids = edit.op === "summary" ? [...edit.sourceIds] : [edit.sourceId];
			if (!ids.length || ids.some((id) => !sourceMap.has(id)))
				throw new ContextEditingError("invalid_proposal", "Proposal references an unknown stable source ID");
			if (ids.some((id) => analysis.incompleteIds.has(id)))
				throw new ContextEditingError(
					"protocol_violation",
					"The live tool exchange is incomplete and cannot be edited",
				);
			if (ids.some((id) => liveIds.has(id)))
				throw new ContextEditingError(
					"protocol_violation",
					"The current live tool exchange is protected until the next request boundary",
				);
			if (ids.some((id) => messageRole(sourceMap.get(id)!.message) === "user"))
				throw new ContextEditingError("protocol_violation", "User instructions are protected and cannot be edited");
			const touchedGroups = groups.filter((group) => ids.some((id) => group.ids.includes(id)));
			if (
				(edit.op === "summary" && touchedGroups.length !== 1) ||
				(touchedGroups.length > 0 &&
					(touchedGroups.length !== 1 ||
						edit.op !== "summary" ||
						touchedGroups[0]!.ids.length !== ids.length ||
						touchedGroups[0]!.ids.some((id) => !ids.includes(id))))
			)
				throw new ContextEditingError(
					"protocol_violation",
					"Tool calls and all matching results must be edited as one complete group",
				);
			const indexes = ids.map((id) => current.findIndex((entry) => entry.sourceIds.includes(id)));
			if (indexes.some((index) => index < 0))
				throw new ContextEditingError("invalid_proposal", "Source entry was already removed or summarized");
			if (edit.op === "summary") {
				const ordered = [...indexes].sort((a, b) => a - b);
				if (ordered[ordered.length - 1]! - ordered[0]! + 1 !== ordered.length)
					throw new ContextEditingError("protocol_violation", "Summary IDs must be contiguous");
				const at = ordered[0]!;
				if (ids.some((id, offset) => !current[ordered[offset]!]!.sourceIds.includes(id)))
					throw new ContextEditingError("protocol_violation", "Summary IDs must follow source order");
				const original = current[at]!.message as unknown as Record<string, unknown>;
				const summaryMessage = clone(original);
				summaryMessage.content = [
					{ type: "text", text: `[Context memory replacing ${ids.join(", ")}]\n${edit.text}` },
				];
				current.splice(at, ordered.length, {
					id: `summary-${randomUUID()}`,
					sourceIds: ids,
					message: summaryMessage as unknown as AgentMessage,
				});
			} else {
				const index = indexes[0]!;
				if (current[index]!.sourceIds.length !== 1)
					throw new ContextEditingError("invalid_proposal", "A summarized group cannot be edited by source ID");
				if (edit.op === "delete") {
					if (hasProtectedAssistantContent(current[index]!.message))
						throw new ContextEditingError(
							"protocol_violation",
							"Signed, reasoning, and native assistant content cannot be deleted directly",
						);
					current.splice(index, 1);
				} else {
					const replacement =
						edit.op === "replace_text" ? replaceText(current[index]!.message, edit.text) : edit.message;
					assertTextOnlyMessage(replacement, "Proposed replacement");
					if (!sameProtectedFields(current[index]!.message, replacement))
						throw new ContextEditingError(
							"protocol_violation",
							"Replacement may not mutate role, identity, tool IDs, or native metadata",
						);
					current[index] = { ...current[index]!, message: clone(replacement) };
				}
			}
		}
		return current;
	}

	hardReset(reason = "explicit hard reset", servingFingerprint?: string): RequestSnapshot {
		this.assertUsable();
		if (servingFingerprint !== undefined && !servingFingerprint)
			throw new ContextEditingError("invalid_config", "Serving fingerprint cannot be empty");
		const previousEpoch = this.state.cacheEpoch;
		const nextEpoch = `epoch-${randomUUID()}`;
		const nextState: DurableContextState = {
			...this.state,
			cacheEpoch: nextEpoch,
			servingFingerprint: servingFingerprint ?? this.state.servingFingerprint,
			lastReset: {
				reason,
				at: new Date().toISOString(),
				previousEpoch,
				newEpoch: nextEpoch,
			},
		};
		try {
			this.store.commitStateForController(nextState);
			this.state = nextState;
		} catch (error) {
			this.poisoned = true;
			throw error;
		}
		return this.buildSnapshot(this.state.effectiveEntries);
	}

	reset(reason?: string): RequestSnapshot {
		return this.hardReset(reason ?? "explicit reset");
	}
}

function validateBudget(
	config: Pick<ContextEditConfig, "maxPromptTokens" | "fixedPromptTokens" | "maxOutputTokens">,
): void {
	for (const name of ["maxPromptTokens", "fixedPromptTokens", "maxOutputTokens"] as const) {
		const value = config[name];
		if (value !== undefined && (!Number.isFinite(value) || !Number.isInteger(value) || value < 0)) {
			throw new ContextEditingError("invalid_config", `${name} must be a finite non-negative integer`);
		}
	}
}

function estimateTokens(messages: readonly AgentMessage[], fixedPromptTokens: number): number {
	const serialized = canonicalJson(messages);
	const utf8Bytes = new TextEncoder().encode(serialized).length;
	return Math.max(0, fixedPromptTokens) + utf8Bytes + messages.length * 8;
}
