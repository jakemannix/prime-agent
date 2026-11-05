import type { AgentMessage } from "@earendil-works/pi-agent-core";

export const CONTEXT_EDITING_SCHEMA_VERSION = 1;
export const CONTEXT_EDITING_SERIALIZER = "prime-context-editing-json-v1";

/** A journal message paired with the session-manager entry ID that owns it. */
export interface SourceEntry {
	readonly id: string;
	readonly message: AgentMessage;
}

export interface ContextEditConfig {
	readonly sessionId: string;
	/** Session-owned durable directory. It must not be shared by sessions. */
	readonly directory?: string;
	readonly storageDir?: string;
	readonly maxPromptTokens?: number;
	/** Fixed system/tool/template overhead excluded from the editable messages. */
	readonly fixedPromptTokens?: number;
	/** Output headroom reserved when checking maxPromptTokens. */
	readonly maxOutputTokens?: number;
	readonly servingFingerprint?: string;
}

export type ContextStoreFaultPoint = "before_manifest_swap" | "after_manifest_swap" | "after_proposal_rename";
export type ContextStoreFaultInjector = (point: ContextStoreFaultPoint) => void;

export type ContextEditOperation =
	| { readonly sourceId: string; readonly op: "replace"; readonly message: AgentMessage }
	| { readonly sourceId: string; readonly op: "replace_text"; readonly text: string }
	| { readonly sourceId: string; readonly op: "delete" }
	| { readonly sourceIds: readonly string[]; readonly op: "summary"; readonly text: string };

export interface ContextBudgetOverride {
	readonly maxPromptTokens?: number;
	readonly fixedPromptTokens?: number;
	readonly maxOutputTokens?: number;
	readonly servingFingerprint?: string;
}

export interface ContextEditProposal {
	readonly schemaVersion?: number;
	/** Stable operation identity. Reuse for retry; generate a fresh value for a new intended operation. */
	readonly proposalId?: string;
	readonly sessionId: string;
	readonly baseRevision: number;
	readonly baseDigest: string;
	readonly edits: readonly ContextEditOperation[];
	/** Request an epoch rotation at the next request boundary without reverting edits. */
	readonly hardReset?: boolean;
	readonly resetReason?: string;
}

export interface ContextEditReceipt {
	readonly id: string;
	/** True for a syntactically staged proposal or a durably committed edit. See phase. */
	readonly accepted: boolean;
	readonly phase?: "staged" | "committed" | "rejected";
	readonly at: string;
	readonly proposalId?: string;
	readonly baseRevision?: number;
	readonly revision: number;
	readonly error?: string;
	readonly operationCount?: number;
}

export interface EffectiveEntry {
	readonly id: string;
	readonly sourceIds: readonly string[];
	readonly message: AgentMessage;
}

export interface RequestSnapshot {
	readonly messages: readonly AgentMessage[];
	readonly revision: number;
	readonly cacheEpoch: string;
	readonly digest: string;
	readonly sourceFrontier?: string;
	readonly sourceDigest: string;
	/** Conservative estimate of the serialized effective prompt, including fixed overhead. */
	readonly estimatedTokens: number;
	readonly reservedOutputTokens: number;
	readonly notices: readonly string[];
}

export interface ContextEditingStatus {
	readonly sessionId: string;
	readonly revision: number;
	readonly cacheEpoch: string;
	readonly sourceCount: number;
	readonly sourceFrontier?: string;
	readonly sourceDigest: string;
	readonly effectiveDigest: string;
	readonly stagingPath: string;
	readonly protocolPath: string;
	readonly stagingReceipt?: ContextEditReceipt;
}

export interface ContextEditingControllerOptions extends ContextEditConfig {
	/** Deterministic durability fault injection. Tests only. */
	readonly storeFaultInjector?: ContextStoreFaultInjector;
}

export class ContextEditingError extends Error {
	readonly code:
		| "invalid_config"
		| "corrupt_state"
		| "session_mismatch"
		| "source_divergence"
		| "invalid_source"
		| "stale_proposal"
		| "invalid_proposal"
		| "budget_exceeded"
		| "protocol_violation";

	constructor(code: ContextEditingError["code"], message: string) {
		super(message);
		this.name = "ContextEditingError";
		this.code = code;
	}
}
