import { createHash } from "node:crypto";

/** The two deliberately separate serving policies supported by the adapter. */
export type SglangAdapterMode = "exact" | "scr";

export interface SglangAdapterIdentity {
	sessionId: string;
	cacheEpoch: string | number;
	revision: string | number;
	requestId: string;
}

export interface SglangAdapterConfig {
	/** The dedicated OpenAI-compatible SGLang base URL. */
	baseUrl: string;
	/** Model allowlist entry. Injection is never allowed without this explicit value. */
	model: string;
	/** Selects an explicitly configured exact or SCR endpoint. */
	mode: SglangAdapterMode;
	/** Operator-pinned tokenizer/template/parser/precision/server identity. */
	servingFingerprint: string;
	/** Hostnames permitted in addition to local development hosts. */
	allowedHosts?: readonly string[];
}

export interface SglangEndpointValidation {
	url: URL;
	mode: SglangAdapterMode;
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/;

function assertIdentifier(name: string, value: string | number): string {
	const text = String(value);
	if (!text || text.length > 256 || !SAFE_ID.test(text)) {
		throw new Error(`Invalid SGLang ${name}; use a short opaque identifier`);
	}
	return text;
}

function digest(parts: readonly string[], prefix: string): string {
	return `${prefix}${createHash("sha256").update(parts.join("\0"), "utf8").digest("hex").slice(0, 32)}`;
}

/**
 * Validate the configured endpoint without probing it. A successful HTTP response
 * does not establish SCR capability; capability negotiation belongs to the server
 * launcher/runtime and is intentionally outside this adapter.
 */
export function validateSglangEndpoint(config: SglangAdapterConfig): SglangEndpointValidation {
	if (config.mode !== "exact" && config.mode !== "scr")
		throw new Error("SGLang mode must be exactly 'exact' or 'scr'");
	if (
		typeof config.servingFingerprint !== "string" ||
		!config.servingFingerprint ||
		config.servingFingerprint.length > 512
	)
		throw new Error("SGLang servingFingerprint must be a nonempty bounded string");
	if (typeof config.baseUrl !== "string" || config.baseUrl.length > 2048) throw new Error("Invalid SGLang base URL");
	let url: URL;
	try {
		url = new URL(config.baseUrl);
	} catch {
		throw new Error("SGLang base URL must be an absolute URL");
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("SGLang base URL must use http or https");
	if (url.username || url.password || url.search || url.hash)
		throw new Error("SGLang base URL must not contain credentials, query, or fragment");
	const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
	const allowed = new Set((config.allowedHosts ?? []).map((entry) => entry.toLowerCase().replace(/^\[|\]$/g, "")));
	if (!LOCAL_HOSTS.has(host) && !allowed.has(host)) {
		throw new Error(`SGLang host '${host}' is not allowlisted`);
	}
	if (url.protocol === "http:" && !LOCAL_HOSTS.has(host)) {
		throw new Error("Remote SGLang endpoints must use HTTPS");
	}
	if (config.model === undefined) throw new Error("SGLang adapter requires an explicit model allowlist entry");
	assertIdentifier("model", config.model);
	return { url: new URL(url), mode: config.mode };
}

/**
 * Add SGLang metadata to a final OpenAI-compatible payload.
 *
 * This function is pure: it clones the payload, keeps user fields intact, and
 * only writes top-level cache_salt/rid. It does not probe or assert SCR support.
 */
export function applySglangContextMetadata(
	payload: unknown,
	identity: SglangAdapterIdentity,
	config: SglangAdapterConfig,
): Record<string, unknown> {
	validateSglangEndpoint(config);
	if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
		throw new Error("SGLang adapter requires an object OpenAI payload");
	}
	const session = assertIdentifier("session ID", identity.sessionId);
	const epoch = assertIdentifier("cache epoch", identity.cacheEpoch);
	const revision = assertIdentifier("revision", identity.revision);
	const request = assertIdentifier("request ID", identity.requestId);
	const clone = structuredClone(payload) as Record<string, unknown>;
	if (Object.hasOwn(clone, "extra_body")) {
		const extra = clone.extra_body;
		if (extra !== undefined && extra !== null && typeof extra !== "object") {
			throw new Error("SGLang payload extra_body must not contain adapter metadata");
		}
		if (extra && !Array.isArray(extra) && (Object.hasOwn(extra, "cache_salt") || Object.hasOwn(extra, "rid"))) {
			throw new Error("SGLang metadata must be top-level, not nested in extra_body");
		}
	}
	if (Object.hasOwn(clone, "cache_salt") || Object.hasOwn(clone, "rid")) {
		throw new Error("SGLang payload already has cache metadata; refusing a second cache owner");
	}
	if (clone.model !== config.model) {
		throw new Error(`SGLang payload model '${String(clone.model)}' does not match configured model`);
	}
	clone.cache_salt = digest(["prime-agent", session, epoch, config.mode, config.servingFingerprint], "prime-");
	clone.rid = digest(
		["prime-agent-request", session, epoch, revision, request, config.mode, config.servingFingerprint],
		"prime-rid-",
	);
	return clone;
}
