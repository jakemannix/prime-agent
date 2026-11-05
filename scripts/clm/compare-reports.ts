import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	compareReplayReports,
	type ReplayComparisonReport,
	type ReplayReport,
} from "./replay.js";

function parseSavedReport(text: string, label: "exact" | "scr"): ReplayReport {
	let value: unknown;
	try {
		value = JSON.parse(text);
	} catch (error) {
		throw new Error(`Invalid ${label} replay report JSON: ${String(error)}`);
	}
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`Saved ${label} replay report must be an object`);
	}
	const report = value as Partial<ReplayReport>;
	if (report.schemaVersion !== 1 || !report.config || !Array.isArray(report.results)) {
		throw new Error(`Saved ${label} replay report is missing schemaVersion, config, or results`);
	}
	if (report.config.mode !== label) {
		throw new Error(`Saved ${label} replay report has mode '${String(report.config.mode)}'`);
	}
	return report as ReplayReport;
}

/** Compare two already-completed runs without contacting either serving endpoint. */
export function compareSavedReports(exact: ReplayReport, scr: ReplayReport): ReplayComparisonReport {
	return compareReplayReports(exact, scr);
}

export async function compareSavedReportFiles(
	exactPath: string,
	scrPath: string,
	outputPath?: string,
): Promise<ReplayComparisonReport> {
	const [exactText, scrText] = await Promise.all([readFile(exactPath, "utf8"), readFile(scrPath, "utf8")]);
	const comparison = compareSavedReports(parseSavedReport(exactText, "exact"), parseSavedReport(scrText, "scr"));
	if (outputPath) await writeFile(outputPath, `${JSON.stringify(comparison, null, 2)}\n`);
	return comparison;
}

function arg(args: string[], name: string): string | undefined {
	const index = args.indexOf(name);
	return index >= 0 ? args[index + 1] : undefined;
}

async function main(argv: string[]): Promise<void> {
	const exactPath = arg(argv, "--exact");
	const scrPath = arg(argv, "--scr");
	const outputPath = arg(argv, "--output");
	if (!exactPath || !scrPath) {
		throw new Error("Usage: compare-reports.ts --exact EXACT_REPORT.json --scr SCR_REPORT.json [--output FILE]");
	}
	const comparison = await compareSavedReportFiles(
		resolve(exactPath),
		resolve(scrPath),
		outputPath ? resolve(outputPath) : undefined,
	);
	if (!outputPath) process.stdout.write(`${JSON.stringify(comparison, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	void main(process.argv.slice(2)).catch((error: unknown) => {
		process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
		process.exitCode = 1;
	});
}
