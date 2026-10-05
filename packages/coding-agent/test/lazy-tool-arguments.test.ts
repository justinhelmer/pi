import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AssistantMessage, normalizeContext } from "@earendil-works/pi-ai";
import * as parser from "@earendil-works/pi-ai/utils/json-parse";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { type JsonAgentSessionEvent, toJsonEvent } from "../src/modes/json-event.ts";
import { serializeJsonLine } from "../src/modes/rpc/jsonl.ts";

const model = {
	id: "fixture",
	name: "fixture",
	api: "openai-responses" as const,
	provider: "openai",
	baseUrl: "https://fixture.invalid/v1",
	reasoning: false,
	input: ["text" as const],
	contextWindow: 128 * 1024,
	maxTokens: 1024,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
function toolEvents(args: string): unknown[] {
	const item = { type: "function_call", id: "fc_fixture", call_id: "call_fixture", name: "fixture", arguments: "" };
	const events: unknown[] = [{ type: "response.output_item.added", output_index: 0, item }];
	for (let at = 0; at < args.length; at += 8)
		events.push({ type: "response.function_call_arguments.delta", output_index: 0, delta: args.slice(at, at + 8) });
	events.push({ type: "response.function_call_arguments.done", output_index: 0, arguments: args });
	events.push({ type: "response.output_item.done", output_index: 0, item: { ...item, arguments: args } });
	events.push({ type: "response.completed", response: { status: "completed", output: [] } });
	return events;
}
const textEvents = [
	{ type: "response.output_item.added", output_index: 0, item: { type: "message", id: "msg_fixture" } },
	{
		type: "response.output_item.done",
		output_index: 0,
		item: { type: "message", id: "msg_fixture", content: [{ type: "output_text", text: "done" }] },
	},
	{ type: "response.completed", response: { status: "completed", output: [] } },
];
async function makeSession(sharedCwd?: string) {
	const cwd = sharedCwd ?? (await mkdtemp(join(tmpdir(), "pi-native-lazy-fixture-")));
	const settingsManager = SettingsManager.inMemory({
		cacheWarming: "off",
		compaction: { enabled: false },
		retry: { enabled: false },
	});
	const sessionManager = SessionManager.inMemory(cwd);
	const resourceLoader = new DefaultResourceLoader({
		cwd,
		agentDir: cwd,
		settingsManager,
		noExtensions: true,
		noSkills: true,
		noPromptTemplates: true,
		noThemes: true,
		noContextFiles: true,
	});
	await resourceLoader.reload();
	const credentials = AuthStorage.inMemory({ openai: { type: "api_key", key: "sk-fixture-only" } });
	const modelRuntime = await ModelRuntime.create({
		credentials,
		modelsPath: null,
		refreshOnCreate: false,
		allowModelNetwork: false,
	});
	const created = await createAgentSession({
		cwd,
		agentDir: cwd,
		model,
		modelRuntime,
		settingsManager,
		sessionManager,
		resourceLoader,
		noTools: "all",
	});
	return {
		...created,
		modelRuntime,
		dispose: async () => {
			created.session.dispose();
			if (!sharedCwd) await rm(cwd, { recursive: true, force: true });
		},
	};
}
afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
	vi.unstubAllEnvs();
});
function assistantMessages(events: JsonAgentSessionEvent[]): AssistantMessage[] {
	return events.flatMap((event) =>
		event.type === "message_end" && event.message.role === "assistant" ? [event.message] : [],
	);
}
// Regression for https://github.com/earendil-works/pi/issues/9062.
describe("stock pi native lazy arguments", () => {
	it("preserves complete factory RPC records against the native eager path", async () => {
		vi.stubEnv("PI_OFFLINE", "1");
		const args = JSON.stringify({ command: 'quoted "text"\nseparator\u2028emoji😀' });
		const normalizeTimes = (value: unknown): unknown => {
			if (Array.isArray(value)) return value.map(normalizeTimes);
			if (value && typeof value === "object") {
				return Object.fromEntries(
					Object.entries(value)
						.filter(([key]) => key !== "timestamp")
						.map(([key, item]) => [key, normalizeTimes(item)]),
				);
			}
			return value;
		};
		const records: unknown[][] = [];
		const cwd = await mkdtemp(join(tmpdir(), "pi-native-lazy-parity-"));
		try {
			for (const eager of [true, false]) {
				let requests = 0;
				vi.stubGlobal(
					"fetch",
					vi.fn(async (url: unknown) => {
						expect(String(url)).toBe("https://fixture.invalid/v1/responses");
						const events = ++requests === 1 ? toolEvents(args) : textEvents;
						return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
							headers: { "content-type": "text/event-stream" },
						});
					}),
				);
				const runtime = await makeSession(cwd);
				if (eager) {
					const native = runtime.session.agent.streamFunction;
					runtime.session.agent.streamFunction = (model, context, options) =>
						native(model, context, { ...options, lazyToolArguments: false });
				}
				const rpc: unknown[] = [];
				const unsubscribe = runtime.session.subscribe((event) => {
					// These are the public model/tool/turn records; session-entry IDs are unrelated to parsing.
					if (
						event.type.startsWith("message_") ||
						event.type.startsWith("tool_execution_") ||
						event.type.startsWith("turn_") ||
						event.type === "agent_start" ||
						event.type === "agent_end"
					) {
						rpc.push(normalizeTimes(JSON.parse(serializeJsonLine(toJsonEvent(event)))));
					}
				});
				try {
					await runtime.session.prompt("fixture", { expandPromptTemplates: false });
					expect(requests).toBe(2);
					records.push(rpc);
				} finally {
					unsubscribe();
					await runtime.dispose();
				}
			}
			expect(records[1]).toEqual(records[0]);
		} finally {
			await rm(cwd, { recursive: true, force: true });
		}
	});
	it("keeps unread prefix work bounded through the real factory, builtin provider and RPC serializer", async () => {
		vi.stubEnv("PI_OFFLINE", "1");
		const args = JSON.stringify({ command: "x".repeat(128 * 1024) });
		let requests = 0;
		const payloads: unknown[] = [];
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: unknown, init: RequestInit) => {
				expect(String(url)).toBe("https://fixture.invalid/v1/responses");
				payloads.push(JSON.parse(init.body as string));
				const events = ++requests === 1 ? toolEvents(args) : textEvents;
				return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					headers: { "content-type": "text/event-stream" },
				});
			}),
		);
		const parse = vi.spyOn(parser, "parseStreamingJson");
		const runtime = await makeSession();
		const stream = vi.spyOn(runtime.modelRuntime, "streamSimple");
		const rpc: JsonAgentSessionEvent[] = [];
		const unsubscribe = runtime.session.subscribe((event) =>
			rpc.push(JSON.parse(serializeJsonLine(toJsonEvent(event)))),
		);
		try {
			await runtime.session.prompt("fixture", { expandPromptTemplates: false });
			expect(requests).toBe(2);
			expect(parse).toHaveBeenCalledTimes(2);
			expect(parse.mock.calls.reduce((total, [text]) => total + (text?.length ?? 0), 0)).toBe(2 * args.length);
			expect(stream.mock.calls.every(([, , options]) => options?.lazyToolArguments === true)).toBe(true);
			expect(payloads.every((payload) => !("lazyToolArguments" in (payload as object)))).toBe(true);
			const deltas = rpc.flatMap((event) =>
				event.type === "message_update" && event.assistantMessageEvent.type === "toolcall_delta" ? [event] : [],
			);
			expect(deltas.length).toBe(Math.ceil(args.length / 8));
			expect(deltas.every((event) => !("message" in event) && !("partial" in event.assistantMessageEvent))).toBe(
				true,
			);
			const messages = assistantMessages(rpc);
			const finalCall = messages[0].content[0];
			if (finalCall.type !== "toolCall") throw new Error("expected a finalized tool call");
			expect(finalCall.arguments).toEqual(JSON.parse(args));
			expect(messages.at(-1)?.content).toMatchObject([{ type: "text", text: "done" }]);
		} finally {
			unsubscribe();
			await runtime.dispose();
		}
	});
	it("preserves explicit eager opt-out on the session's native stream function", async () => {
		vi.stubEnv("PI_OFFLINE", "1");
		const args = JSON.stringify({ command: "x".repeat(16 * 1024) });
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: unknown) => {
				expect(String(url)).toBe("https://fixture.invalid/v1/responses");
				return new Response(
					toolEvents(args)
						.map((event) => `data: ${JSON.stringify(event)}\n\n`)
						.join(""),
					{ headers: { "content-type": "text/event-stream" } },
				);
			}),
		);
		const parse = vi.spyOn(parser, "parseStreamingJson");
		const runtime = await makeSession();
		const spy = vi.spyOn(runtime.modelRuntime, "streamSimple");
		try {
			const options = { apiKey: "sk-fixture-only", lazyToolArguments: false };
			const stream = await runtime.session.agent.streamFunction(
				model,
				normalizeContext({
					messages: [{ role: "user", content: [{ type: "text", text: "fixture" }], timestamp: 0 }],
				}),
				options,
			);
			const result = await stream.result();
			expect(result.stopReason).toBe("toolUse");
			const finalCall = result.content[0];
			if (finalCall.type !== "toolCall") throw new Error("expected a finalized tool call");
			expect(finalCall.arguments).toEqual(JSON.parse(args));
			expect(parse).toHaveBeenCalledTimes(Math.ceil(args.length / 8) + 2);
			expect(spy.mock.calls[0][2]?.lazyToolArguments).toBe(false);
		} finally {
			await runtime.dispose();
		}
	});
	it("allows a native partial-argument subscriber to read every SDK snapshot", async () => {
		vi.stubEnv("PI_OFFLINE", "1");
		const args = JSON.stringify({ command: "x".repeat(8 * 1024) });
		let requests = 0;
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: unknown) => {
				expect(String(url)).toBe("https://fixture.invalid/v1/responses");
				const events = ++requests === 1 ? toolEvents(args) : textEvents;
				return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					headers: { "content-type": "text/event-stream" },
				});
			}),
		);
		const parse = vi.spyOn(parser, "parseStreamingJson");
		const runtime = await makeSession();
		let reads = 0;
		const unsubscribe = runtime.session.subscribe((event) => {
			if (
				event.type === "message_update" &&
				event.message.role === "assistant" &&
				event.assistantMessageEvent.type === "toolcall_delta"
			) {
				const block = event.message.content[event.assistantMessageEvent.contentIndex];
				if (block.type === "toolCall") {
					reads++;
					void JSON.stringify(block.arguments);
				}
			}
		});
		try {
			await runtime.session.prompt("fixture", { expandPromptTemplates: false });
			expect(requests).toBe(2);
			expect(reads).toBe(Math.ceil(args.length / 8));
			expect(parse.mock.calls.length).toBeGreaterThan(2);
			expect(parse.mock.calls.length).toBeLessThanOrEqual(reads + 2);
			expect(parse.mock.calls.reduce((total, [text]) => total + (text?.length ?? 0), 0)).toBeGreaterThan(
				(args.length * args.length) / 32,
			);
		} finally {
			unsubscribe();
			await runtime.dispose();
		}
	});
	it("retains a malformed argument event as a terminal error without another request", async () => {
		vi.stubEnv("PI_OFFLINE", "1");
		let requests = 0;
		const events = [
			{
				type: "response.output_item.added",
				output_index: 0,
				item: { type: "function_call", id: "fc_fixture", call_id: "call_fixture", name: "fixture", arguments: "" },
			},
			{ type: "response.function_call_arguments.delta", output_index: 0, delta: '{"command":"partial' },
			{ type: "response.function_call_arguments.done", output_index: 0, arguments: 1 },
			{ type: "response.completed", response: { status: "completed", output: [] } },
		];
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: unknown) => {
				expect(String(url)).toBe("https://fixture.invalid/v1/responses");
				requests++;
				return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					headers: { "content-type": "text/event-stream" },
				});
			}),
		);
		const runtime = await makeSession();
		const rpc: JsonAgentSessionEvent[] = [];
		const unsubscribe = runtime.session.subscribe((event) =>
			rpc.push(JSON.parse(serializeJsonLine(toJsonEvent(event)))),
		);
		try {
			await runtime.session.prompt("fixture", { expandPromptTemplates: false });
			expect(requests).toBe(1);
			const terminal = assistantMessages(rpc).at(-1);
			expect(terminal?.stopReason).toBe("error");
			expect(terminal?.errorMessage).toContain("trim");
			expect(rpc.some((event) => event.type === "tool_execution_start")).toBe(false);
		} finally {
			unsubscribe();
			await runtime.dispose();
		}
	});
});
