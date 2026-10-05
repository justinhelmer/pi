import { afterEach, describe, expect, it, vi } from "vitest";
import { processResponsesStream } from "../src/api/openai-responses-shared.ts";
import type { AssistantMessage, Model } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import * as jsonParser from "../src/utils/json-parse.ts";

type Event = Parameters<typeof processResponsesStream>[0] extends AsyncIterable<infer T> ? T : never;
const model: Model<"openai-responses"> = {
	id: "fixture",
	name: "fixture",
	api: "openai-responses",
	provider: "fixture",
	baseUrl: "",
	reasoning: false,
	input: ["text"],
	contextWindow: 0,
	maxTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const item = { type: "function_call", id: "fc_fixture", call_id: "call_fixture", name: "fixture", arguments: "" };
const added = (argumentsValue: unknown = "") => ({
	type: "response.output_item.added",
	output_index: 0,
	item: { ...item, arguments: argumentsValue },
});
const delta = (value: unknown) => ({ type: "response.function_call_arguments.delta", output_index: 0, delta: value });
const done = (value: unknown) => ({ type: "response.function_call_arguments.done", output_index: 0, arguments: value });
const itemDone = (value: unknown) => ({
	type: "response.output_item.done",
	output_index: 0,
	item: { ...item, arguments: value },
});
const completed = { type: "response.completed", response: { status: "completed", output: [] } };
function* toolEvents(args: string, chunk = 8): Generator<unknown> {
	yield added();
	for (let at = 0; at < args.length; at += chunk) yield delta(args.slice(at, at + chunk));
	yield done(args);
	yield itemDone(args);
	yield completed;
}
async function replay(events: Iterable<unknown>, lazy: boolean | undefined, read = false) {
	const output: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "pending",
		timestamp: 0,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
	let offered = 0;
	let failure: { name: string; message: string } | undefined;
	const emitted: unknown[] = [];
	class Sink extends AssistantMessageEventStream {
		override push(event: Parameters<AssistantMessageEventStream["push"]>[0]): void {
			if (read) emitted.push(JSON.parse(JSON.stringify(event)));
		}
	}
	const input: AsyncIterable<Event> = {
		async *[Symbol.asyncIterator]() {
			for (const event of events) {
				offered++;
				yield event as Event;
			}
		},
	};
	try {
		await processResponsesStream(
			input,
			output,
			new Sink(),
			model,
			lazy === undefined ? undefined : { lazyToolArguments: lazy },
		);
	} catch (error) {
		const err = error as Error;
		failure = { name: err.name, message: err.message };
	}
	return { offered, failure, output: JSON.parse(JSON.stringify(output)), emitted };
}
afterEach(() => vi.restoreAllMocks());
// Regression for https://github.com/earendil-works/pi/issues/9062.
describe("Responses lazy function arguments", () => {
	it("bounds parser work for unread tiny deltas and preserves final arguments", async () => {
		const parse = vi.spyOn(jsonParser, "parseStreamingJson");
		for (const size of [128 * 1024, 256 * 1024]) {
			parse.mockClear();
			const args = JSON.stringify({ command: "x".repeat(size) });
			const result = await replay(toolEvents(args), true);
			expect(result.failure).toBeUndefined();
			expect(result.output.content[0].arguments).toEqual(JSON.parse(args));
			expect(parse).toHaveBeenCalledTimes(2);
			expect(parse.mock.calls.reduce((total, [text]) => total + (text?.length ?? 0), 0)).toBe(2 * args.length);
		}
	});
	it("preserves default eager snapshots and on-read SDK values", async () => {
		for (const args of [
			"{}",
			"[]",
			"null",
			"true",
			"0",
			"1e",
			'"text"',
			'{"a":"partial',
			'{"a":"raw\ncontrol"}',
			'{"a":"bad\\qescape"}',
			'{"nested":[true,null,{"n":-1.25e2,"text":"\\uD83D\\uDE00"}]}',
		]) {
			const events = [...toolEvents(args, 1)];
			const eager = await replay(events, undefined, true);
			expect(await replay(events, false, true)).toEqual(eager);
			expect(await replay(events, true, true)).toEqual(eager);
		}
	});
	it("preserves the first rejected JSON-domain event and fatal output", async () => {
		const values = [undefined, null, false, true, 0, 1, "", "text", [], {}, { a: 1 }];
		for (const a of values)
			for (const d of values) {
				const events = [added(a), delta(d), done(d), itemDone(a), completed];
				expect(await replay(events, true)).toEqual(await replay(events, undefined));
			}
	});
	it("retains unread values for incomplete, unfinished and failed streams", async () => {
		for (const tail of [
			[completed],
			[],
			[
				{
					type: "response.incomplete",
					response: { status: "incomplete", incomplete_details: { reason: "max_output_tokens" }, output: [] },
				},
			],
			[
				{
					type: "response.failed",
					response: { status: "failed", error: { code: "server_error", message: "fixture" } },
				},
			],
		]) {
			const events = [added(), delta('{"a":"partial'), ...tail];
			expect(await replay(events, true)).toEqual(await replay(events, undefined));
		}
	});
});
