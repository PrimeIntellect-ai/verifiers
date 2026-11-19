import { Effect, Schema } from "effect";
import { writeFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

// The plugin runtime exposes Effect, but not the private schema package.
// Native hooks identify this recoverable error by its public Tool.Error tag.
class ToolError extends Schema.TaggedError()("Tool.Error", { message: Schema.String }) {}

// A transform replaces every executor, including subsequently registered MCP
// tools. A split rollout never delegates an allowed call back to the local tool.
export default {
  id: "verifiers.tool-gate",
  effect: Effect.fn(function* (ctx) {
    const gate = (id, name, input) => Effect.tryPromise(() => new Promise((resolve, reject) => {
      // Native fetch has a header deadline. The host answers after executing B's
      // tool, so the rollout and tool deadlines must own the wait instead.
      const request = (ctx.options.url.startsWith("https:") ? httpsRequest : httpRequest)(ctx.options.url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${ctx.options.secret}`,
        },
      }, (response) => {
        if (response.statusCode !== 200) {
          response.resume();
          reject(new Error(`Tool gate failed (${response.statusCode})`));
          return;
        }
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("error", reject);
        response.on("end", () => {
          try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
          catch (error) { reject(error); }
        });
      });
      request.on("error", reject);
      request.end(JSON.stringify({ tool_call_id: id, name, arguments: input }));
    })).pipe(Effect.orDie);
    if (ctx.options.remote) {
      // Hooks also cover executors added or overwritten by later plugins. A
      // denial is a normal native tool error; verifiers pins the actual remote
      // ToolMessage when the transcript crosses the next model boundary.
      yield* ctx.tool.hook("execute.before", Effect.fn(function* (event) {
        const verdict = yield* gate(event.id, event.tool, event.input);
        if (verdict.action !== "deny") {
          return yield* Effect.die(new Error("Split tool gate did not return a remote result"));
        }
        const content = verdict.message?.content;
        return yield* Effect.fail(new ToolError({
          message: typeof content === "string" ? content : JSON.stringify(content),
        }));
      }));
    }
    yield* ctx.tool.transform((editor) => {
      for (const registered of editor.list()) {
        editor.update(registered.id, (tool) => {
          const execute = tool.execute;
          tool.execute = (input, context) => Effect.gen(function* () {
            const verdict = yield* gate(context.id, registered.id, input);
            if (verdict.action === "deny") {
              const content = verdict.message?.content;
              if (typeof content !== "string") {
                return { content: (content ?? []).map((part) => part.type === "text"
                  ? part : { type: "file", uri: part.image_url.url, mime: part.image_url.url.split(";")[0].slice(5) }) };
              }
              return { content };
            }
            if (verdict.action !== "allow" || ctx.options.remote) {
              return yield* Effect.die(new Error("Split tool gate did not return a remote result"));
            }
            return yield* execute(input, context);
          });
        });
      }
    });
    writeFileSync(ctx.options.readyFile, "ready");
  }),
};
