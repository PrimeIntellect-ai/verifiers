import { Context, Effect } from "effect";

export default {
  id: "verifiers.mcp-ready",
  effect: Effect.fn(function* (ctx) {
    // ACP registrations bypass OpenCode's configured MCP server timeout defaults.
    yield* ctx.mcp.transform((editor) => {
      for (const name of ctx.options.servers) {
        editor.update(name, (server) => {
          server.timeout = { ...server.timeout, execution: ctx.options.timeout };
        });
      }
    });
    yield* ctx.session.hook(
      "prompt",
      Effect.fn(function* () {
        // OpenCode 2.0.26's public plugin API omits discovered MCP tools. Read
        // its native service in the prompt's location context, including empty servers.
        const mcp = yield* Context.Service()("@opencode/MCP");
        const servers = yield* mcp.servers();
        for (const name of ctx.options.servers) {
          const server = servers.find((server) => server.name === name);
          if (server?.status.status !== "connected") {
            return yield* Effect.die(
              new Error(`MCP server ${name} is not connected: ${JSON.stringify(server?.status)}`),
            );
          }
        }
        const expected = (yield* mcp.tools())
          .filter((tool) => ctx.options.servers.includes(tool.server))
          .map((tool) => `${tool.server}_${tool.name}`.replace(/[^a-zA-Z0-9_-]/g, "_"));
        // ACP session/new awaits connection, but OpenCode refreshes its tool
        // registry asynchronously. Do not let the prompt snapshot an old catalog.
        yield* Effect.gen(function* () {
          while (true) {
            const registered = new Set((yield* ctx.tool.list()).map((tool) => tool.id));
            if (expected.every((name) => registered.has(name))) return;
            yield* Effect.sleep("10 millis");
          }
        }).pipe(Effect.timeout(ctx.options.timeout), Effect.orDie);
      }),
    );
  }),
};
