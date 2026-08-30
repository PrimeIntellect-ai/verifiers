/** Route OpenClaw's awaited tool boundaries through Verifiers' ACP extension. */

import { randomUUID } from "node:crypto";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// {tool_content}

const pluginId = "verifiers-tool-interception";
const unavailable = "Tool interception is unavailable.";
const replaced = "Tool result replaced by interception.";
const stateKey = Symbol.for("verifiers.toolInterception");
const state = (globalThis[stateKey] ??= {
  pending: new Map(),
  queued: [],
  waiting: [],
});

async function intercept(pending, queued, waiting, phase, toolCallId, name, content) {
  if (typeof toolCallId !== "string" || !toolCallId) {
    throw new TypeError("OpenClaw omitted the tool call ID");
  }
  const separator = toolCallId.lastIndexOf("|fc_");
  const policyToolCallId = separator > 0 ? toolCallId.slice(0, separator) : toolCallId;
  const id = randomUUID();
  const response = new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
  const request = {
    id,
    body: {
      phase,
      message: {
        role: "tool",
        tool_call_id: policyToolCallId,
        content,
        name,
      },
    },
  };
  const waiter = waiting.shift();
  if (waiter) waiter(request);
  else queued.push(request);
  const timeout = setTimeout(() => {
    pending.get(id)?.reject(new Error("LiveACPClient timed out"));
    pending.delete(id);
    const position = queued.findIndex((item) => item.id === id);
    if (position >= 0) queued.splice(position, 1);
  }, 30_000);
  try {
    return validateToolDecision(await response, "LiveACPClient");
  } finally {
    clearTimeout(timeout);
    pending.delete(id);
  }
}

export default definePluginEntry({
  id: pluginId,
  name: "Verifiers tool interception",
  register(api) {
    const blockedCalls = new Map();
    const { pending, queued, waiting } = state;
    let halted = false;
    api.registerGatewayMethod(
      "verifiers.tool_interception.next",
      async ({ respond }) => {
        const request = queued.shift() ?? (await new Promise((resolve) => waiting.push(resolve)));
        respond(true, request);
      },
      { scope: "operator.write" },
    );
    api.registerGatewayMethod(
      "verifiers.tool_interception.resolve",
      ({ params, respond }) => {
        const request = pending.get(params.id);
        if (request) {
          if (typeof params.error === "string") {
            request.reject(new Error(params.error));
          } else request.resolve(params.decision);
        }
        respond(true, { resolved: Boolean(request) });
      },
      { scope: "operator.write" },
    );
    api.on(
      "before_tool_call",
      async (event) => {
        const toolCallId = event.toolCallId;
        if (halted) {
          if (toolCallId) blockedCalls.set(toolCallId, null);
          return { block: true, blockReason: unavailable };
        }
        let decision;
        try {
          decision = await intercept(
            pending,
            queued,
            waiting,
            "before",
            toolCallId,
            event.toolName,
            "",
          );
        } catch (error) {
          console.error(`${unavailable} ${error}`);
          process.exit(70);
        }
        if (halted) {
          blockedCalls.set(toolCallId, null);
          return { block: true, blockReason: unavailable };
        }
        if (decision.action === "allow") return undefined;
        if (decision.action === "stop") halted = true;
        blockedCalls.set(
          toolCallId,
          decision.action === "rewrite" ? decision.message.content : null,
        );
        return {
          block: true,
          blockReason:
            decision.action === "rewrite"
              ? replaced
              : decision.reason || "Rollout terminated by interception.",
        };
      },
      { timeoutMs: 35_000 },
    );

    api.registerAgentToolResultMiddleware(
      async (event) => {
        const toolCallId = event.toolCallId;
        if (blockedCalls.has(toolCallId)) {
          const replacement = blockedCalls.get(toolCallId);
          blockedCalls.delete(toolCallId);
          return replacement === null
            ? undefined
            : {
                result: {
                  ...event.result,
                  content: hostToolContent(replacement, "OpenClaw"),
                },
              };
        }
        if (halted) return undefined;
        let decision;
        try {
          decision = await intercept(
            pending,
            queued,
            waiting,
            "after",
            toolCallId,
            event.toolName,
            vfToolContent(event.result.content, "OpenClaw"),
          );
        } catch (error) {
          console.error(`${unavailable} ${error}`);
          process.exit(70);
        }
        if (halted || decision.action === "allow") return undefined;
        if (decision.action === "stop") {
          halted = true;
          return undefined;
        }
        return {
          result: {
            ...event.result,
            content: hostToolContent(decision.message.content, "OpenClaw"),
          },
        };
      },
      { runtimes: ["openclaw"] },
    );
  },
});
