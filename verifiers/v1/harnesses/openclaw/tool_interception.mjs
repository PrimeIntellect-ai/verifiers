/** Route OpenClaw's awaited tool boundaries through Verifiers' ACP extension. */

import { randomUUID } from "node:crypto";
import { createConnection } from "node:net";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// {tool_content}

const pluginId = "verifiers-tool-interception";
const unavailable = "Tool interception is unavailable.";
const replaced = "Tool result replaced by interception.";
const stateKey = Symbol.for("verifiers.toolInterception");
const state = (globalThis[stateKey] ??= {
  socketPath: process.env.VF_OPENCLAW_TOOL_SOCKET,
  socket: null,
  connecting: null,
  failure: null,
  buffer: "",
  pending: new Map(),
});
delete process.env.VF_OPENCLAW_TOOL_SOCKET;

function failConnection(error) {
  state.failure ??= error;
  state.socket = null;
  for (const request of state.pending.values()) request.reject(state.failure);
  state.pending.clear();
}

async function adapterSocket() {
  if (state.failure) throw state.failure;
  if (state.socket) return state.socket;
  if (!state.socketPath) throw new Error("OpenClaw interception socket is missing");
  if (!state.connecting) {
    state.connecting = new Promise((resolve, reject) => {
      const deadline = Date.now() + 30_000;
      const connect = () => {
        const socket = createConnection(state.socketPath);
        socket.once("connect", () => {
          socket.removeAllListeners("error");
          socket.setEncoding("utf8");
          socket.on("data", (chunk) => {
            state.buffer += chunk;
            let newline;
            while ((newline = state.buffer.indexOf("\n")) >= 0) {
              const response = JSON.parse(state.buffer.slice(0, newline));
              state.buffer = state.buffer.slice(newline + 1);
              const request = state.pending.get(response.id);
              if (request) {
                if (typeof response.error === "string") {
                  request.reject(new Error(response.error));
                } else request.resolve(response.decision);
              }
            }
          });
          socket.on("error", failConnection);
          socket.on("close", () =>
            failConnection(new Error("LiveACPClient closed the interception socket")),
          );
          state.socket = socket;
          resolve(socket);
        });
        socket.once("error", (error) => {
          socket.destroy();
          if (Date.now() < deadline) setTimeout(connect, 50);
          else reject(error);
        });
      };
      connect();
    }).catch((error) => {
      failConnection(error);
      throw error;
    });
  }
  return state.connecting;
}

async function intercept(phase, toolCallId, name, content) {
  if (typeof toolCallId !== "string" || !toolCallId) {
    throw new TypeError("OpenClaw omitted the tool call ID");
  }
  const separator = toolCallId.lastIndexOf("|fc_");
  const policyToolCallId = separator > 0 ? toolCallId.slice(0, separator) : toolCallId;
  const id = randomUUID();
  const socket = await adapterSocket();
  const response = new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
  });
  socket.write(`${JSON.stringify({
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
  })}\n`);
  const timeout = setTimeout(() => {
    state.pending.get(id)?.reject(new Error("LiveACPClient timed out"));
    state.pending.delete(id);
  }, 30_000);
  try {
    return validateToolDecision(await response, "LiveACPClient");
  } finally {
    clearTimeout(timeout);
    state.pending.delete(id);
  }
}

export default definePluginEntry({
  id: pluginId,
  name: "Verifiers tool interception",
  register(api) {
    const blockedCalls = new Map();
    let halted = false;
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
