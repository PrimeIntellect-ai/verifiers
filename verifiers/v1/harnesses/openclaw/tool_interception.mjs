/** Route OpenClaw's awaited tool boundaries to the rollout's ACP runner. */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// {tool_content}
// {tool_socket}

const pluginId = "verifiers-tool-interception";
const unavailable = "Tool interception is unavailable.";
const replaced = "Tool result replaced by interception.";

async function intercept(phase, toolCallId, name, content) {
  if (typeof toolCallId !== "string" || !toolCallId) {
    throw new TypeError("OpenClaw omitted the tool call ID");
  }
  const separator = toolCallId.lastIndexOf("|fc_");
  const policyToolCallId = separator > 0 ? toolCallId.slice(0, separator) : toolCallId;
  return requestToolPolicy({
    phase,
    message: {
      role: "tool",
      tool_call_id: policyToolCallId,
      content,
      name,
    },
  });
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
      // The socket fails closed at 40s, before the native hook times out.
      { timeoutMs: 45_000 },
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
