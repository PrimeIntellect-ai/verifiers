/** Route OpenClaw's awaited tool boundaries through rollout policy. */

import { readFileSync, rmSync } from "node:fs";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

// {tool_content}

const pluginId = "verifiers-tool-interception";
const unavailable = "Tool interception is unavailable.";
const replaced = "Tool result replaced by interception.";
const credentialsPath = process.env.VF_TOOL_INTERCEPTION_CONFIG;
delete process.env.VF_TOOL_INTERCEPTION_CONFIG;
if (!credentialsPath) throw new TypeError(unavailable);
const credentialsJson = readFileSync(credentialsPath, "utf8");
rmSync(credentialsPath);
const credentials = JSON.parse(credentialsJson);
if (typeof credentials.url !== "string" || typeof credentials.secret !== "string") {
  throw new TypeError(unavailable);
}

async function intercept(phase, toolCallId, name, content) {
  if (typeof toolCallId !== "string" || !toolCallId) {
    throw new TypeError("OpenClaw omitted the tool call ID");
  }
  const separator = toolCallId.lastIndexOf("|fc_");
  const policyToolCallId = separator > 0 ? toolCallId.slice(0, separator) : toolCallId;
  const response = await fetch(credentials.url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${credentials.secret}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phase,
      message: {
        role: "tool",
        tool_call_id: policyToolCallId,
        content,
        name,
      },
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`tool interception returned ${response.status}`);
  return validateToolDecision(await response.json(), "Tool interception");
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
          decision = await intercept("before", toolCallId, event.toolName, "");
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
