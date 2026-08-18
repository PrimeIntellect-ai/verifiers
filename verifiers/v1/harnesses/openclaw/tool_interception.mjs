/** Bridge OpenClaw's native tool hooks to the rollout's /tool policy.
 *
 * `before_tool_call` gates execution; the tool-result middleware runs the
 * post-execution policy and applies pre-execution replacements, so the result
 * OpenClaw records is exactly the one the policy approved. A hook that cannot
 * reach interception halts every further tool synchronously — the harness then
 * fails loudly at its next model turn instead of silently diverging.
 */

import { readFileSync, unlinkSync } from "node:fs";

const PLUGIN_ID = "verifiers-tool-interception";
const credentialsPath = process.env.VF_TOOL_INTERCEPTION_CONFIG;
delete process.env.VF_TOOL_INTERCEPTION_CONFIG;
const credentialsJson = readFileSync(credentialsPath, "utf8");
unlinkSync(credentialsPath);
const { url: TOOL_URL, secret: TOOL_SECRET } = JSON.parse(credentialsJson);
if (typeof TOOL_URL !== "string" || typeof TOOL_SECRET !== "string") {
  throw new TypeError("Tool interception configuration is unavailable");
}
const UNAVAILABLE = "Tool interception is unavailable.";
const REPLACED = "Tool result replaced by interception.";

function normalizeToolCallId(toolCallId) {
  return toolCallId?.split("|", 1)[0];
}

function toMessageContent(content) {
  if (!Array.isArray(content)) return String(content ?? "");
  const converted = content.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text };
    if (part.type === "image") {
      return {
        type: "image_url",
        image_url: { url: `data:${part.mimeType};base64,${part.data}` },
      };
    }
    return { type: "text", text: JSON.stringify(part) };
  });
  return converted.every((part) => part.type === "text")
    ? converted.map((part) => part.text).join("\n")
    : converted;
}

function toOpenClawContent(content) {
  const parts = typeof content === "string" ? [{ type: "text", text: content }] : content;
  const converted = parts.map((part) => {
    if (part.type === "text") return part;
    const url = part.image_url.url;
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    return match
      ? { type: "image", mimeType: match[1], data: match[2] }
      : { type: "text", text: url };
  });
  return converted.length ? converted : [{ type: "text", text: "" }];
}

async function intercept(phase, toolCallId, name, content) {
  if (!TOOL_URL || !TOOL_SECRET || !toolCallId) throw new Error(UNAVAILABLE);
  const response = await fetch(TOOL_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${TOOL_SECRET}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phase,
      message: {
        role: "tool",
        tool_call_id: toolCallId,
        content,
        name,
      },
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`tool interception returned ${response.status}`);
  const decision = await response.json();
  if (!["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new Error("tool interception returned an invalid action");
  }
  if (
    decision.action === "rewrite" &&
    (!decision.message || !("content" in decision.message))
  ) {
    throw new Error("tool interception omitted the rewritten result");
  }
  return decision;
}

function terminalResult(result, reason, unavailable = false) {
  return {
    ...result,
    content: [{ type: "text", text: reason }],
    details: {
      status: "error",
      toolInterceptionStopped: true,
      ...(unavailable ? { toolInterceptionUnavailable: true } : {}),
    },
  };
}

export default {
  id: PLUGIN_ID,
  name: "Verifiers tool interception",
  register(api) {
    const blockedToolCalls = new Map();
    let halted = false;
    let unavailable = false;
    api.on(
      "before_tool_call",
      async (event) => {
        const toolCallId = normalizeToolCallId(event.toolCallId);
        if (halted) {
          if (toolCallId) blockedToolCalls.set(toolCallId, null);
          return { block: true, blockReason: UNAVAILABLE };
        }
        let decision;
        try {
          decision = await intercept("before", toolCallId, event.toolName, "");
        } catch (error) {
          console.error("Tool interception failed:", error);
          halted = true;
          unavailable = true;
          if (toolCallId) blockedToolCalls.set(toolCallId, null);
          return { block: true, blockReason: UNAVAILABLE };
        }
        if (halted) {
          if (toolCallId) blockedToolCalls.set(toolCallId, null);
          return { block: true, blockReason: UNAVAILABLE };
        }
        if (decision.action === "allow") return undefined;
        if (decision.action === "stop") halted = true;
        if (toolCallId) {
          blockedToolCalls.set(
            toolCallId,
            decision.action === "rewrite" ? decision.message.content : null,
          );
        }
        return {
          block: true,
          blockReason:
            decision.action === "rewrite"
              ? REPLACED
              : decision.reason || "Rollout terminated by interception.",
        };
      },
      { timeoutMs: 35_000 },
    );

    api.registerAgentToolResultMiddleware(
      async (event) => {
        const toolCallId = normalizeToolCallId(event.toolCallId);
        // OpenClaw emits `tool_result` for a pre-execution veto. Apply the saved
        // replacement here without sending the synthetic result through policy twice.
        if (blockedToolCalls.has(toolCallId)) {
          const replacement = blockedToolCalls.get(toolCallId);
          blockedToolCalls.delete(toolCallId);
          if (replacement === null) {
            return unavailable
              ? { result: terminalResult(event.result, UNAVAILABLE, true) }
              : undefined;
          }
          return {
            result: {
              ...event.result,
              content: toOpenClawContent(replacement),
            },
          };
        }
        if (halted) {
          return {
            result: terminalResult(event.result, UNAVAILABLE, unavailable),
          };
        }
        let decision;
        try {
          decision = await intercept(
            "after",
            toolCallId,
            event.toolName,
            toMessageContent(event.result.content),
          );
        } catch (error) {
          console.error("Tool interception failed:", error);
          halted = true;
          unavailable = true;
          return { result: terminalResult(event.result, UNAVAILABLE, true) };
        }
        if (halted) {
          return {
            result: terminalResult(event.result, UNAVAILABLE, unavailable),
          };
        }
        if (decision.action === "allow") return undefined;
        if (decision.action === "stop") {
          halted = true;
          return {
            result: terminalResult(
              event.result,
              decision.reason || "Rollout terminated by interception.",
            ),
          };
        }
        return {
          result: {
            ...event.result,
            content: toOpenClawContent(decision.message.content),
          },
        };
      },
      { runtimes: ["openclaw"] },
    );
  },
};
