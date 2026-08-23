/** Route OpenClaw's awaited tool boundaries through rollout policy. */

import { readFileSync, rmSync } from "node:fs";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

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

function messageContent(content) {
  if (!Array.isArray(content) || content.length === 0) {
    throw new TypeError("Invalid OpenClaw tool result");
  }
  const parts = content.map((part) => {
    if (part?.type === "text" && typeof part.text === "string") {
      return { type: "text", text: part.text };
    }
    if (
      part?.type === "image" &&
      typeof part.mimeType === "string" &&
      typeof part.data === "string"
    ) {
      return {
        type: "image_url",
        image_url: { url: `data:${part.mimeType};base64,${part.data}` },
      };
    }
    throw new TypeError("Unsupported OpenClaw tool result content");
  });
  if (!parts.every((part) => part.type === "text")) return parts;
  const text = parts.map((part) => part.text).join("\n");
  if (!text) throw new TypeError("OpenClaw returned empty tool result content");
  return text;
}

function openClawContent(content) {
  const parts = typeof content === "string" ? [{ type: "text", text: content }] : content;
  if (
    !Array.isArray(parts) ||
    parts.length === 0 ||
    parts.every((part) => part?.type === "text" && !part.text)
  ) {
    throw new TypeError("Interception returned empty tool result content");
  }
  return parts.map((part) => {
    if (part?.type === "text" && typeof part.text === "string") {
      return { type: "text", text: part.text };
    }
    const url = part?.type === "image_url" ? part.image_url?.url : undefined;
    const image =
      typeof url === "string"
        ? /^data:(image\/[^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/.exec(url)
        : null;
    if (!image) throw new TypeError("OpenClaw requires inline base64 tool images");
    return { type: "image", mimeType: image[1], data: image[2] };
  });
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
  const decision = await response.json();
  if (!["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new TypeError("Tool interception returned an invalid decision");
  }
  if (decision.action === "rewrite" && !decision.message) {
    throw new TypeError("Tool interception omitted the replacement result");
  }
  return decision;
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
                  content: openClawContent(replacement),
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
            messageContent(event.result.content),
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
            content: openClawContent(decision.message.content),
          },
        };
      },
      { runtimes: ["openclaw"] },
    );
  },
});
