/** Bridge Pi's native tool hooks to the rollout's /tool policy. */

import { readFileSync, unlinkSync } from "node:fs";

async function intercept(
  phase,
  toolCallId,
  name,
  content,
  url,
  secret,
  signal,
) {
  const timeout = AbortSignal.timeout(30_000);
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${secret}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phase,
      content: "nonempty_text",
      message: {
        role: "tool",
        tool_call_id: toolCallId,
        content,
        name,
      },
    }),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!response.ok) throw new Error(`tool interception returned ${response.status}`);
  const decision = await response.json();
  if (!["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new Error("tool interception returned an invalid action");
  }
  if (decision.action === "rewrite" && !decision.message) {
    throw new Error("tool interception omitted the rewritten result");
  }
  return decision;
}

function piContent(content) {
  if (content.every((part) => part.type === "text")) {
    return content.map((part) => part.text).join("\n");
  }
  return content.map((part) =>
    part.type === "text"
      ? { type: "text", text: part.text }
      : {
          type: "image_url",
          image_url: { url: `data:${part.mimeType};base64,${part.data}` },
        },
  );
}

export default function toolInterceptionExtension(pi) {
  const credentialsPath = process.env.VF_TOOL_INTERCEPTION_CONFIG;
  delete process.env.VF_TOOL_INTERCEPTION_CONFIG;
  const credentialsJson = readFileSync(credentialsPath, "utf8");
  unlinkSync(credentialsPath);
  const { url, secret } = JSON.parse(credentialsJson);
  if (typeof url !== "string" || typeof secret !== "string") {
    throw new Error("Tool interception configuration is unavailable");
  }
  const preReplacements = new Map();

  pi.on("message_end", (event) => {
    const message = event.message;
    if (message.role !== "toolResult") return undefined;
    const toolCallId = message.toolCallId.split("|", 1)[0];
    if (!preReplacements.has(toolCallId)) return undefined;
    const content = preReplacements.get(toolCallId);
    preReplacements.delete(toolCallId);
    return {
      message: {
        ...message,
        content: [{ type: "text", text: content }],
        isError: false,
      },
    };
  });

  pi.on("tool_call", async (event, ctx) => {
    // Pi's Responses provider appends its item id after `|`; the model call id
    // before it is the identity stored in the Verifiers trace.
    const toolCallId = event.toolCallId.split("|", 1)[0];
    let decision;
    try {
      decision = await intercept(
        "before",
        toolCallId,
        event.toolName,
        "",
        url,
        secret,
        ctx.signal,
      );
    } catch (error) {
      console.error("Tool interception failed:", error);
      ctx.abort();
      process.exit(1);
    }
    if (decision.action === "allow") return undefined;
    if (decision.action === "stop") ctx.abort();
    if (decision.action === "rewrite") {
      preReplacements.set(toolCallId, decision.message.content);
    }
    return {
      block: true,
      reason:
        decision.action === "rewrite"
          ? decision.message.content
          : decision.reason || "Rollout terminated by interception.",
    };
  });

  pi.on("tool_result", async (event, ctx) => {
    const toolCallId = event.toolCallId.split("|", 1)[0];
    const content = piContent(event.content) || "(no tool output)";
    let decision;
    try {
      decision = await intercept(
        "after",
        toolCallId,
        event.toolName,
        content,
        url,
        secret,
        ctx.signal,
      );
    } catch (error) {
      console.error("Tool interception failed:", error);
      ctx.abort();
      process.exit(1);
    }
    if (decision.action === "allow") {
      if (content !== "(no tool output)") return undefined;
      return {
        content: [{ type: "text", text: content }],
        isError: event.isError,
      };
    }
    if (decision.action === "stop") {
      ctx.abort();
      return {
        content: [
          {
            type: "text",
            text: decision.reason || "Rollout terminated by interception.",
          },
        ],
        isError: true,
      };
    }
    return {
      content: [{ type: "text", text: decision.message.content }],
      isError: event.isError,
    };
  });
}
