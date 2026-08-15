/** Bridge native Claude Code and Pi hooks to the rollout's /tool policy. */

import { readFileSync, unlinkSync } from "node:fs";

// Claude persists larger text after PostToolUse, beyond the hook's synchronous boundary.
const CLAUDE_MAX_INLINE_TEXT = 50_000;
const CLAUDE_BASH_MAX_INLINE_TEXT = 30_000;
const CLAUDE_TEXT_REWRITE = {
  content: "text",
  max_text_utf16_units: CLAUDE_MAX_INLINE_TEXT,
  allow_empty_text: false,
};
const CLAUDE_BASH_REWRITE = {
  ...CLAUDE_TEXT_REWRITE,
  max_text_utf16_units: CLAUDE_BASH_MAX_INLINE_TEXT,
};
const CLAUDE_MCP_REWRITE = {
  max_text_utf16_units: CLAUDE_MAX_INLINE_TEXT,
  allow_empty_text: false,
  allow_empty_parts: false,
  preserve_single_text_part: false,
};

function contentText(content) {
  if (typeof content === "string") return content;
  return content
    .map((part) => (part.type === "text" ? part.text : JSON.stringify(part)))
    .join("\n");
}

async function intercept(
  phase,
  toolCallId,
  name,
  content,
  rewrite,
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
      rewrite,
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

function claudeContent(content, toolName) {
  if (typeof content === "string") {
    if (content.length > CLAUDE_MAX_INLINE_TEXT) {
      throw new Error(`Claude will persist ${toolName}'s output after its hook`);
    }
    return content.trim() ? content : `(${toolName} completed with no output)`;
  }
  if (!Array.isArray(content)) {
    throw new Error(`Claude returned unsupported output for ${toolName}`);
  }
  if (
    content.length === 0 ||
    content.every((part) => part.type === "text" && !part.text?.trim())
  ) {
    return `(${toolName} completed with no output)`;
  }
  const converted = content.map((part) => {
    if (part.type === "text" && typeof part.text === "string") {
      return { type: "text", text: part.text };
    }
    if (part.type === "image" && part.source?.type === "base64") {
      return {
        type: "image_url",
        image_url: {
          url: `data:${part.source.media_type};base64,${part.source.data}`,
        },
      };
    }
    if (part.type === "image" && part.source?.type === "url") {
      return {
        type: "image_url",
        image_url: { url: part.source.url },
      };
    }
    throw new Error(`Claude returned unsupported output for ${toolName}`);
  });
  if (
    converted.every((part) => part.type === "text") &&
    converted.reduce((size, part) => size + part.text.length, 0) >
      CLAUDE_MAX_INLINE_TEXT
  ) {
    throw new Error(`Claude will persist ${toolName}'s output after its hook`);
  }
  return converted.length === 1 && converted[0].type === "text"
    ? converted[0].text
    : converted;
}

function claudeToolContent(hook) {
  if (typeof hook.tool_response === "string") {
    return claudeContent(hook.tool_response, hook.tool_name);
  }
  if (
    hook.tool_name === "Bash" &&
    hook.tool_response &&
    typeof hook.tool_response === "object" &&
    !Array.isArray(hook.tool_response) &&
    typeof hook.tool_response.stdout === "string" &&
    hook.tool_response.isImage === false &&
    hook.tool_response.persistedOutputPath === undefined
  ) {
    return claudeContent(hook.tool_response.stdout, hook.tool_name);
  }
  if (hook.tool_name.startsWith("mcp__")) {
    return claudeContent(hook.tool_response, hook.tool_name);
  }
  throw new Error(`Claude returned unsupported output for ${hook.tool_name}`);
}

function claudeToolOutput(hook, content) {
  if (typeof hook.tool_response === "string") {
    if (
      typeof content !== "string" ||
      !content.trim() ||
      content.length > CLAUDE_MAX_INLINE_TEXT
    ) {
      throw new Error(`Claude cannot replace ${hook.tool_name}'s output`);
    }
    return content;
  }
  if (hook.tool_name.startsWith("mcp__")) {
    if (typeof content === "string") {
      if (!content.trim() || content.length > CLAUDE_MAX_INLINE_TEXT) {
        throw new Error("Claude cannot preserve the rewritten MCP output");
      }
      return content;
    }
    if (
      content.length === 0 ||
      (content.length === 1 && content[0].type === "text")
    ) {
      throw new Error("Claude cannot preserve the rewritten MCP content shape");
    }
    if (
      content.reduce(
        (size, part) => size + (part.type === "text" ? part.text.length : 0),
        0,
      ) > CLAUDE_MAX_INLINE_TEXT
    ) {
      throw new Error("Claude cannot preserve the rewritten MCP output");
    }
    return content.map((part) => {
      if (part.type === "text") return part;
      const url = part.image_url.url;
      const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      return {
        type: "image",
        source: match
          ? { type: "base64", media_type: match[1], data: match[2] }
          : { type: "url", url },
      };
    });
  }
  if (
    hook.tool_name === "Bash" &&
    hook.tool_response &&
    typeof hook.tool_response === "object" &&
    !Array.isArray(hook.tool_response)
  ) {
    if (
      typeof content !== "string" ||
      !content.trim() ||
      content.length > CLAUDE_BASH_MAX_INLINE_TEXT
    ) {
      throw new Error("Claude cannot preserve the rewritten Bash output");
    }
    return { ...hook.tool_response, stdout: content, stderr: "" };
  }
  throw new Error(`Claude cannot replace ${hook.tool_name}'s structured output`);
}

function nativeDecision(hook, decision) {
  if (decision.action === "allow") return undefined;
  const reason = decision.reason || "Rollout terminated by interception.";
  if (decision.action === "stop") {
    return { continue: false, stopReason: reason };
  }

  const content = decision.message.content;
  if (hook.hook_event_name === "PreToolUse") {
    return {
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: contentText(content),
      },
    };
  }
  return {
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      updatedToolOutput: claudeToolOutput(hook, content),
    },
  };
}

async function runClaudeHook(hook, url, secret, signal) {
  try {
    const before = hook.hook_event_name === "PreToolUse";
    const failed = hook.hook_event_name === "PostToolUseFailure";
    const content = before
      ? ""
      : failed
        ? hook.error || "Tool execution failed."
        : claudeToolContent(hook);
    const decision = await intercept(
      before ? "before" : failed ? "after_failure" : "after",
      hook.tool_use_id,
      hook.tool_name,
      content,
      failed
        ? null
        : before
          ? { content: "text" }
          : hook.tool_name === "Bash"
            ? CLAUDE_BASH_REWRITE
            : hook.tool_name.startsWith("mcp__")
              ? CLAUDE_MCP_REWRITE
              : CLAUDE_TEXT_REWRITE,
      url,
      secret,
      signal,
    );
    return nativeDecision(hook, decision) || {};
  } catch {
    return {
      continue: false,
      stopReason: "Tool interception is unavailable.",
    };
  }
}

export function createClaudeToolHook(url, secret) {
  return (hook, _toolUseId, options) =>
    runClaudeHook(hook, url, secret, options?.signal);
}

function piContent(content) {
  const parts = Array.isArray(content) ? content : [{ type: "text", text: String(content) }];
  const converted = parts.map((part) => {
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

function toPiContent(content) {
  if (typeof content === "string") return [{ type: "text", text: content }];
  return content.map((part) => {
    if (part.type === "text") return part;
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
    return match
      ? { type: "image", mimeType: match[1], data: match[2] }
      : { type: "text", text: part.image_url.url };
  });
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
        { content: "text" },
        url,
        secret,
        ctx.signal,
      );
    } catch {
      ctx.abort();
      return {
        block: true,
        reason: "Tool interception is unavailable.",
      };
    }
    if (decision.action === "allow") return undefined;
    if (decision.action === "stop") ctx.abort();
    return {
      block: true,
      reason:
        decision.action === "rewrite"
          ? contentText(decision.message.content)
          : decision.reason || "Rollout terminated by interception.",
    };
  });

  pi.on("tool_result", async (event, ctx) => {
    const toolCallId = event.toolCallId.split("|", 1)[0];
    let decision;
    try {
      decision = await intercept(
        event.isError ? "after_failure" : "after",
        toolCallId,
        event.toolName,
        piContent(event.content),
        { image_urls: "data" },
        url,
        secret,
        ctx.signal,
      );
    } catch {
      ctx.abort();
      return {
        content: [{ type: "text", text: "Tool interception is unavailable." }],
        isError: true,
      };
    }
    if (decision.action === "allow") return undefined;
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
      content: toPiContent(decision.message.content),
      isError: event.isError,
    };
  });
}
