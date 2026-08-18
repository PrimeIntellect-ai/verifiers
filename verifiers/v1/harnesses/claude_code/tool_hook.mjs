/** Inject trace-scoped Claude SDK hooks before claude-agent-acp creates its query.
 *
 * PreToolUse gates execution against the rollout's /tool policy; PostToolUse and
 * PostToolUseFailure run the post-execution policy before Claude advances. The
 * content mappings mirror how the pinned Claude Code release renders each result
 * into its next model request, so an approved result survives byte-exact — a
 * shape the hook cannot preserve fails the hook (and the rollout) instead of
 * silently diverging from the canonical trace.
 */

// Claude persists larger text after PostToolUse, beyond the hook's synchronous boundary.
const CLAUDE_MAX_INLINE_TEXT = 50_000;
const CLAUDE_BASH_MAX_INLINE_TEXT = 30_000;

function contentText(content) {
  if (typeof content === "string") return content;
  return content
    .map((part) => (part.type === "text" ? part.text : JSON.stringify(part)))
    .join("\n");
}

async function intercept(phase, toolCallId, name, content, url, secret, signal) {
  const timeout = AbortSignal.timeout(30_000);
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${secret}`,
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
  if (hook.hook_event_name !== "PostToolUse") {
    throw new Error("Claude cannot replace a failed tool result");
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
      before ? "before" : "after",
      hook.tool_use_id,
      hook.tool_name,
      content,
      url,
      secret,
      signal,
    );
    return nativeDecision(hook, decision) || {};
  } catch (error) {
    console.error("Tool interception failed:", error);
    return {
      continue: false,
      stopReason: "Tool interception is unavailable.",
    };
  }
}

function createClaudeToolHook(url, secret) {
  return (hook, _toolUseId, options) =>
    runClaudeHook(hook, url, secret, options?.signal);
}

const { ClaudeAcpAgent } = await import(process.argv[1]);
const originalNewSession = ClaudeAcpAgent.prototype.newSession;

ClaudeAcpAgent.prototype.newSession = function (params) {
  const interception = params._meta?.vfToolInterception;
  if (
    typeof interception?.url !== "string" ||
    typeof interception?.secret !== "string"
  ) {
    throw new Error("Claude tool interception configuration is unavailable");
  }
  delete params._meta.vfToolInterception;

  const callback = createClaudeToolHook(interception.url, interception.secret);
  const claudeCode = params._meta.claudeCode || {};
  const options = claudeCode.options || {};
  const hooks = options.hooks || {};
  const injected = { ...hooks };
  for (const event of ["PreToolUse", "PostToolUse", "PostToolUseFailure"]) {
    injected[event] = [
      ...(hooks[event] || []),
      { hooks: [callback], timeout: 35 },
    ];
  }
  params._meta.claudeCode = {
    ...claudeCode,
    options: { ...options, hooks: injected },
  };
  return originalNewSession.call(this, params);
};

await import(process.argv[2]);
