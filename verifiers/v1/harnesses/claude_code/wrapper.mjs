/** Route Claude SDK tool hooks through standard ACP permission requests. */

// {tool_content}

const { ClaudeAcpAgent } = await import(process.argv[1]);
const originalNewSession = ClaudeAcpAgent.prototype.newSession;

function mcpParts(response) {
  if (Array.isArray(response)) return [response, (parts) => parts];
  if (
    response &&
    typeof response === "object" &&
    Array.isArray(response.content)
  ) {
    return [response.content, (parts) => ({ ...response, content: parts })];
  }
  throw new TypeError("Claude returned unsupported structured MCP output");
}

function resultCodec(hook) {
  if (typeof hook.tool_response === "string") {
    return {
      content: hook.tool_response,
      output: (content) => content,
      replacement: "nonempty_text",
    };
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
    return {
      content: hook.tool_response.stdout,
      output: (content) => ({
        ...hook.tool_response,
        stdout: content,
        stderr: "",
      }),
      replacement: "nonempty_text",
    };
  }
  if (hook.tool_name.startsWith("mcp__")) {
    const [parts, restore] = mcpParts(hook.tool_response);
    return {
      content: vfToolContent(parts, "Claude"),
      output: (content) => restore(hostToolContent(content, "Claude")),
      replacement: "any",
    };
  }
  const content = JSON.stringify(hook.tool_response);
  if (typeof content !== "string") {
    throw new TypeError(`Claude returned invalid output for ${hook.tool_name}`);
  }
  return {
    content,
    output: (replacement) => replacement,
    replacement: "nonempty_text",
  };
}

async function requestDecision(agent, hook, hookOptions) {
  const before = hook.hook_event_name === "PreToolUse";
  const failed = hook.hook_event_name === "PostToolUseFailure";
  let content;
  let output;
  let replacement;
  let toolInterception;
  try {
    if (before || failed) {
      content = before ? "" : hook.error || "Tool execution failed.";
    } else {
      ({ content, output, replacement } = resultCodec(hook));
    }
    toolInterception = {
      phase: before ? "before" : "after",
      content: failed ? "none" : before ? "nonempty_text" : replacement,
      message: {
        role: "tool",
        tool_call_id: hook.tool_use_id,
        content,
        name: hook.tool_name,
      },
    };
  } catch (error) {
    toolInterception = { error: String(error) };
  }
  const response = await agent.client.requestPermission(
    {
      sessionId: hook.session_id,
      toolCall: {
        toolCallId: hook.tool_use_id,
        title: hook.tool_name,
        rawInput: hook.tool_input,
        _meta: { toolInterception },
      },
      options: [{ optionId: "continue", name: "Continue", kind: "allow_once" }],
    },
    hookOptions?.signal,
  );
  if (toolInterception.error) throw new Error(toolInterception.error);
  const decision = validateToolDecision(
    response?._meta?.toolInterception,
    "LiveACPClient",
  );
  return { decision, output, before, failed };
}

function createHook(agent, blockedCalls) {
  return async (hook, toolUseId, hookOptions) => {
    if (
      hook.hook_event_name === "PostToolUseFailure" &&
      blockedCalls.delete(hook.tool_use_id)
    ) {
      return {};
    }
    try {
      const { decision, output, before, failed } = await requestDecision(
        agent,
        hook,
        hookOptions,
      );
      if (decision.action === "stop") {
        return {
          continue: false,
          stopReason: decision.reason || "Rollout terminated by interception.",
        };
      }
      if (before) {
        if (decision.action === "allow") return {};
        blockedCalls.add(hook.tool_use_id);
        return {
          hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: decision.message.content,
          },
        };
      }
      if (failed || decision.action === "allow") return {};
      return {
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          updatedToolOutput: output(decision.message.content),
        },
      };
    } catch (error) {
      return {
        continue: false,
        stopReason: `Tool interception is unavailable: ${error}`,
      };
    }
  };
}

ClaudeAcpAgent.prototype.newSession = function (params) {
  const callback = createHook(this, new Set());
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
