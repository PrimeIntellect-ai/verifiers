/** Route Claude SDK tool hooks through standard ACP permission requests. */

const maxTextUnits = 50_000;
const { ClaudeAcpAgent } = await import(process.argv[1]);
const originalNewSession = ClaudeAcpAgent.prototype.newSession;

function resultText(hook) {
  let content;
  if (typeof hook.tool_response === "string") {
    content = hook.tool_response;
  } else if (
    hook.tool_name === "Bash" &&
    hook.tool_response &&
    typeof hook.tool_response === "object" &&
    !Array.isArray(hook.tool_response) &&
    typeof hook.tool_response.stdout === "string" &&
    hook.tool_response.isImage === false &&
    hook.tool_response.persistedOutputPath === undefined
  ) {
    content = hook.tool_response.stdout;
  } else {
    content = JSON.stringify(hook.tool_response);
  }
  if (typeof content !== "string" || content.length > maxTextUnits) {
    throw new Error(`Claude returned unsupported output for ${hook.tool_name}`);
  }
  return content.trim() ? content : `(${hook.tool_name} completed with no output)`;
}

function resultOutput(hook, content) {
  // Preserve Bash's native envelope while forcing the approved text to become
  // the exact tool result Claude sends to its next model request.
  if (
    hook.tool_name === "Bash" &&
    hook.tool_response &&
    typeof hook.tool_response === "object" &&
    !Array.isArray(hook.tool_response)
  ) {
    return { ...hook.tool_response, stdout: content, stderr: "" };
  }
  return content;
}

async function requestDecision(agent, hook, hookOptions) {
  const before = hook.hook_event_name === "PreToolUse";
  const failed = hook.hook_event_name === "PostToolUseFailure";
  let content;
  let toolInterception;
  try {
    content = before ? "" : failed ? hook.error || "Tool execution failed." : resultText(hook);
    toolInterception = {
      phase: before ? "before" : "after",
      content: failed ? "none" : "nonempty_text",
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
  const decision = response?._meta?.toolInterception;
  if (toolInterception.error) throw new Error(toolInterception.error);
  if (!decision || !["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new Error("LiveACPClient returned an invalid interception decision");
  }
  return { decision, content, before, failed };
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
      const { decision, content, before, failed } = await requestDecision(
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
      if (failed || decision.action === "allow" && content === hook.tool_response) {
        return {};
      }
      return {
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          updatedToolOutput: resultOutput(
            hook,
            decision.action === "rewrite" ? decision.message.content : content,
          ),
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
