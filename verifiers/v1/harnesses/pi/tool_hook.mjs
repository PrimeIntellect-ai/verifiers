/** Route Pi's awaited native tool hooks to the rollout's ACP runner. */

// {tool_content}
// {tool_socket}

async function intercept(phase, toolCallId, name, content) {
  return requestToolPolicy({
    phase,
    content: "nonempty_text",
    message: {
      role: "tool",
      tool_call_id: toolCallId,
      content,
      name,
    },
  });
}

export default function toolInterceptionExtension(pi) {
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
      decision = await intercept("before", toolCallId, event.toolName, "");
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
    if (preReplacements.has(toolCallId)) return undefined;
    const content = vfToolContent(event.content, "Pi");
    let decision;
    try {
      decision = await intercept("after", toolCallId, event.toolName, content);
    } catch (error) {
      console.error("Tool interception failed:", error);
      ctx.abort();
      process.exit(1);
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
      content: [{ type: "text", text: decision.message.content }],
      isError: event.isError,
    };
  });
}
