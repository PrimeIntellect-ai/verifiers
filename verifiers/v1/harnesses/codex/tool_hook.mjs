/** Bridge Codex's awaited native tool hooks to the rollout's /tool policy. */

import { readFileSync } from "node:fs";

const hook = JSON.parse(readFileSync(0, "utf8"));
const before = hook.hook_event_name === "PreToolUse";

async function run() {
  const url = process.env.VF_CODEX_TOOL_URL;
  const secret = process.env.VF_CODEX_TOOL_SECRET;
  if (!url || !secret) throw new Error("Tool interception is not configured");

  let content = "";
  if (!before) {
    content =
      typeof hook.tool_response === "string"
        ? hook.tool_response
        : JSON.stringify(hook.tool_response);
    if (!content) content = `(${hook.tool_name} completed with no output)`;
  }

  let resultPrefix = "";
  let resultSuffix = "";
  if (before) {
    const command = hook.tool_input?.command;
    if (
      (hook.tool_name === "Bash" || hook.tool_name === "apply_patch") &&
      typeof command === "string"
    ) {
      resultPrefix = "Command blocked by PreToolUse hook: ";
      resultSuffix = `. Command: ${command}`;
    } else {
      resultPrefix = "Tool call blocked by PreToolUse hook: ";
      resultSuffix = `. Tool: ${hook.tool_name}`;
    }
  }

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${secret}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phase: before ? "before" : "after",
      content: "nonempty_text",
      resultPrefix,
      resultSuffix,
      message: {
        role: "tool",
        tool_call_id: hook.tool_use_id,
        content,
        name: hook.tool_name,
      },
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`Tool interception returned ${response.status}`);
  const decision = await response.json();
  if (!["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new Error("Tool interception returned an invalid action");
  }

  let replacement = content;
  if (decision.action === "rewrite") {
    replacement = decision.message?.content;
    if (typeof replacement !== "string" || !replacement) {
      throw new Error("Tool interception returned an invalid replacement");
    }
  } else if (decision.action === "stop") {
    replacement = decision.reason || "Rollout terminated by interception.";
  } else if (before) {
    return;
  }

  if (before) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision: "deny",
          permissionDecisionReason: replacement,
        },
      }),
    );
  } else {
    // Codex PostToolUse has no updatedToolOutput field. Its universal feedback
    // response replaces the model-visible result without blocking the agent loop.
    console.log(JSON.stringify({ continue: false, stopReason: replacement }));
  }
}

run().catch((error) => {
  const reason = `Tool interception is unavailable: ${error.message}`;
  console.log(
    JSON.stringify(
      before
        ? {
            hookSpecificOutput: {
              hookEventName: "PreToolUse",
              permissionDecision: "deny",
              permissionDecisionReason: reason,
            },
          }
        : { continue: false, stopReason: reason },
    ),
  );
});
