// Codex calls this hook before execution, including nested Code Mode tools.
// The rollout URL and bearer are substituted when the runtime copy is written.
import { readFileSync } from "node:fs";

const deny = (reason) =>
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: reason,
      },
    }),
  );
const hook = JSON.parse(readFileSync(0, "utf8"));
try {
  const response = await fetch(__URL__, {
    method: "POST",
    headers: { Authorization: "Bearer " + __SECRET__, "Content-Type": "application/json" },
    body: JSON.stringify({
      tool_call_id: hook.tool_use_id,
      name: hook.tool_name,
      arguments: hook.tool_input,
    }),
  });
  if (!response.ok) throw new Error(`tool gate returned ${response.status}`);
  const decision = await response.json();
  if (decision.action !== "allow") {
    const content = decision.action === "stop" ? decision.reason : decision.message?.content;
    deny(typeof content === "string" ? content : JSON.stringify(content ?? ""));
  }
} catch (error) {
  deny(`tool gate unavailable: ${error}`);
}
