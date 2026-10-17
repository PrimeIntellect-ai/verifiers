// Codex calls this hook before execution, including nested Code Mode tools.
// The rollout URL, bearer and failure file are substituted when the runtime copy is written.
import { readFileSync, writeFileSync } from "node:fs";

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
// The rollout stamps its answers; an unstamped one came from a tunnel or proxy in between.
const retryable = (response) =>
  !response.headers.has("x-verifiers-interception") &&
  ([404, 408, 429].includes(response.status) || response.status >= 500);

// Retry what a tunnel or proxy dropped or answered, marked so the rollout answers a
// repeat with its first verdict.
async function ask(body) {
  const deadline = Date.now() + 300_000;
  for (let retry = 0, delay = 500; ; retry++, delay = Math.min(2 * delay, 10_000)) {
    const failure = await fetch(__URL__, {
      method: "POST",
      headers: {
        Authorization: "Bearer " + __SECRET__,
        "Content-Type": "application/json",
        "x-stainless-retry-count": String(retry),
      },
      body,
      signal: AbortSignal.timeout(120_000),
    }).then(
      async (response) => {
        if (retryable(response)) return `HTTP ${response.status}`;
        if (!response.ok) throw new Error(`tool gate returned ${response.status}`);
        return { decision: await response.json() };
      },
      (error) => String(error),
    );
    if (typeof failure !== "string") return failure.decision;
    if (Date.now() + delay > deadline) throw new Error(`unreachable for 300s: ${failure}`);
    await new Promise((resolve) => setTimeout(resolve, delay * (0.5 + Math.random())));
  }
}

const hook = JSON.parse(readFileSync(0, "utf8"));
try {
  const decision = await ask(
    JSON.stringify({
      tool_call_id: hook.tool_use_id,
      name: hook.tool_name,
      arguments: hook.tool_input,
    }),
  );
  if (decision.action !== "allow") {
    const content = decision.action === "stop" ? decision.reason : decision.message?.content;
    deny(typeof content === "string" ? content : JSON.stringify(content ?? ""));
  }
} catch (error) {
  // A denial would reach the model as the policy's verdict, so the runner fails the turn.
  try {
    writeFileSync(__FAILED__, `tool gate failed for ${hook.tool_use_id}: ${error}`);
  } finally {
    deny(`tool gate unavailable: ${error}`);
  }
}
