// Preserve native policy denials; route every otherwise allowed tool through ACP.
// The shared ACP runner resolves the call ID against the rollout's tool policy.
export const name = "vf-tool-gate";
export const inject = ["tools", "approval"];

export function apply(ctx) {
  ctx.on("tools/pre-execute", async (_exec, next) => {
    const decision = await next();
    return decision.kind === "allow" ? { kind: "ask" } : decision;
  });
}
