// pi-acp relays confirmations as ACP permissions. Put the model call ID
// in the title and its arguments in the message for the runner to recover.
export default function (pi) {
  pi.on("tool_call", async (event, ctx) => {
    const allowed = await ctx.ui.confirm(
      event.toolCallId.split("|", 1)[0],
      JSON.stringify(event.input),
    );
    if (!allowed) return { block: true, reason: "Blocked by the rollout's tool policy." };
  });
}
