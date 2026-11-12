import { writeFileSync } from "node:fs";

export default {
  id: "execution-gate",
  name: "Execution gate",
  register(api) {
    api.on("before_tool_call", async (event) => {
      try {
        const response = await fetch(__URL__, {
          method: "POST",
          headers: {
            Authorization: "Bearer " + __SECRET__,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            tool_call_id: event.toolCallId,
            name: event.toolName,
            arguments: event.params,
          }),
        });
        if (!response.ok) throw new Error(`tool gate returned HTTP ${response.status}`);
        const decision = await response.json();
        if (decision.action === "allow") {
          if (__REMOTE_ONLY__) throw new Error("split execution returned a local allow");
          return;
        }
        if (decision.action !== "deny" && decision.action !== "stop") {
          throw new Error("tool gate returned an unknown action");
        }
        const content = decision.action === "stop" ? decision.reason : decision.message?.content;
        return { block: true, blockReason: typeof content === "string" ? content : JSON.stringify(content ?? "") };
      } catch (error) {
        return { block: true, blockReason: `tool gate unavailable: ${error}` };
      }
    }, { timeoutMs: 86_400_000 });
    // The launcher checks this before allowing an ACP prompt. A missing or
    // rejected plugin must never silently leave the local tool body enabled.
    writeFileSync(__READY__, "ready");
  },
};
