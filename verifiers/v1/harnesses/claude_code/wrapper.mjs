/** Inject trace-scoped SDK hooks before claude-agent-acp creates its query. */

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
