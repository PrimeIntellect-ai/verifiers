// Some Messages-compatible gateways emit null for unknown cache counts. DSH's
// parser accepts omitted counts, but rejects null. Keep the last known counts.
const fetch = globalThis.fetch;
const base = new URL(process.env.DEEPSEEK_BASE_URL);
base.pathname = base.pathname.replace(/\/$/, "");
if (!base.pathname.endsWith("/v1")) base.pathname += "/v1";
base.pathname += "/messages";

function normalize(line) {
  if (!line.startsWith("data:")) return line;
  let data;
  try {
    data = JSON.parse(line.slice(5));
  } catch {
    // Preserve SSE sentinels and non-JSON data for the upstream parser.
    return line;
  }
  let changed = false;
  for (const usage of [data?.usage, data?.message?.usage]) {
    if (!usage) continue;
    for (const key of ["cache_read_input_tokens", "cache_creation_input_tokens"]) {
      if (usage[key] === null) {
        delete usage[key];
        changed = true;
      }
    }
  }
  return changed ? `data: ${JSON.stringify(data)}` : line;
}

globalThis.fetch = async (input, init) => {
  const url = new URL(input instanceof Request ? input.url : input);
  if (url.href === base.href && typeof init?.body === "string") {
    const body = JSON.parse(init.body);
    // DSH enables thinking without a budget; Messages gateways require either
    // a budget or adaptive thinking. Eval sampling overrides apply downstream.
    if (body.thinking?.type === "enabled" && body.thinking.budget_tokens === undefined) {
      body.thinking = {type: "adaptive"};
      init = {...init, body: JSON.stringify(body)};
    }
  }
  const response = await fetch(input, init);
  if (url.href === base.href && !response.ok) {
    const detail = (await response.clone().text()).slice(0, 1000);
    console.error(`DeepSeek Messages HTTP ${response.status}: ${detail}`);
  }
  if (
    url.href !== base.href ||
    !response.body ||
    !response.headers.get("content-type")?.includes("text/event-stream")
  ) return response;

  let pending = "";
  const body = response.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TransformStream({
      transform(chunk, controller) {
        pending += chunk;
        const lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) controller.enqueue(`${normalize(line)}\n`);
      },
      flush(controller) {
        if (pending) controller.enqueue(normalize(pending));
      },
    }))
    .pipeThrough(new TextEncoderStream());
  const headers = new Headers(response.headers);
  headers.delete("content-length");
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
};
