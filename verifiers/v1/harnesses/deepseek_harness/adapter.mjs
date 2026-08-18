/** DSH LLM adapter that talks directly to a Verifiers interception server. */

import {
  attributionHeaders,
  CallId,
  contentHasImage,
  LlmAdapter,
  LlmError,
} from "@deepseek-ai/dsh-llm";

function textOf(blocks) {
  return blocks
    .map((block) => {
      if (block.type === "text") return block.text;
      if (block.type === "tool-result") return textOf(block.content);
      return "";
    })
    .join("");
}

function reasoningOf(blocks) {
  return blocks
    .filter((block) => block.type === "reasoning")
    .map((block) => block.text)
    .join("");
}

function toolCallsOf(blocks) {
  return blocks.filter((block) => block.type === "tool-call");
}

function toolNameMaps(tools, bareToolPrefixes) {
  const internalToProvider = new Map();
  const providerToInternal = new Map();
  // DSH requires namespaced MCP names internally; only Verifiers' bare tools cross the API bare.
  for (const tool of tools || []) {
    const prefix = bareToolPrefixes.find((candidate) => tool.name.startsWith(candidate));
    const providerName = prefix === undefined ? tool.name : tool.name.slice(prefix.length);
    const existing = providerToInternal.get(providerName);
    if (existing !== undefined && existing !== tool.name) {
      throw new LlmError(
        `Tools ${existing} and ${tool.name} both map to ${providerName}.`,
        "INVALID_REQUEST",
      );
    }
    internalToProvider.set(tool.name, providerName);
    providerToInternal.set(providerName, tool.name);
  }
  return { internalToProvider, providerToInternal };
}

function providerOptions(options, internalToProvider) {
  return {
    ...options,
    tools: options.tools?.map((tool) => ({
      ...tool,
      name: internalToProvider.get(tool.name) || tool.name,
    })),
    messages: options.messages.map((message) => ({
      ...message,
      content: message.content.map((block) =>
        block.type === "tool-call"
          ? { ...block, name: internalToProvider.get(block.name) || block.name }
          : block,
      ),
    })),
  };
}

function replayOf(message, transport) {
  const source = message.source.kind === "model" ? message.source : undefined;
  const state = source?.replayState?.response;
  return state?.transport === transport ? state.data : undefined;
}

const TOOL_ENCODERS = {
  chat_completions: (tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }),
  responses: (tool) => ({ type: "function", ...tool }),
  anthropic_messages: (tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.parameters,
  }),
};

function toolsOf(options, transport) {
  return options.tools?.length
    ? { tools: options.tools.map(TOOL_ENCODERS[transport]) }
    : {};
}

function chatRequest(options) {
  const messages = [];
  if (options.system !== undefined) messages.push({ role: "system", content: options.system });

  for (const message of options.messages) {
    if (message.role === "system") {
      messages.push({ role: "system", content: textOf(message.content) });
      continue;
    }
    if (message.role === "assistant") {
      const text = textOf(message.content);
      const reasoning = reasoningOf(message.content);
      const calls = toolCallsOf(message.content);
      const replay = replayOf(message, "chat_completions");
      messages.push({
        role: "assistant",
        content: text || (calls.length > 0 ? null : ""),
        ...(Array.isArray(replay)
          ? { reasoning_details: replay }
          : reasoning
            ? { reasoning_content: reasoning }
            : {}),
        ...(calls.length === 0
          ? {}
          : {
              tool_calls: calls.map((call) => ({
                id: call.id,
                type: "function",
                function: { name: call.name, arguments: call.arguments },
              })),
            }),
      });
      continue;
    }

    const results = message.content.filter((block) => block.type === "tool-result");
    const text = textOf(message.content.filter((block) => block.type !== "tool-result"));
    if (text || results.length === 0) messages.push({ role: "user", content: text });
    for (const result of results) {
      messages.push({
        role: "tool",
        tool_call_id: result.toolCallId,
        content: textOf(result.content) || "(no output)",
      });
    }
  }

  return {
    model: options.model,
    messages,
    stream: false,
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.stop === undefined ? {} : { stop: options.stop }),
    ...(options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens }),
    ...toolsOf(options, "chat_completions"),
  };
}

function reasoningText(message) {
  for (const field of ["reasoning", "reasoning_content"]) {
    if (typeof message[field] === "string" && message[field]) return message[field];
  }
  if (!Array.isArray(message.reasoning_details)) return "";
  return message.reasoning_details
    .map((detail) => detail?.text || detail?.summary || "")
    .filter(Boolean)
    .join("\n");
}

function chatResponse(raw) {
  const choice = raw.choices?.[0];
  const message = choice?.message;
  if (message === undefined) {
    throw new LlmError("Chat Completions response has no assistant message.", "MALFORMED_RESPONSE");
  }
  const blocks = [];
  const reasoning = reasoningText(message);
  if (reasoning) blocks.push({ type: "reasoning", text: reasoning });
  const contentText = typeof message.content === "string"
    ? message.content
    : Array.isArray(message.content)
      ? message.content.map((part) => part?.text || part?.refusal || "").join("")
      : "";
  const text = contentText || (typeof message.refusal === "string" ? message.refusal : "");
  if (text) blocks.push({ type: "text", text });
  for (const call of message.tool_calls || []) {
    blocks.push({
      type: "tool-call",
      id: String(call.id || ""),
      name: String(call.function?.name || ""),
      arguments: String(call.function?.arguments || ""),
    });
  }
  return {
    blocks,
    usage: openAiUsage(raw.usage),
    reason: finishReason(choice?.finish_reason, blocks),
    replayState: {
      transport: "chat_completions",
      data: Array.isArray(message.reasoning_details)
        ? message.reasoning_details
        : null,
    },
  };
}

function responsesRequest(options) {
  const input = [];
  for (const message of options.messages) {
    if (message.role === "assistant") {
      const replay = replayOf(message, "responses");
      if (Array.isArray(replay)) {
        input.push(...replay);
        continue;
      }
      const text = textOf(message.content);
      if (text) {
        input.push({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text }],
        });
      }
      for (const call of toolCallsOf(message.content)) {
        input.push({
          type: "function_call",
          call_id: call.id,
          name: call.name,
          arguments: call.arguments,
        });
      }
      continue;
    }
    if (message.role === "system") {
      input.push({ role: "system", content: textOf(message.content) });
      continue;
    }
    const results = message.content.filter((block) => block.type === "tool-result");
    const text = textOf(message.content.filter((block) => block.type !== "tool-result"));
    if (text || results.length === 0) input.push({ role: "user", content: text });
    for (const result of results) {
      input.push({
        type: "function_call_output",
        call_id: result.toolCallId,
        output: textOf(result.content) || "(no output)",
      });
    }
  }

  return {
    model: options.model,
    input,
    stream: false,
    ...(options.system === undefined ? {} : { instructions: options.system }),
    include: ["reasoning.encrypted_content"],
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.maxTokens === undefined ? {} : { max_output_tokens: options.maxTokens }),
    ...toolsOf(options, "responses"),
  };
}

function responsesResponse(raw) {
  const output = raw.output;
  if (!Array.isArray(output)) {
    throw new LlmError("Responses response has no output array.", "MALFORMED_RESPONSE");
  }
  const blocks = [];
  const reasoning = [];
  let text = "";
  for (const item of output) {
    if (item.type === "reasoning") {
      reasoning.push(...(item.summary || []).map((part) => part?.text || ""));
      reasoning.push(...(item.content || []).map((part) => part?.text || ""));
    } else if (item.type === "message") {
      text += (item.content || [])
        .filter((part) => part?.type === "output_text" || part?.type === "refusal")
        .map((part) => part.text || part.refusal || "")
        .join("");
    } else if (item.type === "function_call" || item.type === "custom_tool_call") {
      blocks.push({
        type: "tool-call",
        id: String(item.call_id || ""),
        name: String(item.name || ""),
        arguments: String(item.arguments ?? item.input ?? ""),
      });
    }
  }
  const reasoningText = reasoning.filter(Boolean).join("\n");
  const incompleteReason = raw.incomplete_details?.reason || "incomplete";
  if (reasoningText) blocks.unshift({ type: "reasoning", text: reasoningText });
  if (text) blocks.splice(reasoningText ? 1 : 0, 0, { type: "text", text });
  return {
    blocks,
    usage: responsesUsage(raw.usage),
    reason: raw.status !== "incomplete"
      ? finishReason(undefined, blocks)
      : incompleteReason === "max_output_tokens"
        ? { kind: "max-tokens" }
        : {
            kind: "error",
            failure: {
              message: `model stopped: ${incompleteReason}`,
              code: String(incompleteReason).toUpperCase(),
            },
          },
    replayState: {
      transport: "responses",
      data: output,
    },
  };
}

function anthropicRequest(options) {
  const messages = [];
  const system = options.system === undefined ? [] : [options.system];
  for (const message of options.messages) {
    if (message.role === "system") {
      system.push(textOf(message.content));
      continue;
    }
    if (message.role === "assistant") {
      const replay = replayOf(message, "anthropic_messages");
      if (Array.isArray(replay)) {
        messages.push({ role: "assistant", content: replay });
        continue;
      }
      const content = [];
      const text = textOf(message.content);
      if (text) content.push({ type: "text", text });
      for (const call of toolCallsOf(message.content)) {
        let input = {};
        try {
          const parsed = JSON.parse(call.arguments);
          if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
            input = parsed;
          }
        } catch {
          // Anthropic requires an object; invalid tool arguments stay executable as {}.
        }
        content.push({ type: "tool_use", id: call.id, name: call.name, input });
      }
      messages.push({ role: "assistant", content });
      continue;
    }
    const content = [];
    const text = textOf(message.content.filter((block) => block.type !== "tool-result"));
    if (text) content.push({ type: "text", text });
    for (const result of message.content.filter((block) => block.type === "tool-result")) {
      content.push({
        type: "tool_result",
        tool_use_id: result.toolCallId,
        content: textOf(result.content) || "(no output)",
        ...(result.isError === undefined ? {} : { is_error: result.isError }),
      });
    }
    if (content.length === 0) content.push({ type: "text", text: "" });
    messages.push({ role: "user", content });
  }

  return {
    model: options.model,
    messages,
    stream: false,
    max_tokens: options.maxTokens,
    ...(system.length === 0 ? {} : { system: system.join("\n\n") }),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.stop === undefined ? {} : { stop_sequences: options.stop }),
    ...toolsOf(options, "anthropic_messages"),
  };
}

function anthropicResponse(raw) {
  if (!Array.isArray(raw.content)) {
    throw new LlmError("Anthropic response has no content array.", "MALFORMED_RESPONSE");
  }
  const blocks = [];
  const reasoning = raw.content
    .filter((block) => block.type === "thinking")
    .map((block) => block.thinking || "")
    .join("");
  if (reasoning) blocks.push({ type: "reasoning", text: reasoning });
  const text = raw.content
    .filter((block) => block.type === "text")
    .map((block) => block.text || "")
    .join("");
  if (text) blocks.push({ type: "text", text });
  for (const call of raw.content.filter((block) => block.type === "tool_use")) {
    blocks.push({
      type: "tool-call",
      id: String(call.id || ""),
      name: String(call.name || ""),
      arguments: JSON.stringify(call.input || {}),
    });
  }
  const stopReasons = {
    end_turn: "stop",
    stop_sequence: "stop",
    max_tokens: "max-tokens",
    tool_use: "tool-calls",
  };
  const kind = stopReasons[raw.stop_reason] || finishReason(undefined, blocks).kind;
  return {
    blocks,
    usage: anthropicUsage(raw.usage),
    reason: { kind },
    replayState: {
      transport: "anthropic_messages",
      data: raw.content,
    },
  };
}

function openAiUsage(usage = {}) {
  const cached = usage.prompt_tokens_details?.cached_tokens || 0;
  const reasoning = usage.completion_tokens_details?.reasoning_tokens;
  return {
    inputTokens: Math.max(0, (usage.prompt_tokens || 0) - cached),
    outputTokens: usage.completion_tokens || 0,
    ...(cached ? { cacheReadTokens: cached } : {}),
    ...(reasoning ? { reasoningTokens: reasoning } : {}),
  };
}

function responsesUsage(usage = {}) {
  const cached = usage.input_tokens_details?.cached_tokens || 0;
  const reasoning = usage.output_tokens_details?.reasoning_tokens;
  return {
    inputTokens: Math.max(0, (usage.input_tokens || 0) - cached),
    outputTokens: usage.output_tokens || 0,
    ...(cached ? { cacheReadTokens: cached } : {}),
    ...(reasoning ? { reasoningTokens: reasoning } : {}),
  };
}

function anthropicUsage(usage = {}) {
  const read = usage.cache_read_input_tokens || 0;
  const write = usage.cache_creation_input_tokens || 0;
  const reasoning = usage.output_tokens_details?.thinking_tokens;
  return {
    inputTokens: usage.input_tokens || 0,
    outputTokens: usage.output_tokens || 0,
    ...(read ? { cacheReadTokens: read } : {}),
    ...(write ? { cacheWriteTokens: write } : {}),
    ...(reasoning ? { reasoningTokens: reasoning } : {}),
  };
}

function finishReason(raw, blocks) {
  if (raw === "length") return { kind: "max-tokens" };
  if (raw === "tool_calls" || blocks.some((block) => block.type === "tool-call")) {
    return { kind: "tool-calls" };
  }
  return { kind: "stop" };
}

function errorCode(status) {
  if (status === 401 || status === 403) return "AUTH";
  if (status === 429) return "RATE_LIMIT";
  if (status === 400) return "INVALID_REQUEST";
  if (status >= 500) return "SERVER";
  return `HTTP_${status}`;
}

async function postJson(url, apiKey, transport, body, signal) {
  const headers = {
    "content-type": "application/json",
    ...attributionHeaders(),
    ...(transport === "anthropic_messages"
      ? { "x-api-key": apiKey, "anthropic-version": "2023-06-01" }
      : { authorization: `Bearer ${apiKey}` }),
  };
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw new LlmError("Verifiers request aborted.", "ABORTED", { cause: error });
    throw new LlmError(`Verifiers request to ${url} failed.`, "TRANSPORT", { cause: error });
  }
  const raw = await response.json().catch(() => undefined);
  if (!response.ok) {
    const detail = raw?.error?.message || raw?.error?.type || response.statusText;
    throw new LlmError(`Verifiers request failed (${response.status}): ${detail}`, errorCode(response.status));
  }
  if (raw === undefined || raw === null || typeof raw !== "object") {
    throw new LlmError("Verifiers returned malformed JSON.", "MALFORMED_RESPONSE");
  }
  return raw;
}

const CODECS = {
  chat_completions: {
    path: "chat/completions",
    request: chatRequest,
    response: chatResponse,
  },
  responses: {
    path: "responses",
    request: responsesRequest,
    response: responsesResponse,
  },
  anthropic_messages: {
    path: "messages",
    request: anthropicRequest,
    response: anthropicResponse,
  },
};

class VerifiersAdapter extends LlmAdapter {
  constructor(config) {
    super();
    this.config = config;
  }

  resolveModel(provider, model) {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ["text"],
      ...(this.config.maxTokens === undefined
        ? {}
        : { defaultMaxTokens: this.config.maxTokens }),
    });
  }

  async *stream(options) {
    if (options.messages.some((message) => contentHasImage(message.content))) {
      throw new LlmError(
        "The Verifiers DSH adapter does not support image content.",
        "UNSUPPORTED_CONTENT",
      );
    }
    const codec = CODECS[this.config.transport];
    const apiKey = process.env.DSH_INTERCEPT_KEY;
    if (!apiKey) throw new LlmError("Missing DSH_INTERCEPT_KEY.", "MISSING_CREDENTIAL");
    const names = toolNameMaps(options.tools, this.config.bareToolPrefixes || []);
    const body = codec.request(providerOptions(options, names.internalToProvider));
    const url = `${this.config.endpoint.replace(/\/$/, "")}/${codec.path}`;
    const raw = await postJson(
      url,
      apiKey,
      this.config.transport,
      body,
      options.signal,
    );
    const result = codec.response(raw);
    const blocks = result.blocks.map((block) =>
      block.type === "tool-call"
        ? { ...block, name: names.providerToInternal.get(block.name) || block.name }
        : block,
    );
    if (blocks.length === 0) {
      throw new LlmError(`Model ${options.model} returned no content.`, "EMPTY_RESPONSE");
    }
    for (const [index, block] of blocks.entries()) {
      yield { type: "block-start", index, blockType: block.type };
      if (block.type === "text") {
        yield { type: "text-delta", index, text: block.text };
      } else if (block.type === "reasoning") {
        yield { type: "reasoning-delta", index, text: block.text };
      } else {
        yield {
          type: "tool-call-delta",
          index,
          id: CallId(block.id),
          name: block.name,
          argumentsDelta: block.arguments,
        };
      }
      yield {
        type: "block-end",
        index,
        block: block.type === "tool-call" ? { ...block, id: CallId(block.id) } : block,
      };
    }
    yield { type: "usage", usage: result.usage };
    yield {
      type: "finish",
      reason: result.reason,
      replayState: { response: result.replayState },
    };
  }
}

export const name = "llm-verifiers";
export const inject = ["llm"];

export function apply(ctx, config) {
  ctx.llm.registerAdapter(["verifiers"], new VerifiersAdapter(config));
}
