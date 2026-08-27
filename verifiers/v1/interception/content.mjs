function vfToolContent(parts, harness) {
  if (!Array.isArray(parts)) throw new TypeError(`Invalid ${harness} tool result`);
  const content = parts.map((part) => {
    if (part?.type === "text" && typeof part.text === "string") {
      return { type: "text", text: part.text };
    }
    if (
      part?.type === "image" &&
      typeof part.mimeType === "string" &&
      typeof part.data === "string"
    ) {
      return {
        type: "image_url",
        image_url: { url: `data:${part.mimeType};base64,${part.data}` },
      };
    }
    throw new TypeError(`Unsupported ${harness} tool result content`);
  });
  return content.every((part) => part.type === "text")
    ? content.map((part) => part.text).join("\n")
    : content;
}

function hostToolContent(content, harness) {
  const parts = typeof content === "string" ? [{ type: "text", text: content }] : content;
  if (!Array.isArray(parts)) throw new TypeError(`Invalid ${harness} replacement`);
  return parts.map((part) => {
    if (part?.type === "text" && typeof part.text === "string") {
      return { type: "text", text: part.text };
    }
    const url = part?.type === "image_url" ? part.image_url?.url : undefined;
    const image =
      typeof url === "string"
        ? /^data:(image\/[^;,]+);base64,([A-Za-z0-9+/]*={0,2})$/.exec(url)
        : null;
    if (!image) throw new TypeError(`${harness} requires inline base64 tool images`);
    return { type: "image", mimeType: image[1], data: image[2] };
  });
}

function validateToolDecision(decision, source) {
  if (!decision || !["allow", "rewrite", "stop"].includes(decision.action)) {
    throw new TypeError(`${source} returned an invalid interception decision`);
  }
  if (decision.action === "rewrite" && !decision.message) {
    throw new TypeError(`${source} omitted the rewritten tool result`);
  }
  return decision;
}
