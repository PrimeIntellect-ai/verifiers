# syntax=docker/dockerfile:1
FROM docker.io/library/python:3.14-bookworm AS base
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/bin/node /usr/local/bin/node
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/
RUN ln -sf /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends curl ca-certificates git ripgrep tmux \
 && rm -rf /var/lib/apt/lists/* \
 && uv pip install --system --no-cache \
      "agent-client-protocol==0.12.1" \
      "openai>=2.54.0,<3.0.0" \
      "mcp==2.3.0" \
      "httpx>=0.27.0" \
      "httpx2>=2.12.0" \
      "tenacity>=8.5.0" \
      "certifi" \
      "tomli-w>=1.0.0" \
 && for tool in python3 node npm npx uv uvx curl git rg; do "$tool" --version || exit 1; done \
 && tmux -V

FROM base AS browser-use
RUN uv pip install --system --no-cache "browser-harness==0.1.13"

FROM base AS claude-code
ARG VERSION=2.1.327
ARG ACP_VERSION=0.108.1
RUN dir="/var/tmp/vf-claude-agent-acp-${VERSION}-${ACP_VERSION}" \
 && npm install --prefix "$dir/packages" --no-audit --no-fund --omit=dev \
      "@anthropic-ai/claude-code@${VERSION}" \
      "@agentclientprotocol/claude-agent-acp@${ACP_VERSION}" \
 && touch "$dir/.ready" \
 && npm cache clean --force \
 && chmod -R a+rX "$dir"

FROM base AS codex
ARG VERSION=0.170.0
ARG ACP_VERSION=2.7.0
RUN dir="/var/tmp/vf-codex-${VERSION}-${ACP_VERSION}" \
 && mkdir -p "$dir/acp" \
 && cd "$dir/acp" \
 && printf '{"overrides":{"@agentclientprotocol/codex-acp":{"@openai/codex":"%s"}}}\n' "$VERSION" > package.json \
 && npm install --ignore-scripts --no-audit --no-fund --omit=dev \
      "@agentclientprotocol/codex-acp@${ACP_VERSION}" \
      "@openai/codex@${VERSION}" \
 && touch "$dir/.ready" \
 && npm cache clean --force \
 && chmod -R a+rX "$dir"

FROM base AS deepseek
ARG VERSION=0.2.2-rc.1
RUN dir="/var/tmp/vf-deepseek-${VERSION}" \
 && npm install --prefix "$dir" --no-audit --no-fund \
      "@deepseek-ai/dsh@${VERSION}" \
 && touch "$dir/.ready" \
 && npm cache clean --force \
 && chmod -R a+rX "$dir"

FROM base AS hermes-agent
ARG VERSION=v0.23.2
RUN dir="/var/tmp/vf-hermes-agent-${VERSION}" \
 && mkdir -p "$dir" \
 && curl -fsSL "https://codeload.github.com/NousResearch/hermes-agent/tar.gz/refs/tags/${VERSION}" \
    | tar -xz --strip-components=1 -C "$dir" \
 && sed -i '/^_REASONING_CARRIER_ROUTES: tuple = (/a\    (_DEFAULT_CARRIERS, frozenset({"custom"}), ()),' \
      "$dir/agent/message_sanitization.py" \
 && UV_CACHE_DIR=/tmp/uv-cache uv sync --project "$dir" --python 3.14 --locked --no-dev --extra acp --extra mcp --extra anthropic \
 && rm -rf /tmp/uv-cache \
 && touch "$dir/.ready" \
 && chmod -R a+rX "$dir"

FROM base AS kimi-code
ARG VERSION=2.6.3
RUN dir="/var/tmp/vf-kimi-code" \
 && npm install --prefix "$dir" --ignore-scripts --no-audit --no-fund --omit=dev \
      "@moonshot-ai/kimi-code@${VERSION}" \
 && printf '%s' "$VERSION" > "$dir/.version" \
 && npm cache clean --force \
 && chmod -R a+rX "$dir"

FROM base AS mini-swe-agent
ARG VERSION=2.4.6
RUN uv pip install --system --no-cache "mini-swe-agent==${VERSION}" "litellm[proxy]==1.110.0"

FROM base AS openclaw
ARG VERSION=2026.11.5
RUN dir="/var/tmp/vf-openclaw-${VERSION}" \
 && npm install --prefix "$dir" --no-audit --no-fund --omit=dev "openclaw@${VERSION}" \
 && file=$(find "$dir/node_modules/openclaw/dist" -type f -name 'session-accessor.sqlite-transcript-store-*.mjs') \
 && [ -n "$file" ] \
 && node - "$file" <<'NODE'
const fs = require("node:fs");
const file = process.argv[2];
const source = fs.readFileSync(file, "utf8");
const item = source.match(/(\w+)\.summary !== void 0 && !Array\.isArray\(\1\.summary\)/)?.[1];
if (!item) throw new Error("could not locate OpenClaw reasoning summary validation");
const patches = [
    [
        "const OPAQUE_REPLAY_TOKEN_RE = /^[A-Za-z0-9+/_-]+={0,2}$/;",
        "const OPAQUE_REPLAY_TOKEN_RE = /^[A-Za-z0-9+/_-]+={0,2}(?:[.][A-Za-z0-9+/_-]+={0,2})*$/;",
    ],
    ['type: "reasoning",\n\t\tsummary: [],', `type: "reasoning",\n\t\tsummary: ${item}.summary ?? [],`],
];
let patched = source;
for (const [before, after] of patches) patched = patched.replaceAll(before, after);
if (!patches.every(([, after]) => patched.includes(after))) {
    throw new Error("OpenClaw transcript sanitizer does not match replay patch");
}
fs.writeFileSync(file, patched);
NODE
RUN dir="/var/tmp/vf-openclaw-${VERSION}" \
 && touch "$dir/.ready" \
 && npm cache clean --force \
 && chmod -R a+rX "$dir"

FROM base AS opencode
ARG VERSION=2.3.2
RUN dir="/var/tmp/vf-opencode/${VERSION}" \
 && npm install --prefix "$dir" --no-audit --no-fund --omit=dev "@opencode/cli@${VERSION}" \
 && npm cache clean --force \
 && chmod -R a+rX "/var/tmp/vf-opencode"

FROM base AS pi
ARG VERSION=1.5.2
ARG ACP_VERSION=0.0.34
RUN packages="/var/tmp/vf-pi/mcp" \
 && npm install --prefix "$packages" --ignore-scripts --no-audit --no-fund --omit=dev \
      "@earendil-works/pi-coding-agent@${VERSION}" \
      "pi-acp@${ACP_VERSION}" \
 && printf '%s:%s' "$VERSION" "$ACP_VERSION" > "$packages/.versions" \
 && npm cache clean --force \
 && chmod -R a+rX "/var/tmp/vf-pi"

FROM base AS prime-agent
ARG VERSION=033eea1705c5ae953fe7cd7a5e7e7c45317e0dfe
ARG RELEASE_VERSION=0.10.5
RUN dir="/var/tmp/vf-prime-agent" \
 && prefix="$dir/$VERSION" \
 && case "$(uname -m)" in \
      x86_64) platform=linux-x64; sha=b9f653f4c32e83a2270f0b52edea6a72c7bdbd1f7b5c3e64c55e75157094f3b3 ;; \
      aarch64|arm64) platform=linux-arm64; sha=8ddee5f1ed1a4443829944117509212f1294abbf0cbed22765f8a6efefaa2516 ;; \
      *) exit 1 ;; \
    esac \
 && mkdir -p "$prefix/bin" \
 && curl -fsSL "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v${RELEASE_VERSION}/prime-agent-${RELEASE_VERSION}-${platform}.tar.gz" -o /tmp/pa.tgz \
 && echo "$sha  /tmp/pa.tgz" | sha256sum -c - \
 && tar -xzf /tmp/pa.tgz -C "$prefix/bin" \
 && rm -f /tmp/pa.tgz \
 && HOME="$dir/home" PRIME_AGENT_KERNEL_VENV="$prefix/kernel-venv" "$prefix/bin/prime-agent" --prime-agent-bootstrap \
 && set -- && for p in "$prefix/bin/skills/"*/pyproject.toml; do set -- "$@" --editable "${p%/pyproject.toml}"; done \
 && uv pip install --python "$prefix/kernel-venv/bin/python" --no-cache "$@" \
 && touch "$prefix/.ready" \
 && chmod -R a+rX "$dir"

FROM base AS rlm
ARG VERSION=8ca926cdd64fd7725b78cf95a4e1667315d3b1f6
RUN hash=$(printf '%s' "$VERSION" | sha256sum | cut -d' ' -f1) \
 && dir="/var/tmp/vf-rlm-${hash}" \
 && mkdir -p "$dir/checkout" \
 && curl -fsSL "https://codeload.github.com/PrimeIntellect-ai/nano-rlm/tar.gz/${VERSION}" \
    | tar -xz --strip-components=1 -C "$dir/checkout" \
 && UV_CACHE_DIR=/tmp/uv-cache UV_TOOL_BIN_DIR="$dir/bin" UV_TOOL_DIR="$dir/tools" \
      uv tool install --python /usr/local/bin/python3 --editable "$dir/checkout" --with "mcp~=1.28" \
 && rm -rf /tmp/uv-cache \
 && touch "$dir/.ready" \
 && chmod -R a+rX "$dir"

FROM base AS stirrup
ARG VERSION=c092cee80ce09cd1dfb93aef053296a9ecd9dbb6
RUN UV_PYTHON_INSTALL_DIR=/var/tmp/vf-stirrup/python UV_CACHE_DIR=/tmp/uv-cache \
      uv venv --python 3.13 /var/tmp/vf-stirrup/venv \
 && UV_CACHE_DIR=/tmp/uv-cache uv pip install --python /var/tmp/vf-stirrup/venv/bin/python \
      "stirrup[mcp] @ https://github.com/ArtificialAnalysis/Stirrup/archive/${VERSION}.tar.gz" \
      "mcp==1.30.0" \
      "agent-client-protocol==1.0.0" \
 && rm -rf /tmp/uv-cache \
 && printf '%s' "$VERSION" > /var/tmp/vf-stirrup/.version \
 && chmod -R a+rX /var/tmp/vf-stirrup

FROM base AS terminus-2
ARG VERSION=0.26.0
RUN uv pip install --system --no-cache "harbor==${VERSION}"
