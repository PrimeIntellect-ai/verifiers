FROM docker.io/library/python:3.14-bookworm
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/bin/node /usr/local/bin/node
COPY --from=docker.io/library/node:24.22.3-bookworm /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/
RUN ln -sf /usr/local/bin/node /usr/local/bin/nodejs \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends curl ca-certificates git ripgrep tmux \
 && rm -rf /var/lib/apt/lists/* \
 && for tool in python3 node npm npx uv uvx curl git rg; do "$tool" --version || exit 1; done \
 && tmux -V
