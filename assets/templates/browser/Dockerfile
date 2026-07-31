# Browser-capable runtime image for the `browser` harness's self-launch mode.
#
# Attach mode (cdp_url set) needs no special image. Self-launch mode (no
# cdp_url) starts a Chromium inside the runtime, so the image must carry one and
# its shared-library closure -- the Playwright install-layer pattern on the
# standard python:3.11-slim base the runtimes default to.
#
# Build and push to any public registry:
#   docker build --platform linux/amd64 -t <registry>/verifiers-browser-runtime:<tag> assets/templates/browser
#   docker push <registry>/verifiers-browser-runtime:<tag>
# then point the runtime at it, e.g. --env.agent.runtime.image <registry>/verifiers-browser-runtime:<tag>.
#
# A ready-to-pull public alternative that already ships Chromium under
# PLAYWRIGHT_BROWSERS_PATH is the official Playwright image, e.g.
#   mcr.microsoft.com/playwright/python:v1.61.0-noble

FROM python:3.11-slim

# Chromium plus its host libraries, at a world-readable shared path so an
# unprivileged agent can run it. Pinned so the launched browser is reproducible.
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright
RUN pip install --no-cache-dir playwright==1.61.0 \
    && playwright install --with-deps chromium \
    && chmod -R a+rX "$PLAYWRIGHT_BROWSERS_PATH"

# uv runs the harness's PEP 723 program.
RUN pip install --no-cache-dir uv

CMD ["python3"]
