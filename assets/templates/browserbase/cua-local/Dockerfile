# Dockerfile
#
# Combined sandbox image for running the CUA local server alongside a
# Next.js (or other) web application. Both services run within the same
# container for simplified sandbox deployment.
#
# Architecture:
#   - CUA local server runs on port 3001 (CUA_SERVER_PORT)
#   - Target application runs on port 3000 (APP_PORT)
#   - Browser is started pointing at http://localhost:3000
#
# Build:
#   docker build --platform linux/amd64 -t cua-local:latest .
#
# Requires:
#   - CUA server binary built: pnpm build:binary:docker (from cua-local/)
#   - Next.js app at /app/nextjs-app when running (or mounted via sandbox)

FROM --platform=linux/amd64 node:20-bookworm-slim

# Install Chromium and runtime dependencies
RUN apt-get update -qq && \
    apt-get install -y -qq \
        chromium \
        fonts-noto-color-emoji \
        fonts-liberation \
        curl \
        supervisor \
        --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Tell anything that looks for Chrome where to find it
ENV CHROMIUM_PATH=/usr/bin/chromium
# Prevent Playwright/Puppeteer from trying to download their own bundle
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=true

# ── CUA local server ─────────────────────────────────────────────────────────
WORKDIR /app/cua-local-server

COPY dist/sea/cua-local-server-linux-x64 ./cua-local-server-linux-x64
RUN chmod +x ./cua-local-server-linux-x64

# CUA server runs on 3001 so it doesn't collide with the app on 3000
ENV CUA_SERVER_PORT=3001
ENV CUA_SERVER_HOST=0.0.0.0

# ── Application placeholder ──────────────────────────────────────────────────
# The actual Next.js app will be uploaded to /app/nextjs-app by the environment
WORKDIR /app/nextjs-app
ENV APP_PORT=3000

# ── Supervisor configuration ─────────────────────────────────────────────────
# Supervisor manages both the CUA server and the Next.js app
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

WORKDIR /app

EXPOSE 3000 3001

# Start supervisor which manages both services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
