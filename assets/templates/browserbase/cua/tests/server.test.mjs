import test from "node:test";
import assert from "node:assert/strict";

import { createServer } from "../server.ts";
import { sessionManager } from "../sessionManager.ts";

function buildFakePage(clickImpl = async () => {}) {
  return {
    async click(...args) {
      return clickImpl(...args);
    },
    async screenshot() {
      return Buffer.from("img");
    },
    mainFrame() {
      return {
        async evaluate() {
          return { w: 800, h: 600 };
        },
      };
    },
    url() {
      return "https://example.com";
    },
  };
}

async function withPatchedPage(page, fn) {
  const originalGetPage = sessionManager.getPage;
  sessionManager.getPage = async () => page;
  try {
    return await fn();
  } finally {
    sessionManager.getPage = originalGetPage;
  }
}

test("action route returns structured validation errors for invalid args", async () => {
  const server = createServer();
  try {
    await withPatchedPage(buildFakePage(), async () => {
      const response = await server.inject({
        method: "POST",
        url: "/sessions/session-1/action",
        payload: {
          type: "click",
          x: "10",
          y: 20,
        },
      });

      assert.equal(response.statusCode, 400);
      const body = JSON.parse(response.body);
      assert.equal(body.code, "INVALID_ACTION_ARGS");
      assert.equal(body.retryable, false);
      assert.match(body.error, /Invalid arguments for click/);
      assert.equal(body.details.length, 1);
      assert.equal(body.details[0].field, "x");
      assert.equal(body.details[0].receivedType, "string");
      assert.equal(body.state.url, "https://example.com");
      assert.deepEqual(body.state.viewport, { width: 800, height: 600 });
    });
  } finally {
    await server.close();
  }
});

test("action route marks rate-limit execution failures as retryable", async () => {
  const server = createServer();
  try {
    await withPatchedPage(
      buildFakePage(async () => {
        throw new Error("rate limit exceeded by upstream provider");
      }),
      async () => {
        const response = await server.inject({
          method: "POST",
          url: "/sessions/session-1/action",
          payload: {
            type: "click",
            x: 10,
            y: 20,
          },
        });

        assert.equal(response.statusCode, 429);
        const body = JSON.parse(response.body);
        assert.equal(body.code, "RATE_LIMITED");
        assert.equal(body.retryable, true);
        assert.match(body.error, /rate limit exceeded/i);
        assert.equal(body.state.url, "https://example.com");
      }
    );
  } finally {
    await server.close();
  }
});
