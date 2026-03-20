/**
 * CUA Local Mode Server
 *
 * A variant of the CUA Primitives API Server designed for localhost applications.
 * This server runs alongside your local app (e.g., Next.js on port 3000) and
 * provides browser automation primitives without internet access.
 *
 * Key differences from the standard CUA server:
 * - No 'goto' action - browser starts at the configured startUrl
 * - Always uses LOCAL mode (no Browserbase)
 * - Designed to run in a sandbox with the target application
 *
 * Usage (standalone):
 *   pnpm dev                              # Start with hot reload
 *   pnpm start                            # Start without hot reload
 *   CUA_SERVER_PORT=3001 pnpm dev         # Custom port (use 3001 to avoid conflict with app on 3000)
 *
 * Environment variables:
 *   CUA_SERVER_PORT - Server port (default: 3001)
 *   CUA_SERVER_HOST - Server host (default: 0.0.0.0)
 */

import { createServer } from "./server";
import { sessionManager } from "./sessionManager";

const PORT = parseInt(process.env.CUA_SERVER_PORT || "3001", 10);
const HOST = process.env.CUA_SERVER_HOST || "0.0.0.0";

async function main() {
  const server = createServer();

  // Graceful shutdown handling
  const shutdown = async (signal: string) => {
    const activeSessions = sessionManager.getActiveSessions();
    console.log(`\n╔════════════════════════════════════════════════════════════╗`);
    console.log(`║  Received ${signal}. Shutting down gracefully...              ║`);
    console.log(`║  Active sessions to close: ${activeSessions.length.toString().padEnd(29)}║`);
    console.log(`╚════════════════════════════════════════════════════════════╝`);

    try {
      // Close all browser sessions
      if (activeSessions.length > 0) {
        console.log(`\nClosing ${activeSessions.length} browser session(s)...`);
        for (const sessionId of activeSessions) {
          console.log(`  - Closing session: ${sessionId}`);
        }
      }
      await sessionManager.destroyAllSessions();
      console.log("All browser sessions closed.");

      // Close the server
      await server.close();
      console.log("Server closed.");
      console.log("\nGoodbye!\n");

      process.exit(0);
    } catch (error) {
      console.error("Error during shutdown:", error);
      process.exit(1);
    }
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  try {
    await server.listen({ port: PORT, host: HOST });
    console.log(`
╔════════════════════════════════════════════════════════════╗
║              CUA Local Mode Server                         ║
╠════════════════════════════════════════════════════════════╣
║  Server running at http://${HOST}:${PORT}                     ║
║                                                            ║
║  Endpoints:                                                ║
║    GET  /health              - Health check                ║
║    GET  /sessions            - List active sessions        ║
║    POST /sessions            - Create browser session      ║
║    DELETE /sessions/:id      - Close browser session       ║
║    GET  /sessions/:id/state  - Get browser state           ║
║    POST /sessions/:id/action - Execute CUA primitive       ║
║                                                            ║
║  Note: 'goto' action is disabled in local mode.            ║
║  Browser will start at the configured startUrl.            ║
║                                                            ║
║  Press Ctrl+C to stop                                      ║
╚════════════════════════════════════════════════════════════╝
`);
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
}

main();
