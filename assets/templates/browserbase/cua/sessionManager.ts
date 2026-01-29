import { Stagehand } from "@browserbasehq/stagehand";
import type { Page } from "@browserbasehq/stagehand";
import { BrowserSession, SessionCreateRequest } from "./types";

/**
 * Generates a unique session ID
 */
function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * BrowserSessionManager
 *
 * Manages multiple Stagehand browser instances by session ID.
 * Handles creation, retrieval, and cleanup of browser sessions.
 */
export class BrowserSessionManager {
  private sessions: Map<string, BrowserSession> = new Map();

  /**
   * Create a new browser session
   */
  async createSession(options?: SessionCreateRequest): Promise<BrowserSession> {
    const sessionId = generateSessionId();
    const startTime = Date.now();
    const envType = options?.env ?? "LOCAL";
    
    console.log(`[Session] Creating ${sessionId} with env: ${envType}, proxies: ${options?.proxies ?? false}`);

    // TODO: Update to accept modelApiKey from client request (MODEL_API_KEY) instead of
    // hardcoding OPENAI_API_KEY. This will allow using different model providers.
    // See: SessionCreateRequest in types.ts, cua_mode.py session_config
    const stagehand = new Stagehand({
      env: envType,
      apiKey: options?.browserbaseApiKey,
      projectId: options?.browserbaseProjectId,
      modelApiKey: process.env.OPENAI_API_KEY,
      verbose: 1,
      disablePino: true, // Disable pino logging to avoid pino-pretty transport issues in SEA binaries
      browserbaseSessionCreateParams: envType === "BROWSERBASE"
        ? {
            projectId: options?.browserbaseProjectId,
            proxies: options?.proxies ?? false,
            browserSettings: {
              viewport: options?.viewport
                ? {
                    width: options.viewport.width,
                    height: options.viewport.height,
                  }
                : { width: 1024, height: 768 },
            },
          }
        : undefined,
      // Only provide localBrowserLaunchOptions for LOCAL mode to avoid Chrome validation in BROWSERBASE mode
      localBrowserLaunchOptions: envType === "LOCAL"
        ? {
            viewport: options?.viewport
              ? {
                  width: options.viewport.width,
                  height: options.viewport.height,
                }
              : { width: 1024, height: 768 },
          }
        : undefined,
    });

    await stagehand.init();

    const page = stagehand.context.pages()[0];

    const session: BrowserSession = {
      id: sessionId,
      stagehand,
      page,
      createdAt: new Date(),
    };

    this.sessions.set(sessionId, session);
    
    const duration = Date.now() - startTime;
    console.log(`[Session] Created ${sessionId} in ${duration}ms (env: ${envType}, active sessions: ${this.sessions.size})`);

    return session;
  }

  /**
   * Get an existing session by ID
   */
  getSession(sessionId: string): BrowserSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Check if a session exists
   */
  hasSession(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }

  /**
   * Destroy a session and close its browser
   */
  async destroySession(sessionId: string): Promise<boolean> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      console.log(`[Session] Destroy requested for non-existent session: ${sessionId}`);
      return false;
    }

    const startTime = Date.now();
    console.log(`[Session] Destroying ${sessionId}`);

    try {
      await session.stagehand.close();
    } catch (error) {
      console.error(`[Session] Error closing ${sessionId}:`, error);
    }

    this.sessions.delete(sessionId);
    
    const duration = Date.now() - startTime;
    console.log(`[Session] Destroyed ${sessionId} in ${duration}ms (remaining sessions: ${this.sessions.size})`);
    
    return true;
  }

  /**
   * Get all active session IDs
   */
  getActiveSessions(): string[] {
    return Array.from(this.sessions.keys());
  }

  /**
   * Get the page for a session
   */
  async getPage(sessionId: string): Promise<Page | undefined> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return undefined;
    }
    // Always get the active page in case it changed
    return await session.stagehand.context.awaitActivePage();
  }

  /**
   * Destroy all sessions (cleanup on server shutdown)
   */
  async destroyAllSessions(): Promise<void> {
    const sessionIds = Array.from(this.sessions.keys());
    if (sessionIds.length > 0) {
      console.log(`[Session] Destroying all ${sessionIds.length} sessions...`);
      await Promise.all(sessionIds.map((id) => this.destroySession(id)));
      console.log(`[Session] All sessions destroyed`);
    }
  }
}

// Singleton instance
export const sessionManager = new BrowserSessionManager();
