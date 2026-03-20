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
 * Manages Stagehand browser instances in LOCAL mode only.
 * Designed for controlling localhost applications without internet access.
 */
export class BrowserSessionManager {
  private sessions: Map<string, BrowserSession> = new Map();

  /**
   * Create a new browser session
   */
  async createSession(options?: SessionCreateRequest): Promise<BrowserSession> {
    const sessionId = generateSessionId();
    const startTime = Date.now();

    console.log(`[Session] Creating ${sessionId} in LOCAL mode`);

    // Build localBrowserLaunchOptions for LOCAL mode.
    // executablePath and cdpUrl are forwarded when provided so that
    // callers running inside a container can point Stagehand at the
    // correct Chromium binary or an already-running browser.
    const localLaunchOptions = {
      viewport: options?.viewport
        ? {
            width: options.viewport.width,
            height: options.viewport.height,
          }
        : { width: 1024, height: 768 },
      headless: options?.headless ?? true,
      args: options?.args ?? [
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-setuid-sandbox",
      ],
      ...(options?.cdpUrl ? { cdpUrl: options.cdpUrl } : {}),
      ...(options?.executablePath && !options?.cdpUrl
        ? { executablePath: options.executablePath }
        : {}),
    };

    const stagehand = new Stagehand({
      env: "LOCAL",
      modelApiKey: process.env.OPENAI_API_KEY,
      verbose: 1,
      disablePino: true,
      localBrowserLaunchOptions: localLaunchOptions,
    });

    await stagehand.init();

    const page = stagehand.context.pages()[0];

    // Navigate to start URL if provided (e.g., localhost:3000 for the target app)
    if (options?.startUrl) {
      console.log(`[Session] Navigating to start URL: ${options.startUrl}`);
      await page.goto(options.startUrl, { waitUntil: "load" });
    }

    const session: BrowserSession = {
      id: sessionId,
      stagehand,
      page,
      createdAt: new Date(),
    };

    this.sessions.set(sessionId, session);

    const duration = Date.now() - startTime;
    console.log(
      `[Session] Created ${sessionId} in ${duration}ms (active sessions: ${this.sessions.size})`
    );

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
    console.log(
      `[Session] Destroyed ${sessionId} in ${duration}ms (remaining sessions: ${this.sessions.size})`
    );

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
