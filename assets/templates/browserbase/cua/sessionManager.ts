import { Stagehand } from "@browserbasehq/stagehand";
import type { Page } from "@browserbasehq/stagehand";
import { BrowserSession, SessionCreateRequest } from "./types";

const DEFAULT_MAX_CONCURRENT_CREATES = 2;
const DEFAULT_MAX_PENDING_CREATES = 200;

/**
 * Generates a unique session ID
 */
function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message || error.name;
  }
  if (typeof error === "string") {
    return error;
  }
  return "unknown session creation error";
}

function inferStatusCode(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) {
    return undefined;
  }
  const obj = error as Record<string, unknown>;
  const statusCandidate = obj.status ?? obj.statusCode;
  if (typeof statusCandidate === "number" && Number.isFinite(statusCandidate)) {
    return statusCandidate;
  }
  const response = obj.response;
  if (typeof response === "object" && response !== null) {
    const responseStatus = (response as Record<string, unknown>).status;
    if (typeof responseStatus === "number" && Number.isFinite(responseStatus)) {
      return responseStatus;
    }
  }
  return undefined;
}

export class SessionCreateError extends Error {
  readonly code: string;
  readonly statusCode: number;
  readonly retryable: boolean;

  constructor(
    message: string,
    opts?: { code?: string; statusCode?: number; retryable?: boolean },
  ) {
    super(message);
    this.name = "SessionCreateError";
    this.code = opts?.code ?? "SESSION_CREATE_FAILED";
    this.statusCode = opts?.statusCode ?? 500;
    this.retryable = opts?.retryable ?? true;
  }
}

function classifySessionCreateError(error: unknown): SessionCreateError {
  const statusCode = inferStatusCode(error);
  const message = describeError(error);

  if (statusCode === 429) {
    return new SessionCreateError(message, {
      code: "SESSION_RATE_LIMITED",
      statusCode: 429,
      retryable: true,
    });
  }
  if (statusCode === 401 || statusCode === 403) {
    return new SessionCreateError(message, {
      code: "SESSION_AUTH_FAILED",
      statusCode,
      retryable: false,
    });
  }
  if (statusCode === 502 || statusCode === 503 || statusCode === 504) {
    return new SessionCreateError(message, {
      code: "SESSION_PROVIDER_UNAVAILABLE",
      statusCode,
      retryable: true,
    });
  }
  if (typeof statusCode === "number" && statusCode >= 500) {
    return new SessionCreateError(message, {
      code: "SESSION_PROVIDER_ERROR",
      statusCode,
      retryable: true,
    });
  }
  if (typeof statusCode === "number" && statusCode >= 400) {
    return new SessionCreateError(message, {
      code: "SESSION_CREATE_INVALID_REQUEST",
      statusCode,
      retryable: false,
    });
  }

  return new SessionCreateError(message, {
    code: "SESSION_CREATE_FAILED",
    statusCode: 503,
    retryable: true,
  });
}

/**
 * BrowserSessionManager
 *
 * Manages multiple Stagehand browser instances by session ID.
 * Handles creation, retrieval, and cleanup of browser sessions.
 */
export class BrowserSessionManager {
  private sessions: Map<string, BrowserSession> = new Map();
  private inFlightCreates = 0;
  private pendingCreateResolvers: Array<() => void> = [];
  private readonly maxConcurrentCreates = parsePositiveInt(
    process.env.CUA_SESSION_CREATE_MAX_CONCURRENT,
    DEFAULT_MAX_CONCURRENT_CREATES,
  );
  private readonly maxPendingCreates = parsePositiveInt(
    process.env.CUA_SESSION_CREATE_MAX_PENDING,
    DEFAULT_MAX_PENDING_CREATES,
  );

  private async acquireCreateSlot(): Promise<void> {
    if (this.inFlightCreates < this.maxConcurrentCreates) {
      this.inFlightCreates += 1;
      return;
    }

    if (this.pendingCreateResolvers.length >= this.maxPendingCreates) {
      throw new SessionCreateError(
        `Session creation queue is full (pending=${this.pendingCreateResolvers.length})`,
        {
          code: "SESSION_CREATE_QUEUE_FULL",
          statusCode: 503,
          retryable: true,
        },
      );
    }

    await new Promise<void>((resolve) => {
      this.pendingCreateResolvers.push(resolve);
    });
    this.inFlightCreates += 1;
  }

  private releaseCreateSlot(): void {
    this.inFlightCreates = Math.max(0, this.inFlightCreates - 1);
    const next = this.pendingCreateResolvers.shift();
    if (next) {
      next();
    }
  }

  /**
   * Create a new browser session
   */
  async createSession(options?: SessionCreateRequest): Promise<BrowserSession> {
    const sessionId = generateSessionId();
    const startTime = Date.now();
    const envType = options?.env ?? "LOCAL";
    await this.acquireCreateSlot();
    console.log(
      `[Session] Creating ${sessionId} with env: ${envType}, proxies: ${options?.proxies ?? false}, in_flight_creates: ${this.inFlightCreates}, queued_creates: ${this.pendingCreateResolvers.length}`,
    );

    let stagehand: Stagehand | null = null;
    try {
      // TODO: Update to accept modelApiKey from client request (MODEL_API_KEY) instead of
      // hardcoding OPENAI_API_KEY. This will allow using different model providers.
      // See: SessionCreateRequest in types.ts, cua_mode.py session_config
      // Stagehand runtime accepts modelApiKey, but some published typings omit it.
      // Keep runtime behavior while avoiding type drift failures.
      stagehand = new Stagehand({
        env: envType,
        apiKey: options?.browserbaseApiKey,
        projectId: options?.browserbaseProjectId,
        modelApiKey: process.env.OPENAI_API_KEY,
        verbose: 1,
        disablePino: true, // Disable pino logging to avoid pino-pretty transport issues in SEA binaries
        browserbaseSessionCreateParams:
          envType === "BROWSERBASE"
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
        localBrowserLaunchOptions:
          envType === "LOCAL"
            ? {
                viewport: options?.viewport
                  ? {
                      width: options.viewport.width,
                      height: options.viewport.height,
                    }
                  : { width: 1024, height: 768 },
              }
            : undefined,
      } as any);

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
      console.log(
        `[Session] Created ${sessionId} in ${duration}ms (env: ${envType}, active sessions: ${this.sessions.size})`,
      );

      return session;
    } catch (error) {
      const classified =
        error instanceof SessionCreateError
          ? error
          : classifySessionCreateError(error);
      console.error(
        `[Session] Failed to create ${sessionId}: ${classified.code} (${classified.statusCode}) - ${classified.message}`,
      );
      if (stagehand) {
        try {
          await stagehand.close();
        } catch {
          // no-op: best effort cleanup
        }
      }
      throw classified;
    } finally {
      this.releaseCreateSlot();
    }
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
