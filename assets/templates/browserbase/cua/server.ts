import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { SessionCreateError, sessionManager } from "./sessionManager";
import { executeAction } from "./actionExecutor";
import { captureBrowserState } from "./stateCapture";
import {
  ActionRequest,
  ActionResponse,
  SessionCreateRequest,
  SessionCreateResponse,
  ErrorResponse,
} from "./types";
import { ActionValidationError } from "./actionExecutor";

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRateLimitError(error: unknown): boolean {
  const message = getErrorMessage(error).toLowerCase();
  return (
    message.includes("rate limit") ||
    message.includes("too many requests") ||
    message.includes("status code 429") ||
    message.includes("http 429")
  );
}

function isTimeoutError(error: unknown): boolean {
  const message = getErrorMessage(error).toLowerCase();
  return (
    message.includes("timeout") ||
    message.includes("timed out") ||
    message.includes("etimedout")
  );
}

function buildErrorResponse(
  error: unknown,
  fallbackCode: string,
): { statusCode: number; body: ErrorResponse } {
  if (error instanceof ActionValidationError) {
    return {
      statusCode: 400,
      body: {
        error: error.message,
        code: error.code,
        retryable: false,
        details: error.details,
      },
    };
  }

  const errorMessage = getErrorMessage(error);
  if (isRateLimitError(error)) {
    return {
      statusCode: 429,
      body: {
        error: errorMessage,
        code: "RATE_LIMITED",
        retryable: true,
      },
    };
  }

  if (isTimeoutError(error)) {
    return {
      statusCode: 504,
      body: {
        error: errorMessage,
        code: `${fallbackCode}_TIMEOUT`,
        retryable: true,
      },
    };
  }

  return {
    statusCode: 500,
    body: {
      error: errorMessage,
      code: fallbackCode,
      retryable: false,
    },
  };
}

/**
 * Create and configure the Fastify server with CUA primitive routes
 */
export function createServer(): FastifyInstance {
  // Use simple JSON logging to avoid pino-pretty transport issues in SEA binaries
  // pino-pretty uses dynamic imports that don't work in Single Executable Applications
  const server = Fastify({
    logger: {
      level: process.env.LOG_LEVEL || "info",
    },
  });

  // Health check endpoint
  server.get("/health", async () => {
    return { status: "ok", activeSessions: sessionManager.getActiveSessions().length };
  });

  // List all active sessions
  server.get("/sessions", async () => {
    return { sessions: sessionManager.getActiveSessions() };
  });

  // Create a new browser session
  server.post<{
    Body: SessionCreateRequest;
    Reply: SessionCreateResponse | ErrorResponse;
  }>("/sessions", async (request, reply) => {
    try {
      const session = await sessionManager.createSession(request.body);
      const state = await captureBrowserState(session.page);

      return {
        sessionId: session.id,
        state,
      };
    } catch (error) {
      if (error instanceof SessionCreateError) {
        reply.status(error.statusCode);
        return {
          error: error.message,
          code: error.code,
          retryable: error.retryable,
          statusCode: error.statusCode,
        };
      }
      const { statusCode, body } = buildErrorResponse(error, "SESSION_CREATE_FAILED");
      reply.status(statusCode);
      return body;
    }
  });

  // Get session state
  server.get<{
    Params: { id: string };
    Reply: { state: ReturnType<typeof captureBrowserState> extends Promise<infer T> ? T : never } | ErrorResponse;
  }>("/sessions/:id/state", async (request, reply) => {
    const { id } = request.params;

    const page = await sessionManager.getPage(id);
    if (!page) {
      reply.status(404);
      return { error: `Session ${id} not found`, code: "SESSION_NOT_FOUND" };
    }

    try {
      const state = await captureBrowserState(page);
      return { state };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      reply.status(500);
      return { error: errorMessage, code: "STATE_CAPTURE_FAILED" };
    }
  });

  // Delete a session
  server.delete<{
    Params: { id: string };
    Reply: { success: boolean } | ErrorResponse;
  }>("/sessions/:id", async (request, reply) => {
    const { id } = request.params;

    if (!sessionManager.hasSession(id)) {
      reply.status(404);
      return { error: `Session ${id} not found`, code: "SESSION_NOT_FOUND" };
    }

    try {
      await sessionManager.destroySession(id);
      return { success: true };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      reply.status(500);
      return { error: errorMessage, code: "SESSION_DESTROY_FAILED" };
    }
  });

  // Execute an action on a session
  server.post<{
    Params: { id: string };
    Body: ActionRequest & { tool_call_id?: string };
    Reply: ActionResponse | ErrorResponse;
  }>("/sessions/:id/action", async (request, reply) => {
    const { id } = request.params;
    // Extract tool_call_id from payload (if present) and pass the rest as action
    const { tool_call_id, ...action } = request.body;

    // Log the action being executed with tool_call_id for correlation
    const actionDetails = { ...action };
    // Truncate long text fields for readability
    if (actionDetails.text && actionDetails.text.length > 50) {
      actionDetails.text = actionDetails.text.substring(0, 50) + "...";
    }
    request.log.info(
      { sessionId: id, toolCallId: tool_call_id, action: actionDetails },
      `Executing action: ${action.type}`
    );

    const page = await sessionManager.getPage(id);
    if (!page) {
      reply.status(404);
      return { error: `Session ${id} not found`, code: "SESSION_NOT_FOUND" };
    }

    try {
      // Execute the action with logger for detailed timing
      const result = await executeAction(page, action, request.log);

      // Add delay after action execution (render time)
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Capture state after action (always includes screenshot)
      const state = await captureBrowserState(page);

      return {
        success: result.success,
        error: result.error,
        state,
      };
    } catch (error) {
      // Try to capture state even on error
      let state;
      try {
        state = await captureBrowserState(page);
      } catch {
        state = {
          screenshot: "",
          url: "",
          viewport: { width: 0, height: 0 },
        };
      }

      const { statusCode, body } = buildErrorResponse(error, "ACTION_EXECUTION_FAILED");
      body.state = state;
      reply.status(statusCode);
      return body;
    }
  });

  return server;
}
