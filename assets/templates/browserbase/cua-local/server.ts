import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { sessionManager } from "./sessionManager";
import { executeAction } from "./actionExecutor";
import { captureBrowserState } from "./stateCapture";
import {
  ActionRequest,
  ActionResponse,
  SessionCreateRequest,
  SessionCreateResponse,
  ErrorResponse,
} from "./types";

/**
 * Create and configure the Fastify server with CUA primitive routes.
 * Supports both basic CUA primitives and extended inspection actions
 * (get_page_text, read_page, find, form_input).
 */
export function createServer(): FastifyInstance {
  // Use simple JSON logging to avoid pino-pretty transport issues in SEA binaries
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
      const errorMessage = error instanceof Error ? error.message : String(error);
      reply.status(500);
      return { error: errorMessage, code: "SESSION_CREATE_FAILED" };
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

    const session = sessionManager.getSession(id);
    const page = session ? await session.stagehand.context.awaitActivePage() : undefined;
    if (!session || !page) {
      reply.status(404);
      return { error: `Session ${id} not found`, code: "SESSION_NOT_FOUND" };
    }

    try {
      // Execute the action with logger for detailed timing
      // Pass browserContext for CDP access (used by read_page)
      const result = await executeAction(page, action, request.log, session.stagehand.context);

      // Add delay after action execution (render time)
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Capture state after action (always includes screenshot)
      const state = await captureBrowserState(page);

      return {
        success: result.success,
        error: result.error,
        state,
        ...(result.data !== undefined ? { data: result.data } : {}),
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

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

      reply.status(500);
      return {
        success: false,
        error: errorMessage,
        state,
      };
    }
  });

  return server;
}
