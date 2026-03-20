import { Stagehand } from "@browserbasehq/stagehand";
import type { Page } from "@browserbasehq/stagehand";

/**
 * CUA Action Types - Browser primitives for local app interaction.
 * Note: 'goto' is omitted as this mode is for localhost apps without internet access.
 */
export type ActionType =
  | "click"
  | "double_click"
  | "doubleClick"
  | "tripleClick"
  | "type"
  | "keypress"
  | "scroll"
  | "drag"
  | "move"
  | "back"
  | "forward"
  | "wait"
  | "screenshot";

/**
 * Action Request - Sent by external agent to execute a browser primitive
 */
export interface ActionRequest {
  type: ActionType;
  // Mouse/click params
  x?: number;
  y?: number;
  button?: "left" | "right" | "middle";
  clickCount?: number;
  // Type/keyboard params
  text?: string;
  keys?: string | string[];
  // Scroll params
  scroll_x?: number;
  scroll_y?: number;
  // Wait params
  timeMs?: number;
  // Drag params
  path?: Array<{ x: number; y: number }>;
}

/**
 * Viewport dimensions
 */
export interface Viewport {
  width: number;
  height: number;
}

/**
 * Browser State - Full state returned after each action
 */
export interface BrowserState {
  screenshot: string; // base64 PNG
  url: string;
  viewport: Viewport;
}

/**
 * Action Execution Result - Internal result from action executor
 */
export interface ActionExecutionResult {
  success: boolean;
  error?: string;
}

/**
 * Action Response - Full response sent back to external agent
 */
export interface ActionResponse {
  success: boolean;
  error?: string;
  state: BrowserState;
}

/**
 * Session Create Request - Local mode only
 */
export interface SessionCreateRequest {
  viewport?: Viewport;
  /**
   * Path to the Chromium/Chrome executable inside the container.
   * Example: "/usr/bin/chromium"
   */
  executablePath?: string;
  /**
   * CDP WebSocket URL to connect to an already-running browser.
   * Takes precedence over executablePath when both are set.
   * Example: "ws://localhost:9222"
   */
  cdpUrl?: string;
  args?: string[];
  headless?: boolean;
  /**
   * Initial URL to navigate to after session creation.
   * For local app mode, this should be the localhost URL of your app.
   * Example: "http://localhost:3000"
   */
  startUrl?: string;
}

/**
 * Session Create Response
 */
export interface SessionCreateResponse {
  sessionId: string;
  state: BrowserState;
}

/**
 * Browser Session - Internal representation of an active session
 */
export interface BrowserSession {
  id: string;
  stagehand: Stagehand;
  page: Page;
  createdAt: Date;
}

/**
 * Error Response
 */
export interface ErrorResponse {
  error: string;
  code: string;
}
