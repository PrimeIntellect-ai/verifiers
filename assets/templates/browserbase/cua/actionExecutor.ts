import type { Page } from "@browserbasehq/stagehand";
import {
  ActionRequest,
  ActionExecutionResult,
  ValidationErrorDetail,
} from "./types";

/**
 * Logger interface for structured logging (compatible with Fastify logger)
 */
export interface ActionLogger {
  info: (obj: object, msg?: string) => void;
  error: (obj: object, msg?: string) => void;
}

export class ActionValidationError extends Error {
  readonly code = "INVALID_ACTION_ARGS";
  readonly details: ValidationErrorDetail[];

  constructor(actionType: string, details: ValidationErrorDetail[]) {
    super(`Invalid arguments for ${actionType}`);
    this.name = "ActionValidationError";
    this.details = details;
  }
}

/**
 * Key mapping for converting various key representations to Playwright-compatible names
 */
const KEY_MAP: Record<string, string> = {
  ENTER: "Enter",
  RETURN: "Enter",
  ESCAPE: "Escape",
  ESC: "Escape",
  BACKSPACE: "Backspace",
  TAB: "Tab",
  SPACE: " ",
  DELETE: "Delete",
  DEL: "Delete",
  ARROWUP: "ArrowUp",
  ARROWDOWN: "ArrowDown",
  ARROWLEFT: "ArrowLeft",
  ARROWRIGHT: "ArrowRight",
  ARROW_UP: "ArrowUp",
  ARROW_DOWN: "ArrowDown",
  ARROW_LEFT: "ArrowLeft",
  ARROW_RIGHT: "ArrowRight",
  UP: "ArrowUp",
  DOWN: "ArrowDown",
  LEFT: "ArrowLeft",
  RIGHT: "ArrowRight",
  SHIFT: "Shift",
  CONTROL: "Control",
  CTRL: "Control",
  ALT: "Alt",
  OPTION: "Alt",
  META: "Meta",
  COMMAND: "Meta",
  CMD: "Meta",
  SUPER: "Meta",
  WINDOWS: "Meta",
  WIN: "Meta",
  HOME: "Home",
  END: "End",
  PAGEUP: "PageUp",
  PAGEDOWN: "PageDown",
  PAGE_UP: "PageUp",
  PAGE_DOWN: "PageDown",
  PGUP: "PageUp",
  PGDN: "PageDown",
};

function mapKeyToPlaywright(key: string): string {
  if (!key) return key;
  const upperKey = key.toUpperCase();
  return KEY_MAP[upperKey] || key;
}

function describeValueType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function integerFieldError(
  field: string,
  value: unknown,
  expected = "an integer"
): ValidationErrorDetail {
  return {
    field,
    expected,
    receivedType: describeValueType(value),
    receivedValue: value,
    message: `${field} must be ${expected}, received ${JSON.stringify(value)} (${describeValueType(value)})`,
  };
}

function stringFieldError(
  field: string,
  value: unknown,
  expected = "a string"
): ValidationErrorDetail {
  return {
    field,
    expected,
    receivedType: describeValueType(value),
    receivedValue: value,
    message: `${field} must be ${expected}, received ${JSON.stringify(value)} (${describeValueType(value)})`,
  };
}

function validateClickLikeArgs(
  actionType: string,
  x: unknown,
  y: unknown
): { x: number; y: number } {
  const details: ValidationErrorDetail[] = [];
  if (!Number.isInteger(x)) {
    details.push(integerFieldError("x", x, "an integer pixel coordinate"));
  }
  if (!Number.isInteger(y)) {
    details.push(integerFieldError("y", y, "an integer pixel coordinate"));
  }
  if (details.length > 0) {
    throw new ActionValidationError(actionType, details);
  }
  return { x: x as number, y: y as number };
}

function validateOptionalScrollInteger(
  field: string,
  value: unknown,
  defaultValue: number,
  expected = "an integer"
): number {
  if (value === undefined) {
    return defaultValue;
  }
  if (!Number.isInteger(value)) {
    throw new ActionValidationError("scroll", [
      integerFieldError(field, value, expected),
    ]);
  }
  return value as number;
}

/**
 * ActionExecutor
 *
 * Executes CUA browser primitives on a Page object.
 * Adapted from V3CuaAgentHandler.executeAction logic.
 */
export async function executeAction(
  page: Page,
  action: ActionRequest,
  logger?: ActionLogger,
): Promise<ActionExecutionResult> {
  const startTime = Date.now();
  
  // Log action start with parameters (truncate long text values)
  const logParams = { ...action };
  if (logParams.text && typeof logParams.text === "string" && logParams.text.length > 50) {
    logParams.text = logParams.text.substring(0, 50) + "...";
  }
  logger?.info(
    { action: action.type, params: logParams },
    `[Action] Starting: ${action.type}`
  );

  try {
    let result: ActionExecutionResult;
    
    switch (action.type) {
      case "click": {
        const { x, y, button = "left", clickCount = 1 } = action;
        const coords = validateClickLikeArgs("click", x, y);
        await page.click(coords.x, coords.y, {
          button: button as "left" | "right" | "middle",
          clickCount,
        });
        result = { success: true };
        break;
      }

      case "double_click":
      case "doubleClick": {
        const { x, y } = action;
        const coords = validateClickLikeArgs("double_click", x, y);
        await page.click(coords.x, coords.y, {
          button: "left",
          clickCount: 2,
        });
        result = { success: true };
        break;
      }

      case "tripleClick": {
        const { x, y } = action;
        const coords = validateClickLikeArgs("tripleClick", x, y);
        await page.click(coords.x, coords.y, {
          button: "left",
          clickCount: 3,
        });
        result = { success: true };
        break;
      }

      case "type": {
        const { text } = action;
        if (typeof text !== "string") {
          throw new ActionValidationError("type", [
            stringFieldError("text", text, "a string to type"),
          ]);
        }
        await page.type(text);
        result = { success: true };
        break;
      }

      case "keypress": {
        const { keys } = action;
        if (!keys) {
          throw new ActionValidationError("keypress", [
            stringFieldError("keys", keys, "a string or array of strings"),
          ]);
        }
        if (
          typeof keys !== "string" &&
          (!Array.isArray(keys) || keys.some((key) => typeof key !== "string"))
        ) {
          throw new ActionValidationError("keypress", [
            stringFieldError("keys", keys, "a string or array of strings"),
          ]);
        }
        const keyList = Array.isArray(keys) ? keys : [keys];
        for (const rawKey of keyList) {
          const mapped = mapKeyToPlaywright(String(rawKey));
          await page.keyPress(mapped);
        }
        result = { success: true };
        break;
      }

      case "scroll": {
        const x = validateOptionalScrollInteger("x", action.x, 0, "an integer pixel coordinate");
        const y = validateOptionalScrollInteger("y", action.y, 0, "an integer pixel coordinate");
        const scroll_x = validateOptionalScrollInteger("scroll_x", action.scroll_x, 0, "an integer pixel delta");
        const scroll_y = validateOptionalScrollInteger("scroll_y", action.scroll_y, 0, "an integer pixel delta");
        await page.scroll(
          x,
          y,
          scroll_x,
          scroll_y,
        );
        result = { success: true };
        break;
      }

      case "drag": {
        const { path } = action;
        if (!Array.isArray(path) || path.length < 2) {
          result = {
            success: false,
            error: "drag requires path array with at least 2 points",
          };
          break;
        }
        const start = path[0];
        const end = path[path.length - 1];
        await page.dragAndDrop(start.x, start.y, end.x, end.y, {
          steps: Math.min(20, Math.max(5, path.length)),
          delay: 10,
        });
        result = { success: true };
        break;
      }

      case "move": {
        // No direct cursor-only move in the Page API
        // This is a no-op similar to V3CuaAgentHandler
        result = { success: true };
        break;
      }

      case "wait": {
        const time = action.timeMs ?? 1000;
        if (!Number.isInteger(time) || time < 0) {
          throw new ActionValidationError("wait", [
            integerFieldError("timeMs", time, "a non-negative integer number of milliseconds"),
          ]);
        }
        await new Promise((r) => setTimeout(r, time));
        result = { success: true };
        break;
      }

      case "screenshot": {
        // Screenshot is handled separately in state capture
        // This is a no-op as the response always includes a screenshot
        result = { success: true };
        break;
      }

      case "goto": {
        const { url } = action;
        if (typeof url !== "string") {
          throw new ActionValidationError("goto", [
            stringFieldError("url", url, "a URL string"),
          ]);
        }
        await page.goto(url, { waitUntil: "load" });
        result = { success: true };
        break;
      }

      case "back": {
        await page.goBack();
        result = { success: true };
        break;
      }

      case "forward": {
        await page.goForward();
        result = { success: true };
        break;
      }

      default:
        result = {
          success: false,
          error: `Unknown action type: ${action.type}`,
        };
    }
    
    // Log action completion with timing
    const duration = Date.now() - startTime;
    if (result.success) {
      logger?.info(
        { action: action.type, duration, success: true },
        `[Action] Completed: ${action.type} in ${duration}ms`
      );
    } else {
      logger?.error(
        { action: action.type, duration, success: false, error: result.error },
        `[Action] Failed: ${action.type} after ${duration}ms - ${result.error}`
      );
    }
    
    return result;
  } catch (error) {
    const duration = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : String(error);
    
    logger?.error(
      { action: action.type, duration, success: false, error: errorMessage },
      `[Action] Exception: ${action.type} after ${duration}ms - ${errorMessage}`
    );

    throw error;
  }
}
