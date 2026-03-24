import type { Page } from "@browserbasehq/stagehand";
import { ActionRequest, ActionExecutionResult } from "./types";

/**
 * Logger interface for structured logging (compatible with Fastify logger)
 */
export interface ActionLogger {
  info: (obj: object, msg?: string) => void;
  error: (obj: object, msg?: string) => void;
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

// ======================== read_page helpers ========================

/** Interactive ARIA roles for filter="interactive" */
const INTERACTIVE_ROLES = new Set([
  "button", "link", "textbox", "searchbox", "combobox", "listbox",
  "option", "checkbox", "radio", "switch", "slider", "spinbutton",
  "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
  "treeitem", "gridcell", "columnheader", "rowheader",
  "input", "select", "textarea",  // HTML-inferred roles
]);

/**
 * read_page via Chrome DevTools Protocol – uses the browser's real
 * accessibility tree rather than walking the DOM ourselves.
 *
 * Flow:
 * 1. page.evaluate() → assign data-ref to every visible element, collect
 *    a coordinate + metadata map keyed by data-ref.
 * 2. CDP DOM.getDocument(depth=-1) → build backendNodeId → data-ref map.
 * 3. CDP Accessibility.getFullAXTree → get the real AX tree with roles,
 *    names, properties, and backendDOMNodeId for each node.
 * 4. Walk the AX tree, joining coordinates from step 1 via the maps from
 *    steps 2 and 3.
 */
async function readPageViaCDP(
  page: Page,
  browserContext: any /* BrowserContext */ | undefined,
  filterMode: string,
  maxDepth: number,
  focusRef: string | null,
  maxOutput: number,
): Promise<string> {
  if (!browserContext) throw new Error("No browser context for CDP");

  // --- Step 1: assign data-ref attrs & collect coordinates ---------------
  const elemData: Record<string, { x: number; y: number; href?: string; inputType?: string }> =
    await page.mainFrame().evaluate(() => {
      const data: Record<string, { x: number; y: number; href?: string; inputType?: string }> = {};
      let refCounter = 0;
      for (const el of document.querySelectorAll("*")) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        const ref = `ref_${++refCounter}`;
        el.setAttribute("data-ref", ref);
        const entry: any = {
          x: Math.round(rect.x + rect.width / 2),
          y: Math.round(rect.y + rect.height / 2),
        };
        const href = el.getAttribute("href");
        if (href) entry.href = href;
        if (el.tagName === "INPUT") entry.inputType = (el as HTMLInputElement).type;
        data[ref] = entry;
      }
      return data;
    });

  // --- Step 2 & 3: CDP calls (parallel) ----------------------------------
  const cdp = await (browserContext as any).newCDPSession(page);
  try {
    const [domResult, axResult] = await Promise.all([
      cdp.send("DOM.getDocument", { depth: -1, pierce: true }),
      cdp.send("Accessibility.getFullAXTree"),
    ]);

    // Flatten DOM tree → backendNodeId → data-ref
    const backendToRef = new Map<number, string>();
    function walkDom(node: any) {
      if (node.nodeType === 1 && node.attributes) {
        for (let i = 0; i < node.attributes.length; i += 2) {
          if (node.attributes[i] === "data-ref") {
            backendToRef.set(node.backendNodeId, node.attributes[i + 1]);
            break;
          }
        }
      }
      if (node.children) for (const c of node.children) walkDom(c);
      if (node.shadowRoots) for (const c of node.shadowRoots) walkDom(c);
      if (node.contentDocument) walkDom(node.contentDocument);
    }
    walkDom(domResult.root);

    // --- Step 4: build and format the AX tree ----------------------------
    const axNodes: any[] = axResult.nodes;
    const axMap = new Map<string, any>();
    for (const n of axNodes) axMap.set(n.nodeId, n);

    // Determine root(s)
    let rootIds: string[];
    if (focusRef) {
      // Find the AX node whose DOM element has the given data-ref
      let targetBackendId: number | undefined;
      for (const [bid, ref] of backendToRef) {
        if (ref === focusRef) { targetBackendId = bid; break; }
      }
      if (targetBackendId === undefined) {
        return `Element with ref ${focusRef} not found. Call read_page without ref_id first.`;
      }
      const targetAx = axNodes.find(n => n.backendDOMNodeId === targetBackendId);
      rootIds = targetAx ? [targetAx.nodeId] : [];
    } else {
      rootIds = axNodes.filter(n => !n.parentId).map(n => n.nodeId);
    }

    const lines: string[] = [];
    let charCount = 0;
    let truncated = false;

    function formatProps(node: any, ref: string | undefined): string {
      const parts: string[] = [];
      if (node.properties) {
        for (const p of node.properties) {
          const v = p.value?.value;
          switch (p.name) {
            case "focused": if (v) parts.push("focused"); break;
            case "expanded": parts.push(`expanded=${v ? "True" : "False"}`); break;
            case "checked":
              if (v === "true" || v === true) parts.push("checked=true");
              else if (v === "mixed") parts.push("checked=mixed");
              break;
            case "disabled": if (v) parts.push("disabled"); break;
            case "required": if (v) parts.push("required"); break;
            case "invalid": if (v && v !== "false") parts.push(`invalid="${v}"`); break;
            case "hasPopup": if (v && v !== "false") parts.push(`hasPopup="${v}"`); break;
            case "level": parts.push(`level=${v}`); break;
          }
        }
      }
      // URL from DOM for links
      const coords = ref ? elemData[ref] : undefined;
      if (coords?.href) parts.push(`url="${coords.href}"`);
      if (coords?.inputType && coords.inputType !== "text") parts.push(`type=${coords.inputType}`);
      return parts.length > 0 ? " " + parts.join(" ") : "";
    }

    function walkAx(nodeId: string, depth: number) {
      if (truncated || depth > maxDepth) return;
      const node = axMap.get(nodeId);
      if (!node) return;

      // Skip ignored nodes but process their children
      if (node.ignored) {
        if (node.childIds) for (const cid of node.childIds) walkAx(cid, depth);
        return;
      }

      const role: string = node.role?.value || "generic";
      const name: string = node.name?.value || "";

      // filter="interactive": skip non-interactive, walk children
      if (filterMode === "interactive" && !INTERACTIVE_ROLES.has(role)) {
        if (node.childIds) for (const cid of node.childIds) walkAx(cid, depth);
        return;
      }

      // Resolve ref + coordinates
      const ref = node.backendDOMNodeId != null
        ? backendToRef.get(node.backendDOMNodeId) : undefined;
      const coords = ref ? elemData[ref] : undefined;

      const indent = "  ".repeat(depth);
      const nameStr = name ? ` "${name}"` : "";
      const refStr = ref ? ` [ref=${ref}]` : "";
      const coordStr = coords ? ` (x=${coords.x},y=${coords.y})` : "";
      const attrs = formatProps(node, ref);

      const line = `${indent}- ${role}${nameStr}${refStr}${coordStr}${attrs}`;
      charCount += line.length + 1;
      if (charCount > maxOutput) { truncated = true; return; }
      lines.push(line);

      if (node.childIds) {
        for (const cid of node.childIds) walkAx(cid, depth + 1);
      }
    }

    for (const rid of rootIds) walkAx(rid, 0);

    if (truncated) {
      return `Output exceeds ${maxOutput} characters. Try filter="interactive" to get only clickable elements, use a smaller depth, or specify ref_id to focus on a specific element.`;
    }
    return lines.join("\n");

  } finally {
    await cdp.detach();
  }
}

/**
 * Fallback read_page using DOM walking (no CDP required).
 * Less accurate roles and names, but works if CDP is unavailable.
 */
async function readPageFallback(
  page: Page,
  filterMode: string,
  maxDepth: number,
  focusRef: string | null,
  maxOutput: number,
): Promise<string> {
  return page.mainFrame().evaluate(
    ({ filterMode, maxDepth, focusRef, maxOutput }) => {
      const MAX_OUTPUT = maxOutput;
      let refCounter = 0;
      let charCount = 0;
      let truncated = false;
      const lines: string[] = [];

      function getRole(el: Element): string {
        const ariaRole = el.getAttribute("role");
        if (ariaRole) return ariaRole;
        const tag = el.tagName.toLowerCase();
        const m: Record<string, string> = {
          a: "link", button: "button", input: "input", select: "select",
          textarea: "textarea", img: "image", h1: "heading", h2: "heading",
          h3: "heading", h4: "heading", h5: "heading", h6: "heading",
          nav: "navigation", main: "main", header: "banner", footer: "contentinfo",
          aside: "complementary", section: "region", form: "form", table: "table",
          ul: "list", ol: "list", li: "listitem", dialog: "dialog",
        };
        return m[tag] || "generic";
      }

      function getLabel(el: Element): string {
        return el.getAttribute("aria-label")
          || el.getAttribute("title")
          || el.getAttribute("placeholder")
          || el.getAttribute("alt")
          || (() => {
            const t = el.textContent?.trim() || "";
            return t.length > 80 ? t.substring(0, 77) + "..." : t;
          })();
      }

      function isInteractive(el: Element): boolean {
        const tag = el.tagName.toLowerCase();
        if (["a", "button", "input", "select", "textarea"].includes(tag)) return true;
        const role = el.getAttribute("role");
        if (role && ["button", "link", "combobox", "listbox", "option",
            "checkbox", "radio", "switch", "textbox", "searchbox",
            "slider", "spinbutton", "tab", "menuitem"].includes(role)) return true;
        if (el.getAttribute("tabindex") !== null) return true;
        return false;
      }

      function walk(el: Element, depth: number) {
        if (truncated || depth > maxDepth) return;
        if (filterMode === "interactive" && !isInteractive(el)) {
          for (const c of el.children) walk(c, depth);
          return;
        }
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;

        const ref = `ref_${++refCounter}`;
        el.setAttribute("data-ref", ref);
        const role = getRole(el);
        const label = getLabel(el);
        const x = Math.round(rect.x + rect.width / 2);
        const y = Math.round(rect.y + rect.height / 2);
        const indent = "  ".repeat(depth);
        const nameStr = label ? ` "${label}"` : "";
        const line = `${indent}- ${role}${nameStr} [ref=${ref}] (x=${x},y=${y})`;
        charCount += line.length + 1;
        if (charCount > MAX_OUTPUT) { truncated = true; return; }
        lines.push(line);
        if (filterMode !== "interactive") {
          for (const c of el.children) walk(c, depth + 1);
        }
      }

      let root: Element = document.body;
      if (focusRef) {
        const el = document.querySelector(`[data-ref="${focusRef}"]`);
        if (el) root = el;
        else return `Element with ref ${focusRef} not found. Call read_page without ref_id first.`;
      }
      walk(root, 0);
      if (truncated) {
        return `Output exceeds ${MAX_OUTPUT} characters. Try filter="interactive" to get only clickable elements, use a smaller depth, or specify ref_id to focus on a specific element.`;
      }
      return lines.join("\n");
    },
    { filterMode, maxDepth, focusRef, maxOutput },
  );
}

/**
 * ActionExecutor
 *
 * Executes CUA browser primitives and extended inspection actions on a Page object.
 * Supports: click, type, scroll, get_page_text, read_page, find, form_input.
 */
export async function executeAction(
  page: Page,
  action: ActionRequest,
  logger?: ActionLogger,
  browserContext?: any /* BrowserContext */,
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
        if (typeof x !== "number" || typeof y !== "number") {
          result = {
            success: false,
            error: "click requires x and y coordinates",
          };
          break;
        }
        await page.click(x, y, {
          button: button as "left" | "right" | "middle",
          clickCount,
        });
        result = { success: true };
        break;
      }

      case "double_click":
      case "doubleClick": {
        const { x, y } = action;
        if (typeof x !== "number" || typeof y !== "number") {
          result = {
            success: false,
            error: "double_click requires x and y coordinates",
          };
          break;
        }
        await page.click(x, y, {
          button: "left",
          clickCount: 2,
        });
        result = { success: true };
        break;
      }

      case "tripleClick": {
        const { x, y } = action;
        if (typeof x !== "number" || typeof y !== "number") {
          result = {
            success: false,
            error: "tripleClick requires x and y coordinates",
          };
          break;
        }
        await page.click(x, y, {
          button: "left",
          clickCount: 3,
        });
        result = { success: true };
        break;
      }

      case "type": {
        const { text } = action;
        if (typeof text !== "string") {
          result = { success: false, error: "type requires text parameter" };
          break;
        }
        await page.type(text);
        result = { success: true };
        break;
      }

      case "keypress": {
        const { keys } = action;
        if (!keys) {
          result = { success: false, error: "keypress requires keys parameter" };
          break;
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
        const { x = 0, y = 0, scroll_x = 0, scroll_y = 0 } = action;
        await page.scroll(
          x as number,
          y as number,
          scroll_x as number,
          scroll_y as number,
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
        // This is a no-op
        result = { success: true };
        break;
      }

      case "wait": {
        const time = action.timeMs ?? 1000;
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

      case "get_page_text": {
        const text = await page.mainFrame().evaluate(() => {
          return document.body.innerText || "";
        });
        result = { success: true, data: text };
        break;
      }

      case "read_page": {
        const filterMode = action.filter || "all";
        const maxDepth = action.depth ?? 100;
        const focusRef = action.ref_id || null;
        const MAX_OUTPUT = 50000;

        try {
          // Use CDP to get the real accessibility tree from Chromium
          const treeData = await readPageViaCDP(
            page, browserContext, filterMode, maxDepth, focusRef, MAX_OUTPUT
          );
          result = { success: true, data: treeData };
        } catch (cdpError) {
          // Fallback: DOM-walking approach if CDP is unavailable
          logger?.error(
            { error: String(cdpError) },
            "[read_page] CDP failed, falling back to DOM walk"
          );
          const treeData = await readPageFallback(
            page, filterMode, maxDepth, focusRef, MAX_OUTPUT
          );
          result = { success: true, data: treeData };
        }
        break;
      }

      case "find": {
        const { query } = action;
        if (!query) {
          result = { success: false, error: "find requires query parameter" };
          break;
        }
        const findData = await page.mainFrame().evaluate((searchQuery: string) => {
          let refCounter = 0;
          const results: string[] = [];
          // Split query into individual terms for matching
          const queryLower = searchQuery.toLowerCase();
          const queryTerms = queryLower.split(/\s+/).filter(t => t.length > 0);

          function getRole(el: Element): string {
            const ariaRole = el.getAttribute("role");
            if (ariaRole) return ariaRole;
            const tag = el.tagName.toLowerCase();
            const roleMap: Record<string, string> = {
              a: "link", button: "button", input: "input", select: "select",
              textarea: "textarea", img: "image",
            };
            return roleMap[tag] || tag;
          }

          const allElements = document.querySelectorAll("*");
          for (const el of allElements) {
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;

            const text = el.textContent?.trim() || "";
            const ariaLabel = el.getAttribute("aria-label") || "";
            const title = el.getAttribute("title") || "";
            const placeholder = el.getAttribute("placeholder") || "";
            const alt = el.getAttribute("alt") || "";
            const href = el.getAttribute("href") || "";

            const searchable = [text, ariaLabel, title, placeholder, alt, href]
              .join(" ").toLowerCase();

            // Match if all query terms appear somewhere in the searchable text
            const matches = queryTerms.every(term => searchable.includes(term));
            if (!matches) continue;

            // Skip if a child already matched (avoid duplicates for parent containers)
            const hasMatchingChild = Array.from(el.children).some(child => {
              const childSearchable = [
                child.textContent?.trim() || "",
                child.getAttribute("aria-label") || "",
                child.getAttribute("title") || "",
                child.getAttribute("placeholder") || "",
                child.getAttribute("alt") || "",
                child.getAttribute("href") || "",
              ].join(" ").toLowerCase();
              return queryTerms.every(term => childSearchable.includes(term));
            });
            if (hasMatchingChild && el.children.length > 0) continue;

            const ref = `ref_${++refCounter}`;
            el.setAttribute("data-ref", ref);

            const role = getRole(el);
            const label = ariaLabel || title || placeholder || alt ||
              (text.length > 80 ? text.substring(0, 77) + "..." : text);
            const x = Math.round(rect.x + rect.width / 2);
            const y = Math.round(rect.y + rect.height / 2);

            // Include url for links
            const urlAttr = el.getAttribute("href");
            const urlStr = urlAttr ? ` url="${urlAttr}"` : "";

            results.push(`[${ref}] ${role} "${label}" — (x=${x},y=${y})${urlStr}`);
            if (results.length >= 20) break;
          }
          return results.join("\n") ||
            `No elements found matching '${searchQuery}'. Try read_page for full tree.`;
        }, query);
        result = { success: true, data: findData };
        break;
      }

      case "form_input": {
        const { ref, value } = action;
        if (!ref) {
          result = { success: false, error: "form_input requires ref parameter" };
          break;
        }
        if (value === undefined || value === null) {
          result = { success: false, error: "form_input requires value parameter" };
          break;
        }
        const inputResult = await page.mainFrame().evaluate(
          ({ ref, value }: { ref: string; value: string }) => {
            const el = document.querySelector(`[data-ref="${ref}"]`);
            if (!el) return { ok: false, msg: `Element with ref ${ref} not found` };

            const tag = el.tagName.toLowerCase();

            if (tag === "select") {
              const select = el as HTMLSelectElement;
              // Try exact match on value, then text, then case-insensitive text
              const options = Array.from(select.options);
              const option =
                options.find(o => o.value === value) ||
                options.find(o => o.textContent?.trim() === value) ||
                options.find(o => o.textContent?.trim().toLowerCase() === value.toLowerCase());
              if (option) {
                select.value = option.value;
                select.dispatchEvent(new Event("change", { bubbles: true }));
                return { ok: true, msg: `Selected ${option.textContent?.trim()}` };
              }
              return { ok: false, msg: `Option not found: ${value}` };
            }

            if (tag === "input" || tag === "textarea") {
              const input = el as HTMLInputElement | HTMLTextAreaElement;
              const inputType = (el as HTMLInputElement).type;

              if (inputType === "checkbox" || inputType === "radio") {
                const shouldCheck = value === "true" || value === "on" || value === "1";
                (el as HTMLInputElement).checked = shouldCheck;
                el.dispatchEvent(new Event("change", { bubbles: true }));
                return { ok: true, msg: `Set ${inputType} to ${shouldCheck}` };
              }

              // For text-like inputs, focus, clear, and set value
              input.focus();
              input.value = "";
              input.value = value;
              input.dispatchEvent(new Event("input", { bubbles: true }));
              input.dispatchEvent(new Event("change", { bubbles: true }));
              return { ok: true, msg: `Set value to "${value}"` };
            }

            // For contenteditable elements
            if ((el as HTMLElement).contentEditable === "true") {
              (el as HTMLElement).innerText = value;
              el.dispatchEvent(new Event("input", { bubbles: true }));
              return { ok: true, msg: `Set content to "${value}"` };
            }

            return { ok: false, msg: `Element ${tag} is not a form input` };
          },
          { ref, value: String(value) },
        );
        result = {
          success: inputResult.ok,
          error: inputResult.ok ? undefined : inputResult.msg,
          data: inputResult.msg,
        };
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

    return {
      success: false,
      error: errorMessage,
    };
  }
}
