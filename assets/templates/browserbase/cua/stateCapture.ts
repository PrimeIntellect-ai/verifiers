import type { Page } from "@browserbasehq/stagehand";
import { BrowserState, Viewport } from "./types";

/**
 * Get the current viewport dimensions from a page
 */
export async function getViewport(page: Page): Promise<Viewport> {
  try {
    const { w, h } = await page
      .mainFrame()
      .evaluate<{
        w: number;
        h: number;
      }>("({ w: window.innerWidth, h: window.innerHeight })");
    return { width: w, height: h };
  } catch {
    // Default fallback if evaluation fails
    return { width: 1024, height: 768 };
  }
}

/**
 * Take a screenshot and return as base64 string
 */
export async function takeScreenshot(page: Page): Promise<string> {
  const buffer = await page.screenshot({ fullPage: false });
  return buffer.toString("base64");
}

/**
 * Get the current URL from a page
 */
export function getUrl(page: Page): string {
  return page.url();
}

/**
 * Capture the full browser state (screenshot, URL, viewport)
 */
export async function captureBrowserState(page: Page): Promise<BrowserState> {
  const startTime = Date.now();
  
  const [screenshot, viewport] = await Promise.all([
    takeScreenshot(page),
    getViewport(page),
  ]);

  const duration = Date.now() - startTime;
  // Base64 string is ~1.33x the size of binary, so divide by 1.33 to get approx binary size
  const sizeKB = ((screenshot.length * 0.75) / 1024).toFixed(1);
  const url = getUrl(page);
  
  console.log(`[State] Captured in ${duration}ms, screenshot: ${sizeKB}KB, url: ${url.substring(0, 60)}${url.length > 60 ? '...' : ''}`);

  return {
    screenshot,
    url,
    viewport,
  };
}
