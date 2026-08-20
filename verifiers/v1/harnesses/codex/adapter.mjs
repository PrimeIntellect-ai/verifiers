/** Run codex-acp with a full-network workspace sandbox for model tools. */

import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const adapterPath = process.argv[2];
const patchedPath = process.argv[3];
let source = await readFile(adapterPath, "utf8");
const original = '    { "type": "dangerFullAccess" },\n    "danger-full-access"';
const replacement = `    {
      type: "workspaceWrite",
      writableRoots: [],
      networkAccess: true,
      excludeTmpdirEnvVar: false,
      excludeSlashTmp: false
    },
    "workspace-write"`;

if (
  source.indexOf(original) < 0 ||
  source.indexOf(original) !== source.lastIndexOf(original)
) {
  throw new Error("Unsupported codex-acp agent mode definition");
}
source = source.replace(/^#![^\n]*\n/, "").replace(original, replacement);
await writeFile(patchedPath, source);
await import(pathToFileURL(patchedPath));
