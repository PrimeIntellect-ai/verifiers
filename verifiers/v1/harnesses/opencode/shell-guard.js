import { Effect } from "effect";

const NUL_ERROR =
  "printf '%s\\n' 'error: the command contains a NUL byte, which a process argument " +
  "cannot carry; produce the byte inside the command instead, e.g. with printf' >&2; exit 2";

export default {
  id: "verifiers.shell-guard",
  effect: Effect.fn(function* (ctx) {
    const { maxOutputBytes, node, cap } = ctx.options;
    yield* ctx.shell.hook("create.before", (shell) =>
      Effect.sync(() => {
        // OpenCode 2.1.4 never settles a shell whose argv holds a NUL: the spawn fails
        // outside its handlers, and the session waits on the tool forever.
        if (shell.command.includes("\0")) {
          shell.command = NUL_ERROR;
          return;
        }
        // OpenCode writes a command's whole output to a file under its data directory,
        // so a command that never stops printing (a REPL spinning on stdin's EOF) fills
        // the box's disk before its timeout; OpenCode's database then fails a write and
        // the session dies without ending its turn. Its output goes through `cap`,
        // which passes the first `maxOutputBytes` and drops the rest. Process
        // substitution is bash's and zsh's; another shell runs the command as it was.
        if (maxOutputBytes && /(^|\/)(bash|zsh)$/.test(shell.shell)) {
          shell.command = `exec > >('${node}' '${cap}' ${maxOutputBytes}) 2>&1\n${shell.command}`;
        }
      }),
    );
  }),
};
