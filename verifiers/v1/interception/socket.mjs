import { createConnection } from "node:net";
import { createInterface } from "node:readline";

// OpenClaw can load the plugin more than once in one process. All instances
// share the connection, which the runner removes from the filesystem on accept.
const state = (globalThis[Symbol.for("verifiers.toolInterception")] ??= {
  path: process.env.VF_TOOL_INTERCEPTION_SOCKET,
  connecting: null,
  failure: null,
  pending: new Map(),
  nextId: 0,
});
delete process.env.VF_TOOL_INTERCEPTION_SOCKET;

function failConnection(error) {
  state.failure ??= error;
  for (const request of state.pending.values()) request.reject(state.failure);
  state.pending.clear();
}

async function requestToolPolicy(body) {
  if (state.failure) throw state.failure;
  if (!state.connecting) {
    state.connecting = new Promise((resolve, reject) => {
      if (!state.path) throw new Error("Tool interception socket is missing");
      const socket = createConnection(state.path);
      socket.once("connect", () => {
        socket.unref();
        resolve(socket);
      });
      socket.on("error", (error) => {
        failConnection(error);
        reject(error);
      });
      socket.on("close", () => {
        const error = new Error("LiveACPClient closed the interception socket");
        failConnection(error);
        reject(error);
      });
      createInterface({ input: socket }).on("line", (line) => {
        try {
          const response = JSON.parse(line);
          const request = state.pending.get(response.id);
          if (!request) throw new Error("Unknown tool policy response ID");
          if (response.error) request.reject(new Error(response.error));
          else request.resolve(validateToolDecision(response.decision, "LiveACPClient"));
          state.pending.delete(response.id);
        } catch (error) {
          failConnection(error);
          socket.destroy();
        }
      });
    });
  }
  const socket = await state.connecting;
  if (state.failure) throw state.failure;
  const id = ++state.nextId;
  const response = new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
  });
  const timeout = setTimeout(() => {
    failConnection(new Error("LiveACPClient timed out"));
    socket.destroy();
  }, 30_000);
  socket.write(`${JSON.stringify({ id, body })}\n`);
  try {
    return await response;
  } finally {
    clearTimeout(timeout);
    state.pending.delete(id);
  }
}
