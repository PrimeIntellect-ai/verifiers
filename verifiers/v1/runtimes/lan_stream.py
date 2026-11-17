"""Shared sandbox-side client for a process-scoped, persistent LAN stream."""

PYTHON = """# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets==15.0.1"]
# ///
"""

# Both the evaluator relay and native harness bridge embed this client. A single
# reader demultiplexes replies while writes, reads and lifecycle calls overlap.
CLIENT_SOURCE = r"""
import concurrent.futures
import json
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

class LanStream:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.lock = threading.RLock()
        self.connection = None
        self.pending = {}
        self.closed = False

    def disconnect(self, connection):
        with self.lock:
            if self.connection is connection:
                self.connection = None
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(OSError("LAN execution stream disconnected"))
        if connection is not None:
            connection.close()

    def receive(self, connection):
        try:
            for raw in connection:
                response = json.loads(raw)
                with self.lock:
                    # A response from a replaced socket cannot complete a retry.
                    if self.connection is not connection:
                        return
                    future = self.pending.get(response["id"])
                    if future is not None and not future.done():
                        if "error" in response:
                            future.set_exception(RuntimeError(response["error"]))
                        else:
                            future.set_result(response["result"])
        except (ConnectionClosed, OSError):
            pass
        finally:
            self.disconnect(connection)

    def request(self, body):
        request_id = body["request_id"]
        payload = json.dumps(body)
        for attempt in range(3):
            connection = None
            future = concurrent.futures.Future()
            try:
                with self.lock:
                    if self.closed:
                        raise RuntimeError("LAN execution stream is closed")
                    if request_id in self.pending:
                        raise RuntimeError("LAN request is already pending")
                    if self.connection is None:
                        self.connection = connect(
                            self.endpoint["url"],
                            additional_headers={"Authorization": "Bearer " + self.endpoint["secret"]},
                            proxy=None, compression=None, open_timeout=10,
                            close_timeout=1, max_size=32 * 1024 * 1024,
                        )
                        threading.Thread(target=self.receive, args=(self.connection,), daemon=True).start()
                    connection = self.connection
                    self.pending[request_id] = future
                    connection.send(payload)
                return future.result(timeout=3600)
            except (ConnectionClosed, OSError, concurrent.futures.TimeoutError):
                self.disconnect(connection)
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
            finally:
                with self.lock:
                    if self.pending.get(request_id) is future:
                        del self.pending[request_id]
        # Retries reuse the exact request ID and body. The server replays the
        # receipt, including consumed output, instead of repeating the operation.

    def close(self):
        with self.lock:
            self.closed = True
            connection = self.connection
        self.disconnect(connection)
"""
