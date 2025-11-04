from __future__ import annotations

import gzip
import http.server
import io
import json
import socketserver
from pathlib import Path
from urllib.parse import parse_qs, urlparse

socketserver.TCPServer.allow_reuse_address = True

ROOT = Path(__file__).resolve().parent.parent / "fixtures"
CATALOG = {
    "fruits": ["apple", "banana", "cherry", "date"],
    "colors": ["red", "green", "blue", "violet"],
    "shapes": ["circle", "square", "triangle"],
}
AUTH_TOKEN = "opensesame"

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "MiniHTTPD/0.1"

    def _set_headers(self, status=200, headers=None):
        self.send_response(status)
        headers = headers or {}
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

    def do_HEAD(self):
        return self.do_GET(head_only=True)

    def do_GET(self, head_only=False):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query or "")

        if path == "/notfound":
            self._set_headers(404, {"Content-Type": "text/plain"})
            if not head_only:
                self.wfile.write(b"not found\n")
            return

        if path == "/headers":
            # Echo back a chosen header (for deterministic tests)
            value = self.headers.get("X-Debug", "none")
            body = json.dumps({"echo": value}).encode("utf-8")
            self._set_headers(200, {"Content-Type": "application/json"})
            if not head_only:
                self.wfile.write(body)
            return

        if path == "/redirect1":
            self.send_response(302)
            self.send_header("Location", "/html/final.html")
            self.end_headers()
            return

        if path == "/redirect_chain":
            self.send_response(302)
            self.send_header("Location", "/redirect_step2")
            self.end_headers()
            return

        if path == "/redirect_step2":
            self.send_response(302)
            self.send_header("Location", "/html/final.html")
            self.end_headers()
            return

        if path == "/compressed":
            payload = json.dumps({"ok": True, "note": "gzipped"}).encode("utf-8")
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(payload)
            compressed = buf.getvalue()
            self._set_headers(200, {"Content-Type": "application/json", "Content-Encoding": "gzip"})
            if not head_only:
                self.wfile.write(compressed)
            return

        if path == "/json/data_large.jsonl":
            # Generate deterministic 100 lines
            lines = [json.dumps({"i": i, "v": i * 2}) for i in range(100)]
            data = ("\n".join(lines) + "\n").encode("utf-8")
            self._set_headers(200, {"Content-Type": "application/json"})
            if not head_only:
                self.wfile.write(data)
            return

        if path == "/query":
            category = (qs.get("category") or [""])[0]
            items = CATALOG.get(category, [])
            limit_str = (qs.get("limit") or [str(len(items))])[0]
            try:
                limit = max(0, min(len(items), int(limit_str)))
            except ValueError:
                limit = len(items)
            limited = items[:limit]
            body = json.dumps({
                "category": category,
                "limit": limit,
                "results": limited,
                "count": len(limited),
                "summary": ", ".join(limited) if limited else "none",
                "first": limited[0] if limited else None,
            }).encode("utf-8")
            self._set_headers(200, {"Content-Type": "application/json"})
            if not head_only:
                self.wfile.write(body)
            return

        if path == "/auth":
            token = self.headers.get("X-Token", "")
            if token == AUTH_TOKEN:
                body = json.dumps({"secret": "orchid", "message": "access granted"}).encode("utf-8")
                status = 200
            else:
                body = json.dumps({"error": "missing or invalid token"}).encode("utf-8")
                status = 401
            self._set_headers(status, {"Content-Type": "application/json"})
            if not head_only:
                self.wfile.write(body)
            return

        # Static files under fixtures/
        local = ROOT / path.lstrip("/")
        if local.is_dir():
            local = local / "index.html"
        if local.exists() and local.is_file():
            ctype = "text/html" if local.suffix.lower() in {".html", ".htm"} else "application/json"
            if local.suffix.lower() == ".txt":
                ctype = "text/plain"
            if local.suffix.lower() == ".jsonl":
                ctype = "application/json"
            data = b"" if head_only else local.read_bytes()
            self._set_headers(200, {"Content-Type": ctype, "Content-Length": str((local.stat().st_size))})
            if not head_only:
                self.wfile.write(data)
            return

        # Fallback
        self._set_headers(404, {"Content-Type": "text/plain"})
        if not head_only:
            self.wfile.write(b"not found\n")

def serve(port=31415):
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        sa = httpd.socket.getsockname()
        print(f"Serving fixtures at http://{sa[0]}:{sa[1]} (root={ROOT})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31415)
    args = p.parse_args()
    serve(args.port)
