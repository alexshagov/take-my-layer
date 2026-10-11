"""Contract test stub (HARNESS §8.1 / Q2): serves ONLY fabricated responses.

Isolates the @ai-sdk parsing question from all MLX/memory issues. Variant
selected by env STUB_VARIANT: v1 = the fabricated tool_calls shape the
pre-fork proxy emitted (that leg was removed at fork; this stub is the kept
contract record — docs/POLICY.md §4).
"""
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VARIANT = os.environ.get("STUB_VARIANT", "v1")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        b = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        stream = bool(payload.get("stream"))
        model = payload.get("model", "stub")
        cid = f"chatcmpl-{uuid.uuid4()}"
        tc = {"function": {"name": "read",
                           "arguments": json.dumps({"filePath": "README.md"})},
              "type": "function", "id": f"call-{uuid.uuid4()}"}
        if not stream:
            out = {"id": cid, "system_fingerprint": "decide-proxy",
                   "object": "chat.completion", "model": model,
                   "created": int(time.time()),
                   "choices": [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"role": "assistant", "content": "\n\n",
                                            "tool_calls": [tc]}}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return

        def chunk(delta, finish=None):
            return {"id": cid, "system_fingerprint": "decide-proxy",
                    "object": "chat.completion.chunk", "model": model,
                    "created": int(time.time()),
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "delta": delta}]}

        parts = []
        if VARIANT == "v2":                      # real-server-like preamble
            parts.append(chunk({"role": "assistant"}))
            parts.append(chunk({"reasoning": "thinking about the file"}))
        parts.append(chunk({"role": "assistant", "tool_calls": [tc]}))
        parts.append(chunk({"role": "assistant"}, finish="tool_calls"))
        sse = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
        b = sse.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Contract-test stub: serves ONLY fabricated chat-completion "
                    "tool_calls (variant via env STUB_VARIANT=v1|v2). Binds "
                    "127.0.0.1 only.")
    ap.add_argument("port", nargs="?", type=int, default=8091,
                    help="TCP port (default 8091)")
    port = ap.parse_args().port
    print(f"stub ({VARIANT}) on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
