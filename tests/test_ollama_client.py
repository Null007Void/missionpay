import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from missionpay.ollama import OllamaJsonModel


class _Handler(BaseHTTPRequestHandler):
    received: dict[str, object] | None = None

    def do_POST(self):  # noqa: N802 - stdlib callback name
        length = int(self.headers["Content-Length"])
        type(self).received = json.loads(self.rfile.read(length))
        body = json.dumps(
            {"message": {"role": "assistant", "content": '{"facts":[],"missing":[]}'}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def test_ollama_client_requests_non_streaming_deterministic_json():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = OllamaJsonModel(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_id="qwen3:4b",
            timeout_seconds=5,
        )
        content = model.complete_json("system", "receipt")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert content == '{"facts":[],"missing":[]}'
    assert _Handler.received == {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "receipt"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": 7},
    }
