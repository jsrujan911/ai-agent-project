# -*- coding: utf-8 -*-
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).parent
STATIC_DIR = ROOT_DIR / "static"
HOST = "127.0.0.1"
PORT = 8000
rag_app = None


def get_env_value(name, default=""):
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return os.getenv(name, default)

    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip()

    return os.getenv(name, default)


def get_rag_app():
    global rag_app
    if rag_app is None:
        import app

        rag_app = app
    return rag_app


class RagWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/":
            self.path = "/index.html"
            return super().do_GET()

        if parsed_path.path == "/api/status":
            return self.send_json(
                {
                    "model": get_env_value("LLM_MODEL", "mistral"),
                    "embedding_model": "nomic-embed-text",
                    "retriever_k": int(get_env_value("RETRIEVER_K", "3")),
                    "fetch_k": int(get_env_value("RETRIEVER_FETCH_K", "12")),
                    "similarity_threshold": float(get_env_value("SIMILARITY_THRESHOLD", "0.18")),
                    "arxiv_query": get_env_value("ARXIV_QUERY", ""),
                    "data_dir": "data",
                }
            )

        return super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path != "/api/ask":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            history = payload.get("history", [])

            if not question:
                self.send_json({"error": "Question is required."}, status=400)
                return

            result = get_rag_app().answer_question(question, history=history)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, payload, status=200):
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main():
    server = ThreadingHTTPServer((HOST, PORT), RagWebHandler)
    print(f"RAG website is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
