#!/usr/bin/env python3
"""仅用标准库的本地界面服务：静态页面 + /api/generate + /api/upload。

运行: python3 server.py [--port 8765]
"""
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".pydeps"))
sys.path.insert(0, str(ROOT))

from quiltstudio.api import generate, UPLOAD_DIR  # noqa: E402

_gen_lock = threading.Lock()
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("[server] " + fmt % args + "\n")

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", "/web/index.html")
            self.end_headers()
            return
        if parsed.path == "/api/manifest":
            mp = ROOT / "output" / "manifest.json"
            if not mp.exists():
                self._json({"runs": []})
            else:
                self._json(json.loads(mp.read_text(encoding="utf-8")))
            return
        if parsed.path == "/api/run":
            qs = parse_qs(parsed.query)
            rid = Path(qs.get("id", [""])[0]).name
            rp = ROOT / "output" / "runs" / rid / "run.json"
            if not rp.exists():
                self._json({"error": f"run not found: {rid}"}, 404)
            else:
                self._json(json.loads(rp.read_text(encoding="utf-8")))
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            if parsed.path == "/api/generate":
                params = json.loads(raw.decode("utf-8") or "{}")
                with _gen_lock:
                    rj = generate(params)
                self._json({"ok": True, "run_id": rj["run_id"],
                            "status": rj["status"]})
                return
            if parsed.path == "/api/upload":
                qs = parse_qs(parsed.query)
                name = Path(qs.get("name", ["upload.png"])[0]).name
                ext = Path(name).suffix.lower()
                if ext not in ALLOWED_EXT:
                    self._json({"error": f"不支持的扩展名 {ext}"}, 400)
                    return
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                stem = Path(name).stem[:40] or "upload"
                dst = UPLOAD_DIR / name
                i = 1
                while dst.exists():
                    dst = UPLOAD_DIR / f"{stem}_{i}{ext}"
                    i += 1
                dst.write_bytes(raw)
                try:
                    from PIL import Image
                    with Image.open(dst) as im:
                        im.verify()
                except Exception as e:
                    dst.unlink(missing_ok=True)
                    self._json({"error": f"图片无法解析: {e}"}, 400)
                    return
                self._json({"ok": True, "name": dst.name,
                            "url": f"/output/uploads/{dst.name}"})
                return
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"纹理扩展工作室: http://{host}:{port}/web/index.html")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.host, a.port)
