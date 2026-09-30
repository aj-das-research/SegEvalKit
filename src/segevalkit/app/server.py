"""A small local HTTP server (standard library only) that serves one `ViewerSession` to the browser.

Routes
------
``/``                                   the viewer page
``/static/<file>``                      vendored assets (NiiVue, app script, stylesheet)
``/api/manifest``                       structures, models, lesions, stored results (JSON)
``/api/metrics?model=M&structure=S``    live Dice / NSD / HD95 / ASSD / RVD for one pair (JSON)
``/vol/image.nii.gz``                   the image on the viewer grid
``/vol/labels/<ref|model>.nii.gz``      label map, values = structure index
``/vol/error/<model>/<structure>.nii.gz``  error map: 1 TP, 2 FN (missed), 3 FP (added)
``/mesh/<ref|model>/<structure>.mz3``   surface mesh in world mm (gzipped MZ3)
``/mesh/<model>/<tp|fn|fp>/<structure>.mz3``  surface of one error class

The server binds to ``127.0.0.1`` by default. On a cluster, run it on the
compute node and forward the port: ``ssh -L 8765:<node>:8765 <login-host>``.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from .session import ViewerSession

__all__ = ["make_server", "serve", "page_html", "STATIC"]

STATIC = Path(__file__).resolve().parent / "static"
_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml"}


def page_html(mode: str = "server", embed: Optional[str] = None, inline_assets: bool = False) -> str:
    """The viewer page. ``embed`` is a ``<script>`` block with embedded data (static export)."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    boot = f'<script>window.SEK_MODE = "{mode}";</script>'
    if embed:
        boot += "\n" + embed
    if inline_assets:
        css = (STATIC / "app.css").read_text(encoding="utf-8")
        nv = (STATIC / "niivue.umd.js").read_text(encoding="utf-8")
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        html = html.replace('<link rel="stylesheet" href="static/app.css">', f"<style>\n{css}\n</style>")
        notice = (STATIC / "LICENSE-niivue.txt").read_text(encoding="utf-8").replace("--", "- -")
        html = html.replace('<script src="static/niivue.umd.js"></script>',
                            f"<!--\n{notice}\n-->\n<script>\n" + nv.replace("</script", "<\\/script") + "\n</script>")
        html = html.replace('<script src="static/app.js"></script>', f"<script>\n{app}\n</script>")
    return html.replace("<!--SEK_BOOT-->", boot)


def _handler(session: ViewerSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "segevalkit-view"

        def log_message(self, fmt, *args):  # quiet by default
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, allow_nan=False, default=_clean).encode(), "application/json")

        def do_GET(self):  # noqa: N802 (stdlib API)
            u = urlparse(self.path)
            path = unquote(u.path)
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, page_html().encode(), _TYPES[".html"])
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC not in f.parents or not f.is_file():
                        raise KeyError(path)
                    return self._send(200, f.read_bytes(), _TYPES.get(f.suffix, "application/octet-stream"), True)
                if path == "/api/manifest":
                    return self._json(_sanitize(session.manifest()))
                if path == "/api/metrics":
                    q = parse_qs(u.query)
                    m, s = q.get("model", [""])[0], q.get("structure", [""])[0]
                    if m not in session.models or s not in session.index:
                        raise KeyError(path)
                    return self._json(_sanitize(session.live_metrics(m, s)))
                body, ctype = session.resource(path)
                return self._send(200, body, ctype)
            except KeyError:
                return self._json({"error": f"not found: {path}"}, 404)
            except Exception as exc:  # pragma: no cover - surfaced in the page
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def _clean(o):
    try:
        return float(o)
    except Exception:  # pragma: no cover
        return str(o)


def _sanitize(o):
    """Replace NaN / inf by None so the JSON is valid."""
    import math

    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return _sanitize(o.item())
    return o


def make_server(session: ViewerSession, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create (but do not start) the server; ``port=0`` picks a free port."""
    srv = ThreadingHTTPServer((host, port), _handler(session))
    srv.daemon_threads = True
    return srv


def serve(session: ViewerSession, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          block: bool = True) -> Tuple[ThreadingHTTPServer, str]:
    """Start the viewer server and (optionally) open the browser. Returns the server and its URL."""
    srv = make_server(session, host, port)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{srv.server_address[1]}/"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    if block:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            srv.server_close()
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, url
