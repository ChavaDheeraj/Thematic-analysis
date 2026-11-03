"""The API inside a browser tab (Pyodide), for the edition that needs no server.

The page's engine worker (``frontend/src/engine/worker.ts``) calls :func:`start` once and then
:func:`handle` for every ``/api`` request instead of sending it over the network. It is the same
FastAPI app as the server edition. Three things differ, because a browser tab has no threads:

* FastAPI and Starlette run synchronous endpoints and file reads in a thread pool; here they run inline.
* Analyses and GNN experiments are queued as before, but run as jobs that hand control back to the page
  between stages, so the page stays usable and shows progress.
* The workspace lives in the browser's storage (IndexedDB). After each job step the worker saves changes
  there; it also saves after every request.
"""
from __future__ import annotations

import asyncio
import os
import traceback
from urllib.parse import unquote, urlsplit

app = None
_persist = None  # a JavaScript function from the worker that saves the workspace to IndexedDB
_later = None  # a JavaScript promise for packages loaded after start-up; jobs wait for it


def _inline_threads() -> None:
    import anyio.to_thread

    async def run_sync(func, *args, abandon_on_cancel=False, cancellable=None, limiter=None):
        return func(*args)

    anyio.to_thread.run_sync = run_sync


async def save() -> None:
    if _persist is not None:
        await _persist()


class JobQueue:
    """Stands in for the server's one-worker thread pool: jobs run one at a time, after the request that
    queued them has answered, and give the page a turn after every step."""

    def __init__(self) -> None:
        self.jobs: list = []
        self.busy = False

    def submit(self, fn, *args) -> None:
        self.jobs.append((fn, args))
        if not self.busy:
            self.busy = True
            asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        try:
            while self.jobs:
                fn, args = self.jobs.pop(0)
                await asyncio.sleep(0.01)  # the request that queued the job answers first
                if _later is not None:
                    await _later
                try:
                    steps = getattr(fn, "steps", None)
                    if steps is None:
                        fn(*args)
                    else:
                        for _ in steps(*args):
                            await save()
                            await asyncio.sleep(0.01)
                except Exception:  # jobs record their own failures; this keeps the queue running
                    traceback.print_exc()
                await save()
        finally:
            self.busy = False


def load_later(promise) -> None:
    """Packages the worker loads after start-up (statsmodels); queued jobs start once they are in."""
    global _later
    _later = promise


def start(home: str, persist=None) -> None:
    """Import the app with its workspace at ``home`` (the IndexedDB-backed folder)."""
    global app, _persist
    _persist = persist
    os.environ["THEMATICML_HOME"] = home
    os.environ["THEMATICML_EDITION"] = "browser"
    os.environ["THEMATICML_FRONTEND"] = "/nonexistent"  # the page itself comes from the website, not from Python
    _inline_threads()
    import thematicml_api.main as main

    main.executor = JobQueue()
    app = main.app


async def handle(method: str, url: str, headers, body):
    """Run one request through the app; returns ``[status, [[name, value], ...], body bytes]``."""
    headers = headers.to_py() if hasattr(headers, "to_py") else headers
    body = body.to_bytes() if hasattr(body, "to_bytes") else bytes(body or b"")
    parts = urlsplit(url)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": method.upper(),
        "scheme": "http", "path": unquote(parts.path), "raw_path": parts.path.encode(), "query_string": parts.query.encode(),
        "root_path": "", "headers": [(k.lower().encode("latin-1"), str(v).encode("latin-1")) for k, v in headers],
        "client": ("127.0.0.1", 0), "server": ("browser", 80), "state": {},
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()  # an in-process client never disconnects

    out = {"status": None, "headers": [], "body": []}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = [[k.decode("latin-1"), v.decode("latin-1")] for k, v in message.get("headers", [])]
        elif message["type"] == "http.response.body":
            out["body"].append(bytes(message.get("body", b"")))

    try:
        await app(scope, receive, send)
    except Exception:  # Starlette has already answered 500; keep the details in the console
        traceback.print_exc()
    if out["status"] is None:
        return [500, [["content-type", "application/json"]], b'{"detail": "The analysis engine failed on this request."}']
    return [out["status"], out["headers"], b"".join(out["body"])]
