"""Local, content-free execution traces; provider usage is never estimated as billed usage."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import time
import uuid

_active_trace = ContextVar("rag_trace", default=None)


class RunTrace:
    def __init__(self, workflow, progress=lambda message: None):
        self.id = uuid.uuid4().hex
        self.workflow = workflow
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.started = time.perf_counter()
        self.progress = progress
        self.steps = []
        self.http_calls = []
        self.cache = {"embedding_hits": 0, "embedding_misses": 0}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                      "reported_calls": 0, "unreported_calls": 0}

    @contextmanager
    def capture(self):
        token = _active_trace.set(self)
        try:
            yield self
        finally:
            _active_trace.reset(token)

    @contextmanager
    def stage(self, name, message, **details):
        self.progress(message)
        start = time.perf_counter()
        step = {"name": name, "state": "done", **details}
        try:
            yield step
        except Exception as error:
            step.update(state="error", error_type=type(error).__name__)
            raise
        finally:
            step["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.steps.append(step)

    def finish(self, state="done", **details):
        return {"id": self.id, "workflow": self.workflow, "created_at": self.created_at,
                "state": state, "total_ms": round((time.perf_counter() - self.started) * 1000, 1),
                "steps": self.steps.copy(), "http_calls": self.http_calls.copy(),
                "embedding_cache": self.cache.copy(), "usage": self.usage.copy(), **details}


def record_http(route, start, response=None, error_type=None):
    trace = _active_trace.get()
    if trace is None:
        return
    trace.http_calls.append({"service": {"/embeddings": "embedding", "/chat/completions": "llm",
                                         "/rerank": "rerank"}.get(route, "model"),
                             "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                             "state": "error" if error_type else "done", "error_type": error_type})
    if route != "/chat/completions" or error_type:
        return
    usage = response.get("usage") if isinstance(response, dict) else None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys):
        for key in keys:
            trace.usage[key] += usage[key]
        trace.usage["reported_calls"] += 1
    else:
        trace.usage["unreported_calls"] += 1


def record_cache(hits, misses):
    trace = _active_trace.get()
    if trace is not None:
        trace.cache["embedding_hits"] += hits
        trace.cache["embedding_misses"] += misses
