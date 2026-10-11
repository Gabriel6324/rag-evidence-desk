from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal
import asyncio
import json
import re
import threading
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import Config, ROOT, UserError, runtime_paths
from .documents import MAX_FILE
from .engine import Engine
from .evaluation import run_evaluation, run_ablation, save_report
from .settings import SettingsInput, ProbeInput, SettingsStore, connection_test, available_models
from .mcp_server import build_mcp
from .errors import diagnose, save_diagnostic, request_stage
from .guides import load_guides


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1500)
    top_k: int = Field(default=5, ge=1, le=12)
    strategy: Literal["hybrid", "vector"] = "hybrid"
    threshold: float = Field(default=.08, ge=0, le=1)
    expand: bool = True
    doc_ids: list[str] = Field(default_factory=list, max_length=60)
    workflow: Literal["rag", "agent"] = "rag"
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    rerank: bool = False
    context_chars: int = Field(default=9000, ge=1200, le=24000)
    lexical_rescue: bool = True
    answer_guard: bool = True


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=420, ge=120, le=1600)
    overlap: int = Field(default=70, ge=0, le=399)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    live: bool = False
    ablation: bool = False


class Jobs:
    def __init__(self, config=None):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self.lock = threading.Lock()
        self.items = {}

    @contextmanager
    def exclusive(self):
        """Don't replace model configuration midway through a model/tool run."""
        with self.lock:
            if any(j["state"] == "running" for j in self.items.values()):
                raise UserError("正在处理任务，请等它完成后再保存模型设置。")
            yield

    def start(self, task, operation="任务"):
        with self.lock:
            if any(j["state"] == "running" for j in self.items.values()):
                raise UserError("已有任务正在运行，请稍等。")
            if len(self.items) > 40:
                self.items.clear()
            key = uuid.uuid4().hex
            self.items[key] = {"id": key, "state": "running", "message": operation + "：开始处理", "operation": operation}
        def update(message):
            with self.lock:
                self.items[key]["message"] = message
        def work():
            try:
                result = task(update)
                with self.lock:
                    self.items[key].update(state="done", message="已完成", result=result)
            except Exception as e:
                cfg = self.config() if callable(self.config) else self.config
                with self.lock:
                    stage = self.items[key]["message"]
                secrets = [getattr(cfg, k, "") for k in ("llm_key", "embedding_key", "rerank_key", "qdrant_key")]
                error = diagnose(e, stage=stage, operation=operation, secrets=secrets)
                if cfg:
                    error["diagnostic_file"] = save_diagnostic(error, cfg.data_dir)
                with self.lock:
                    self.items[key].update(state="error", message=error["message"], error=error)
        self.executor.submit(work)
        return {"job_id": key}

    def get(self, key):
        with self.lock:
            if key not in self.items:
                raise UserError("任务不存在，可能已重启服务。")
            return dict(self.items[key])


def create_app(config=None, bootstrap=True, restart_callback=None):
    cfg = config or Config.from_env()
    jobs = Jobs(lambda: getattr(getattr(app, "state", None), "engine", None).config if hasattr(app.state, "engine") else cfg)
    settings_path = (config.data_dir if config else runtime_paths()[0]) / ".env"
    settings_store = SettingsStore(cfg, settings_path)
    mcp = build_mcp(lambda: app.state.engine)
    mcp_app = mcp.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True)

    @asynccontextmanager
    async def lifespan(app):
        engine = None
        stage = "打开本机资料数据库与向量索引"
        try:
            engine = Engine(cfg)
            app.state.engine = engine
            if bootstrap and not engine.meta("bootstrapped"):
                stage = "首次载入演示资料"
                engine.seed()
                engine.set_meta("bootstrapped", True)
            if bootstrap and cfg.embedding_mode == "hash" and engine.documents() and engine.stale():
                stage = "首次建立或更新演示资料索引"
                engine.build()
            async with mcp.session_manager.run():
                yield
        except Exception as error:
            app.state.startup_failure = diagnose(error, stage=stage, operation="启动知据",
                secrets=[cfg.llm_key, cfg.embedding_key, cfg.rerank_key, cfg.qdrant_key])
            raise
        finally:
            jobs.executor.shutdown(wait=True)
            if engine is not None:
                engine.close()

    app = FastAPI(title="RAG 资料问答系统", version=__version__, lifespan=lifespan)
    app.state.settings_store = settings_store
    app.state.instance_id = uuid.uuid4().hex
    app.state.mcp_server = mcp
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_guard(request, call_next):
        if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return failure_response(request, UserError("仅允许从本机应用页面发起操作。"), 403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return failure_response(request, UserError("不接受跨站请求。"), 403)
            length = request.headers.get("content-length")
            if length is None:
                return failure_response(request, UserError("请求必须包含 Content-Length。"), 411)
            try:
                content_size = int(length)
            except ValueError:
                return failure_response(request, UserError("Content-Length 必须是非负整数。"), 400)
            if content_size < 0:
                return failure_response(request, UserError("Content-Length 必须是非负整数。"), 400)
            if content_size > MAX_FILE + 1024 * 1024:
                return failure_response(request, UserError("请求大小超过限制。"), 413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(UserError)
    async def user_error(request, error):
        return failure_response(request, error, 400)

    def failure_response(request, error, status_code):
        current = app.state.engine.config if hasattr(app.state, "engine") else cfg
        secrets = [getattr(current, k) for k in ("llm_key", "embedding_key", "rerank_key", "qdrant_key")]
        stage = request_stage(request.url.path)
        diagnostic = diagnose(error, stage=stage, operation=stage, secrets=secrets)
        diagnostic["diagnostic_file"] = save_diagnostic(diagnostic, current.data_dir)
        return JSONResponse({"detail": diagnostic["message"], "error": diagnostic}, status_code=status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        # Pydantic's raw input can include an API key. Return only field labels.
        fields = [".".join(str(p) for p in item["loc"][1:]) for item in error.errors()]
        return failure_response(request, UserError("请检查填写内容：" + "、".join(fields[:8])), 422)

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        return failure_response(request, error, 500)

    @app.get("/api/guides")
    def guides():
        return load_guides()

    @app.get("/api/status")
    def status():
        return app.state.engine.status()

    @app.get("/api/health")
    def health():
        return {"status": "ready", "version": __version__, "instance_id": app.state.instance_id,
                "needs_rebuild": app.state.engine.status()["needs_rebuild"]}

    @app.get("/api/integrations")
    def integrations(request: Request):
        return {"mcp": {"url": str(request.base_url).rstrip("/") + "/mcp/", "transport": "Streamable HTTP",
                        "tools": ["list_documents", "search_documents", "read_source"], "read_only": True}}

    @app.get("/api/settings")
    def settings():
        with settings_store.lock:
            return settings_store.public()

    @app.post("/api/settings")
    def save_settings(req: SettingsInput, background: BackgroundTasks):
        engine = app.state.engine
        with jobs.exclusive(), engine.lock:
            new_config = settings_store.save(req)
            engine.config = engine.embedder.config = new_config
            if restart_callback:
                background.add_task(restart_callback)
            return {"ok": True, "settings": settings_store.public(), "needs_rebuild": engine.stale(),
                    "restart_requested": restart_callback is not None, "previous_instance": app.state.instance_id,
                    "message": ("设置已保存，正在自动重启服务。" if restart_callback else "设置已保存并立即生效。") +
                               ("向量配置已变化，请重建索引。" if engine.stale() else "资料与会话会保留。")}

    @app.post("/api/settings/test")
    def test_settings(req: ProbeInput):
        with settings_store.lock:
            candidate = settings_store.candidate(req, validate=False)
        return connection_test(candidate, req.kind)

    @app.post("/api/settings/models")
    def models_list(req: ProbeInput):
        with settings_store.lock:
            candidate = settings_store.candidate(req, validate=False)
        return available_models(candidate, req.kind)

    @app.get("/api/sessions")
    def sessions():
        with app.state.engine.lock:
            return app.state.engine.sessions.list()

    @app.post("/api/sessions")
    def new_session():
        with app.state.engine.lock:
            return app.state.engine.sessions.create()

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        with app.state.engine.lock:
            return app.state.engine.sessions.detail(session_id)

    @app.delete("/api/sessions/{session_id}")
    def remove_session(session_id: str):
        with app.state.engine.lock:
            app.state.engine.sessions.remove(session_id)
        return {"ok": True}

    @app.get("/api/traces")
    def traces():
        with app.state.engine.lock:
            return app.state.engine.sessions.traces()

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str):
        with app.state.engine.lock:
            return app.state.engine.sessions.trace(trace_id)

    @app.post("/api/documents")
    def upload(file: UploadFile = File(...)):
        content = file.file.read(MAX_FILE + 1)
        return app.state.engine.ingest(file.filename or "", content)

    @app.get("/api/documents/{doc_id}")
    def source(doc_id: str):
        return app.state.engine.source(doc_id)

    @app.delete("/api/documents/{doc_id}")
    def delete(doc_id: str):
        app.state.engine.remove(doc_id)
        return {"ok": True}

    @app.post("/api/samples")
    def seed():
        app.state.engine.seed()
        return {"ok": True}

    @app.post("/api/build")
    def build(req: BuildRequest):
        if req.overlap >= req.size:
            raise UserError("overlap 必须小于 chunk 长度。")
        return jobs.start(lambda progress: app.state.engine.build(req.size, req.overlap, progress), "重建资料索引")

    @app.post("/api/ask")
    def ask(req: AskRequest):
        return jobs.start(lambda progress: app.state.engine.ask(progress=progress, **req.model_dump()), "资料问答")

    @app.get("/api/jobs/{key}")
    def job(key: str):
        return jobs.get(key)

    @app.get("/api/jobs/{key}/events")
    def events(key: str, request: Request):
        jobs.get(key)  # Validate before starting the HTTP stream.
        async def stream():
            previous = None
            while not await request.is_disconnected():
                item = jobs.get(key)
                snapshot = (item["state"], item["message"])
                if snapshot != previous:
                    previous = snapshot
                    if item["state"] == "done":
                        yield "event: done\ndata: " + json.dumps(item["result"], ensure_ascii=False) + "\n\n"
                        return
                    if item["state"] == "error":
                        yield "event: failed\ndata: " + json.dumps({"message": item["message"], "error": item["error"]}, ensure_ascii=False) + "\n\n"
                        return
                    yield "event: progress\ndata: " + json.dumps({"message": item["message"]}, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(.15)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/evaluate")
    def evaluate(req: EvaluationRequest):
        if req.live and req.ablation:
            raise UserError("优化前后对比使用离线资料；真实模型评测请单独运行。")
        current_config = app.state.engine.config
        if req.live and current_config.answer_mode != "llm":
            raise UserError("请先连接对话模型，再运行真实模型评测。")
        def run(progress):
            options = {"config": current_config, "live": req.live, "progress": progress}
            if req.live:
                # A bounded 36-question comparison rather than 216 paid calls.
                options.update(sizes=(420,), ks=(5,))
            if req.ablation:
                result = run_ablation(config=current_config, progress=progress)
            else:
                result = run_evaluation(**options)
            progress("保存评测报告：JSON、CSV 与 Markdown")
            names = save_report(result, current_config.data_dir / "reports", "eval_" + uuid.uuid4().hex[:10])
            summary = {k: v for k, v in result.items() if k != "runs"}
            summary["runs"] = [{k: v for k, v in r.items() if k != "details"} for r in result["runs"]]
            summary["files"] = names
            return summary
        return jobs.start(run, "真实模型评测" if req.live else "优化前后对比" if req.ablation else "离线参数对照评测")

    @app.get("/api/reports/{name}")
    def report(name: str):
        if not re.fullmatch(r"eval_[a-f0-9]{10}\.(json|csv|md)", name):
            raise UserError("无效的报告名。")
        path = cfg.data_dir / "reports" / name
        if not path.is_file():
            raise UserError("报告不存在，请重新运行评测。")
        return FileResponse(path, filename=name)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    app.mount("/mcp", mcp_app)
    return app
