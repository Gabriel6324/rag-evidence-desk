from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import re
import threading
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import Config, ROOT, UserError
from .documents import MAX_FILE
from .engine import Engine
from .evaluation import run_evaluation, save_report


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1500)
    top_k: int = Field(default=5, ge=1, le=12)
    strategy: str = "hybrid"
    threshold: float = Field(default=.08, ge=0, le=1)
    expand: bool = True
    doc_ids: list[str] = Field(default_factory=list, max_length=60)


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=420, ge=120, le=1600)
    overlap: int = Field(default=70, ge=0, le=399)


class Jobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self.lock = threading.Lock()
        self.items = {}

    def start(self, task):
        with self.lock:
            if any(j["state"] == "running" for j in self.items.values()):
                raise UserError("已有任务正在运行，请稍等。")
            if len(self.items) > 40:
                self.items.clear()
            key = uuid.uuid4().hex
            self.items[key] = {"id": key, "state": "running", "message": "任务已开始"}
        def update(message):
            with self.lock:
                self.items[key]["message"] = message
        def work():
            try:
                result = task(update)
                with self.lock:
                    self.items[key].update(state="done", message="已完成", result=result)
            except Exception as e:
                with self.lock:
                    self.items[key].update(state="error", message=str(e) if isinstance(e, UserError) else f"任务失败（{type(e).__name__}），请检查依赖、配置或 Qdrant 连接。")
        self.executor.submit(work)
        return {"job_id": key}

    def get(self, key):
        with self.lock:
            if key not in self.items:
                raise UserError("任务不存在，可能已重启服务。")
            return dict(self.items[key])


def create_app(config=None, bootstrap=True):
    cfg = config or Config.from_env()
    jobs = Jobs()

    @asynccontextmanager
    async def lifespan(app):
        engine = Engine(cfg)
        app.state.engine = engine
        if bootstrap and not engine.meta("bootstrapped"):
            engine.seed()
            engine.set_meta("bootstrapped", True)
        if bootstrap and cfg.embedding_mode == "hash" and engine.documents() and engine.stale():
            engine.build()
        yield
        jobs.executor.shutdown(wait=True)
        engine.close()

    app = FastAPI(title="RAG 资料问答系统", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_guard(request, call_next):
        if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "仅允许从本机应用页面发起操作。"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "不接受跨站请求。"}, status_code=403)
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "请求必须包含 Content-Length。"}, status_code=411)
            try:
                too_large = int(length) < 0 or int(length) > MAX_FILE + 1024 * 1024
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse({"detail": "请求超过大小限制。"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(UserError)
    async def user_error(_, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected(_, error):
        return JSONResponse({"detail": f"操作失败（{type(error).__name__}）。请检查配置与运行环境。"}, status_code=500)

    @app.get("/api/status")
    def status():
        return app.state.engine.status()

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
        return jobs.start(lambda progress: app.state.engine.build(req.size, req.overlap, progress))

    @app.post("/api/ask")
    def ask(req: AskRequest):
        return jobs.start(lambda _: app.state.engine.ask(**req.model_dump()))

    @app.get("/api/jobs/{key}")
    def job(key: str):
        return jobs.get(key)

    @app.post("/api/evaluate")
    def evaluate():
        def run(progress):
            result = run_evaluation(progress=progress)
            names = save_report(result, cfg.data_dir / "reports", "eval_" + uuid.uuid4().hex[:10])
            summary = {k: v for k, v in result.items() if k != "runs"}
            summary["runs"] = [{k: v for k, v in r.items() if k != "details"} for r in result["runs"]]
            summary["files"] = names
            return summary
        return jobs.start(run)

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
    return app
