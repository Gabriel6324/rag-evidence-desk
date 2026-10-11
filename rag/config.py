from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent


def runtime_paths():
    """Frozen Windows builds keep settings/data outside the bundled, read-only assets."""
    if getattr(sys, "frozen", False):
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "ZhiJuEvidenceDesk"
    else:
        base = ROOT
    settings = Path(os.getenv("RAG_CONFIG_DIR", str(base))).resolve()
    return settings, Path(os.getenv("RAG_DATA_DIR", str(base / "data"))).resolve()


class UserError(Exception):
    """An actionable error safe to show in the browser."""


def read_env(path: Path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if value.startswith('"'):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = value.strip('"')
            else:
                value = value.strip("'")
            values[key.strip()] = value
    return values


def load_env(path: Path):
    for key, value in read_env(path).items():
        os.environ.setdefault(key, value)


@dataclass
class Config:
    data_dir: Path
    embedding_mode: str = "hash"
    answer_mode: str = "extractive"
    embedding_url: str = "https://api.openai.com/v1"
    embedding_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_url: str = "https://api.openai.com/v1"
    llm_key: str = ""
    llm_model: str = ""
    json_mode: bool = True
    qdrant_url: str = ""
    qdrant_key: str = ""
    timeout: int = 60
    rerank_mode: str = "none"
    rerank_url: str = "https://api.siliconflow.cn/v1"
    rerank_key: str = ""
    rerank_model: str = ""
    qdrant_memory: bool = False  # Isolated evaluation only; never exposed as a user setting.

    @classmethod
    def from_env(cls, validate=True):
        settings, default_data = runtime_paths()
        # Reading configuration must not permanently inject old file values into
        # the process environment: the visual settings editor supports hot reload.
        values = {**read_env(settings / ".env"), **os.environ}
        get = values.get
        try:
            timeout = int(get("API_TIMEOUT_SECONDS", "60"))
        except ValueError:
            if validate:
                raise UserError("API_TIMEOUT_SECONDS 必须填写 5–300 之间的整数。") from None
            timeout = 60
        json_mode = get("LLM_JSON_MODE", "true").strip().lower()
        if json_mode not in {"true", "false"}:
            if validate:
                raise UserError("LLM_JSON_MODE 必须填写 true 或 false。")
            json_mode = "true"
        c = cls(
            data_dir=Path(get("RAG_DATA_DIR", str(default_data))).resolve(),
            embedding_mode=get("RAG_EMBEDDING_MODE", "hash"),
            answer_mode=get("RAG_ANSWER_MODE", "extractive"),
            embedding_url=get("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            embedding_key=get("EMBEDDING_API_KEY", ""),
            embedding_model=get("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_url=get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_key=get("LLM_API_KEY", ""),
            llm_model=get("LLM_MODEL", ""),
            json_mode=json_mode == "true",
            qdrant_url=get("QDRANT_URL", ""),
            qdrant_key=get("QDRANT_API_KEY", ""),
            timeout=timeout,
            rerank_mode=get("RAG_RERANK_MODE", "none"),
            rerank_url=get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            rerank_key=get("RERANK_API_KEY", ""),
            rerank_model=get("RERANK_MODEL", ""),
        )
        if validate:
            c.validate()
        return c

    def validate(self):
        if self.embedding_mode not in {"hash", "api"} or self.answer_mode not in {"extractive", "llm"}:
            raise UserError("运行模式无效，请在模型设置中重新选择模式。")
        if not 5 <= self.timeout <= 300:
            raise UserError("API_TIMEOUT_SECONDS 必须在 5–300 之间。")
        if self.rerank_mode not in {"none", "api"}:
            raise UserError("RAG_RERANK_MODE 必须填写 none 或 api。")
        urls = []
        if self.embedding_mode == "api":
            if not self.embedding_model:
                raise UserError("请设置 EMBEDDING_MODEL。")
            urls.append(self.embedding_url)
        if self.answer_mode == "llm":
            if not self.llm_model:
                raise UserError("请在模型设置中填写对话模型名称。")
            urls.append(self.llm_url)
        if self.qdrant_url:
            urls.append(self.qdrant_url)
        if self.rerank_mode == "api":
            if not self.rerank_model:
                raise UserError("请设置 RERANK_MODEL。")
            urls.append(self.rerank_url)
        for url in urls:
            try:
                p = urlparse(url)
                port = p.port
            except ValueError:
                raise UserError("API 地址或端口无效，请检查 BASE_URL / QDRANT_URL。") from None
            if (p.scheme not in {"http", "https"} or not p.hostname or p.username
                    or p.password or p.query or p.fragment or port == 0):
                raise UserError("API 地址必须为不含账号、查询参数的 HTTP(S) 地址。")
            if p.scheme == "http" and p.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise UserError("远程模型 / Qdrant 请使用 HTTPS；HTTP 仅允许本机服务。")

    @property
    def embedding_fingerprint(self):
        name = "hash-ngram-v1:2048" if self.embedding_mode == "hash" else f"api:{self.embedding_url}:{self.embedding_model}"
        return hashlib.sha256(name.encode()).hexdigest()[:20]

    @property
    def storage_fingerprint(self):
        # Changing vector DB targets must not leave a seemingly ready local manifest.
        return hashlib.sha256((self.qdrant_url or ("qdrant-memory" if self.qdrant_memory else "qdrant-local-v1")).encode()).hexdigest()[:20]

    def public(self):
        return {
            "embedding_mode": self.embedding_mode,
            "embedding_model": "中文 n-gram 特征哈希（非语义模型）" if self.embedding_mode == "hash" else self.embedding_model,
            "answer_mode": self.answer_mode,
            "llm_model": self.llm_model if self.answer_mode == "llm" else "原文摘录（无大模型）",
            "qdrant_mode": "远程服务" if self.qdrant_url else "评测内存隔离" if self.qdrant_memory else "本地持久化",
            "rerank_mode": self.rerank_mode,
            "rerank_model": self.rerank_model if self.rerank_mode == "api" else "未启用",
            "agent_mode": "模型工具调用" if self.answer_mode == "llm" else "规则流程演示（无模型决策）",
            "external_requests": self.embedding_mode == "api" or self.answer_mode == "llm" or bool(self.qdrant_url) or self.rerank_mode == "api",
        }
