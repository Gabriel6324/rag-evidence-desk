from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent


class UserError(Exception):
    """An actionable error safe to show in the browser."""


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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

    @classmethod
    def from_env(cls):
        load_env(ROOT / ".env")
        c = cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data"))).resolve(),
            embedding_mode=os.getenv("RAG_EMBEDDING_MODE", "hash"),
            answer_mode=os.getenv("RAG_ANSWER_MODE", "extractive"),
            embedding_url=os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            embedding_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            json_mode=os.getenv("LLM_JSON_MODE", "true").lower() == "true",
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_key=os.getenv("QDRANT_API_KEY", ""),
            timeout=int(os.getenv("API_TIMEOUT_SECONDS", "60")),
        )
        c.validate()
        return c

    def validate(self):
        if self.embedding_mode not in {"hash", "api"} or self.answer_mode not in {"extractive", "llm"}:
            raise UserError("模式配置无效。请检查 .env 中的 RAG_EMBEDDING_MODE 和 RAG_ANSWER_MODE。")
        if not 5 <= self.timeout <= 300:
            raise UserError("API_TIMEOUT_SECONDS 必须在 5–300 之间。")
        urls = []
        if self.embedding_mode == "api":
            if not self.embedding_model:
                raise UserError("请设置 EMBEDDING_MODEL。")
            urls.append(self.embedding_url)
        if self.answer_mode == "llm":
            if not self.llm_model:
                raise UserError("请在 .env 中填写 LLM_MODEL。")
            urls.append(self.llm_url)
        if self.qdrant_url:
            urls.append(self.qdrant_url)
        for url in urls:
            p = urlparse(url)
            if (p.scheme not in {"http", "https"} or not p.hostname or p.username
                    or p.password or p.query or p.fragment):
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
        return hashlib.sha256((self.qdrant_url or "qdrant-local-v1").encode()).hexdigest()[:20]

    def public(self):
        return {
            "embedding_mode": self.embedding_mode,
            "embedding_model": "中文 n-gram 特征哈希（非语义模型）" if self.embedding_mode == "hash" else self.embedding_model,
            "answer_mode": self.answer_mode,
            "llm_model": self.llm_model if self.answer_mode == "llm" else "原文摘录（无大模型）",
            "qdrant_mode": "远程服务" if self.qdrant_url else "本地持久化",
            "external_requests": self.embedding_mode == "api" or self.answer_mode == "llm" or bool(self.qdrant_url),
        }
