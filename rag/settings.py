"""Visual configuration, atomic writeback, and explicit low-cost connection probes.

Keys are write-only over HTTP. Tests send fixed synthetic text, never user files.
Settings changes never alter document paths or the Qdrant storage target.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal
from urllib.parse import urlparse

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import Config, UserError
from .models import post_json


class ConnectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    url: str = Field(default="", max_length=1500)
    model: str = Field(default="", max_length=150)
    key: str | None = Field(default=None, max_length=4096)
    clear_key: bool = False


class SettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm: ConnectionInput
    embedding: ConnectionInput
    rerank: ConnectionInput
    timeout: int = Field(default=60, ge=5, le=300)
    json_mode: bool = True


class ProbeInput(SettingsInput):
    kind: Literal["llm", "embedding", "rerank"] = "llm"


ENV_FIELDS = {
    "RAG_ANSWER_MODE": "answer_mode", "LLM_BASE_URL": "llm_url", "LLM_API_KEY": "llm_key",
    "LLM_MODEL": "llm_model", "RAG_EMBEDDING_MODE": "embedding_mode",
    "EMBEDDING_BASE_URL": "embedding_url", "EMBEDDING_API_KEY": "embedding_key",
    "EMBEDDING_MODEL": "embedding_model", "RAG_RERANK_MODE": "rerank_mode",
    "RERANK_BASE_URL": "rerank_url", "RERANK_API_KEY": "rerank_key", "RERANK_MODEL": "rerank_model",
    "API_TIMEOUT_SECONDS": "timeout", "LLM_JSON_MODE": "json_mode",
}


def single_line(value):
    if any(c in value for c in "\r\n\x00"):
        raise UserError("地址、模型名和密钥必须填写在一行内，请检查复制的内容。")
    return value.strip()


class SettingsStore:
    def __init__(self, config: Config, path: Path):
        self.config, self.path = config, Path(path)
        self.lock = threading.RLock()

    def public(self):
        result = {"timeout": self.config.timeout, "json_mode": self.config.json_mode}
        for kind in ("llm", "embedding", "rerank"):
            mode = self.config.answer_mode if kind == "llm" else getattr(self.config, kind + "_mode")
            result[kind] = {"enabled": mode in {"llm", "api"},
                            "url": getattr(self.config, kind + "_url"),
                            "model": getattr(self.config, kind + "_model"),
                            "has_key": bool(getattr(self.config, kind + "_key"))}
        return result

    def candidate(self, req: SettingsInput, validate=True):
        values = {"timeout": req.timeout, "json_mode": req.json_mode}
        for kind in ("llm", "embedding", "rerank"):
            connection = getattr(req, kind)
            values[kind + "_url"] = single_line(connection.url).rstrip("/")
            values[kind + "_model"] = single_line(connection.model)
            previous_key = getattr(self.config, kind + "_key")
            if previous_key and not connection.key and not connection.clear_key:
                try:
                    previous_url = urlparse(getattr(self.config, kind + "_url"))
                    next_url = urlparse(values[kind + "_url"])
                except ValueError:
                    raise UserError("服务地址格式有误，请修正地址并重新填写 Key，或清除原密钥。") from None
                if (previous_url.scheme, previous_url.netloc) != (next_url.scheme, next_url.netloc):
                    raise UserError("服务地址已换到其他服务，请重新填写该服务的 Key，或勾选清除原密钥。")
            # Empty input preserves a previously saved key. Clearing is explicit.
            values[kind + "_key"] = ("" if connection.clear_key else single_line(connection.key)
                                      if connection.key else getattr(self.config, kind + "_key"))
        values.update(answer_mode="llm" if req.llm.enabled else "extractive",
                      embedding_mode="api" if req.embedding.enabled else "hash",
                      rerank_mode="api" if req.rerank.enabled else "none")
        candidate = replace(self.config, **values)
        if validate:
            candidate.validate()
        return candidate

    def save(self, req: SettingsInput):
        with self.lock:
            candidate = self.candidate(req)
            entries = {}
            for name, attribute in ENV_FIELDS.items():
                value = getattr(candidate, attribute)
                value = str(value).lower() if isinstance(value, bool) else str(value)
                # Explicit process environment still takes precedence on restart.
                if name in os.environ and os.environ[name] != value:
                    raise UserError("此连接由启动环境管理，暂时不能在窗口中覆盖。请用普通双击方式启动后再设置。")
                entries[name] = json.dumps(value, ensure_ascii=False)
            lines = self.path.read_text(encoding="utf-8-sig").splitlines() if self.path.exists() else []
            preserved = [line for line in lines if line.split("=", 1)[0].strip() not in ENV_FIELDS]
            content = "\n".join(preserved + [f"{key}={value}" for key, value in entries.items()]) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".settings_", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as out:
                    out.write(content)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(name, self.path)
            finally:
                if Path(name).exists():
                    Path(name).unlink()
            self.config = candidate
            return candidate


def connection_test(config: Config, kind: str):
    # Validate only the selected connection; editing a second tab must not block
    # testing an otherwise complete first tab.
    url, key, model = (getattr(config, kind + suffix) for suffix in ("_url", "_key", "_model"))
    if not model:
        raise UserError("请先填写模型名，或点击“获取可用模型”后选择一个模型。")
    probe = replace(config, answer_mode="llm" if kind == "llm" else "extractive",
                    embedding_mode="api" if kind == "embedding" else "hash",
                    rerank_mode="api" if kind == "rerank" else "none")
    probe.validate()
    started = time.perf_counter()
    checks, warnings = [], []
    timeout = min(config.timeout, 20)
    if kind == "llm":
        body = {"model": model, "messages": [{"role": "user", "content": '连接测试。只输出 JSON：{"ok":true}'}],
                "max_tokens": 128}
        if config.json_mode:
            body["response_format"] = {"type": "json_object"}
        if urlparse(url).hostname == "api.deepseek.com":
            # A tiny probe and forced tool choice require non-thinking mode.
            body["thinking"] = {"type": "disabled"}
        response = post_json(url, "/chat/completions", key, body, timeout)
        try:
            value = json.loads(response["choices"][0]["message"]["content"])
            if not isinstance(value, dict) or value.get("ok") is not True:
                raise ValueError()
        except (KeyError, TypeError, IndexError, ValueError):
            raise UserError("接口已响应，但未返回约定 JSON。请更换模型，或调整“JSON 输出约束”。") from None
        checks.append("对话接口与 JSON 输出通过")
        tool_body = {"model": model, "messages": [{"role": "user", "content": "调用 connection_probe，参数 ok 为 true。"}],
                     "max_tokens": 128, "tools": [{"type": "function", "function": {"name": "connection_probe",
                     "description": "连接测试，不执行外部操作。", "parameters": {"type": "object",
                     "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}}],
                     "tool_choice": {"type": "function", "function": {"name": "connection_probe"}}}
        if urlparse(url).hostname == "api.deepseek.com":
            tool_body["thinking"] = {"type": "disabled"}
        try:
            tool_response = post_json(url, "/chat/completions", key, tool_body, timeout)
            call = tool_response["choices"][0]["message"]["tool_calls"][0]["function"]
            if call["name"] != "connection_probe" or json.loads(call["arguments"]).get("ok") is not True:
                raise ValueError()
            checks.append("Agent 工具调用通过")
        except (UserError, KeyError, TypeError, IndexError, ValueError):
            warnings.append("对话可用，但工具调用未通过；建议使用 RAG 流程，Agent 需支持 Tool Calling 的模型。")
    elif kind == "embedding":
        response = post_json(url, "/embeddings", key, {"model": model, "input": ["知据连接测试。"], "encoding_format": "float"}, timeout)
        try:
            item = response["data"][0]
            vector = np.asarray(item["embedding"], dtype=np.float64)
            if item["index"] != 0 or vector.ndim != 1 or not 8 <= vector.size <= 8192 or not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError()
        except (KeyError, TypeError, IndexError, ValueError):
            raise UserError("接口已响应，但向量维度或数值无效，请确认选择的是 Embedding 模型。") from None
        checks.append(f"向量接口通过 · {vector.size} 维")
    else:
        response = post_json(url, "/rerank", key, {"model": model, "query": "知据", "documents": ["知据用于资料问答。", "这是无关文字。"], "top_n": 2}, timeout)
        try:
            entries = response["results"]
            if len(entries) != 2 or {i["index"] for i in entries} != {0, 1} or not all(np.isfinite(float(i["relevance_score"])) for i in entries):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise UserError("接口已响应，但重排结果无效，请确认选择的是 Rerank 模型。") from None
        checks.append("重排接口通过")
    return {"ok": True, "checks": checks, "warnings": warnings, "duration_ms": round((time.perf_counter() - started) * 1000)}


def available_models(config: Config, kind: str):
    url, key = getattr(config, kind + "_url"), getattr(config, kind + "_key")
    # Use Config's existing HTTPS/loopback validation, without requiring a model.
    replace(config, answer_mode="llm", llm_url=url, llm_model="model-list-probe",
            embedding_mode="hash", rerank_mode="none").validate()
    try:
        with httpx.Client(timeout=min(config.timeout, 20), follow_redirects=False) as client:
            response = client.get(url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
        if response.status_code >= 300:
            raise UserError(f"无法获取模型列表（HTTP {response.status_code}），你仍可以直接填写服务商提供的模型名。")
        data = response.json()["data"]
        models = sorted({item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str) and 0 < len(item["id"]) <= 150})[:300]
        if not models:
            raise ValueError()
        return {"models": models, "note": "列表可能包含对话、向量和重排模型，请按当前用途选择。"}
    except UserError:
        raise
    except Exception:
        raise UserError("服务商未提供可用模型列表，请直接填写控制台中的模型名。") from None
