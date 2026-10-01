from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from . import __version__
from .config import Config, ROOT, UserError
from .documents import extract, safe_name, chunk_units, SUSPICIOUS
from .models import Embeddings, tokens, generate


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def subqueries(question, expand):
    if not expand:
        return [question]
    # Transparent, reproducible query splitting. This is not an LLM planner.
    pieces = re.split(r"[？?；;]|以及|同时|并且|然后|另外", question)
    out = [question]
    for part in pieces:
        part = part.strip(" ，,。")
        if 5 <= len(part) < len(question) and part not in out:
            out.append(part)
    return out[:4]


class Engine:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(config.data_dir / "catalog.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, name TEXT, sha TEXT UNIQUE, units TEXT, chars INTEGER, created TEXT);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        if self.meta("workspace") is None:
            self.set_meta("workspace", uuid.uuid4().hex[:12])
            self.set_meta("revision", 0)
        self.embedder = Embeddings(config)
        if config.qdrant_url:
            self.qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_key or None, timeout=config.timeout)
        else:
            self.qdrant = QdrantClient(path=str(config.data_dir / "qdrant"))
        self.active = self.meta("active")
        self.collection_available = bool(
            self.active and self.active.get("collection") and
            self.active.get("storage_fingerprint") == config.storage_fingerprint and
            self.qdrant.collection_exists(self.active["collection"])
        )
        self.chunks = self.active.get("chunks", []) if self.active else []
        self._prepare_bm25()

    def meta(self, key):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def set_meta(self, key, value):
        try:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def documents(self):
        with self.lock:
            return [dict(zip(["id", "name", "chars", "created"], row)) for row in self.db.execute(
                "SELECT id,name,chars,created FROM docs ORDER BY created,name,id")]

    def ingest(self, name, content):
        name = safe_name(name)
        sha = hashlib.sha256(content).hexdigest()
        with self.lock:
            old = self.db.execute("SELECT id,name FROM docs WHERE sha=?", (sha,)).fetchone()
            if old:
                return {"id": old[0], "name": old[1], "duplicate": True}
            if self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] >= 60:
                raise UserError("演示系统最多导入 60 份资料。")
            units = extract(name, content)
            chars = sum(len(u["text"]) for u in units)
            total = self.db.execute("SELECT COALESCE(SUM(chars),0) FROM docs").fetchone()[0]
            if total + chars > 2_000_000:
                raise UserError("资料库最多 200 万字符，请移除部分资料。")
            doc_id = uuid.uuid4().hex
            try:
                self.db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (doc_id, name, sha, json.dumps(units, ensure_ascii=False), chars, now()))
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('revision',?)", (json.dumps(self.meta("revision") + 1),))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return {"id": doc_id, "name": name, "chars": chars, "duplicate": False}

    def remove(self, doc_id):
        with self.lock:
            try:
                cursor = self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,))
                if cursor.rowcount == 0:
                    raise UserError("资料不存在或已删除。")
                self.set_meta("revision", self.meta("revision") + 1)
            except Exception:
                self.db.rollback()
                raise

    def source(self, doc_id):
        with self.lock:
            r = self.db.execute("SELECT name,units FROM docs WHERE id=?", (doc_id,)).fetchone()
            if r is None:
                raise UserError("资料不存在。")
            return {"name": r[0], "units": json.loads(r[1])}

    def seed(self):
        for path in sorted((ROOT / "samples").glob("*.md")):
            self.ingest(path.name, path.read_bytes())

    def stale(self):
        return (not self.active or not self.collection_available or self.active["revision"] != self.meta("revision") or
                self.active["fingerprint"] != self.config.embedding_fingerprint or
                self.active.get("storage_fingerprint") != self.config.storage_fingerprint)

    def status(self):
        with self.lock:
            idx = {k: v for k, v in (self.active or {}).items() if k not in {"chunks", "fingerprint", "collection", "storage_fingerprint"}}
            return {"config": self.config.public(), "documents": self.documents(), "index": idx,
                    "needs_rebuild": self.stale(), "version": __version__}

    def build(self, size=420, overlap=70, progress=lambda s: None):
        with self.lock:
            started = time.perf_counter()
            chunks = []
            progress("正在按来源切分资料…")
            for doc_id, name, units in self.db.execute("SELECT id,name,units FROM docs ORDER BY name,sha"):
                chunks += chunk_units(doc_id, name, json.loads(units), size, overlap)
            if not chunks:
                raise UserError("请先导入至少一份含文字的资料。")
            if len(chunks) > 6000:
                raise UserError("切分超过 6000 个片段，请增大 chunk 或减少资料。")
            for i, c in enumerate(chunks):
                c["point_id"] = i
            progress(f"正在为 {len(chunks)} 个片段生成向量…")
            vectors = self.embedder.embed([c["text"] for c in chunks])
            dim = len(vectors[0])
            collection = f"rag_{self.meta('workspace')}_{uuid.uuid4().hex[:12]}"
            try:
                self.qdrant.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
                for start in range(0, len(chunks), 64):
                    progress(f"正在写入 Qdrant：{min(start + 64, len(chunks))}/{len(chunks)}")
                    self.qdrant.upsert(collection, points=[models.PointStruct(id=c["point_id"], vector=v,
                        payload={"doc_id": c["doc_id"], "blocked": c["blocked"]})
                        for c, v in zip(chunks[start:start + 64], vectors[start:start + 64])], wait=True)
                active = {"collection": collection, "revision": self.meta("revision"), "fingerprint": self.config.embedding_fingerprint,
                          "storage_fingerprint": self.config.storage_fingerprint,
                          "size": size, "overlap": overlap, "dimension": dim, "count": len(chunks),
                          "blocked_count": sum(c["blocked"] for c in chunks), "built_at": now(), "chunks": chunks,
                          "build_ms": round((time.perf_counter() - started) * 1000)}
                # Only publish after all writes succeed. A failure preserves the previous active index.
                self.set_meta("active", active)
            except Exception:
                try:
                    self.qdrant.delete_collection(collection)
                except Exception:
                    pass
                raise
            old = self.active
            self.active, self.chunks = active, chunks
            self.collection_available = True
            self._prepare_bm25()
            if old and old.get("storage_fingerprint") == self.config.storage_fingerprint:
                try:
                    self.qdrant.delete_collection(old["collection"])
                except Exception:
                    pass  # Orphan collections are harmless; never delete another workspace's data.
            progress("索引已就绪")
            return {k: v for k, v in active.items() if k not in {"chunks", "fingerprint", "collection", "storage_fingerprint"}}

    def _prepare_bm25(self):
        self.tf = [Counter(tokens(c["text"])) for c in self.chunks]
        self.df = Counter(t for c in self.tf for t in c)
        self.lengths = [sum(c.values()) for c in self.tf]
        self.avglen = sum(self.lengths) / max(1, len(self.lengths))

    def _bm25(self, query, eligible):
        result = []
        n = max(1, len(self.chunks))
        for i in eligible:
            score = 0.0
            for term in set(tokens(query)):
                freq = self.tf[i].get(term, 0)
                if freq:
                    idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * freq * 2.2 / (freq + 1.2 * (.25 + .75 * self.lengths[i] / max(1, self.avglen)))
            if score > 0:
                result.append((i, score))
        return sorted(result, key=lambda x: (-x[1], x[0]))

    def retrieve(self, question, top_k=5, strategy="hybrid", threshold=.08, expand=True, doc_ids=None):
        with self.lock:
            if self.stale():
                raise UserError("资料或向量配置已变化，请先重建索引，避免查询旧资料。")
            if not 1 <= top_k <= 12 or strategy not in {"hybrid", "vector"} or not 0 <= threshold <= 1:
                raise UserError("检索参数无效。")
            selected = set(doc_ids or [])
            eligible = [i for i, c in enumerate(self.chunks) if not c["blocked"] and (not selected or c["doc_id"] in selected)]
            if not eligible:
                return [], subqueries(question, expand)
            queries = subqueries(question, expand)
            vectors = self.embedder.embed(queries)
            if len(vectors[0]) != self.active["dimension"]:
                raise UserError("查询向量维度与索引不一致，请重建索引。")
            rrf, cosines, lexical = defaultdict(float), defaultdict(lambda: -1.0), defaultdict(float)
            limit = min(max(top_k * 5, 30), len(eligible))
            conditions = [models.FieldCondition(key="blocked", match=models.MatchValue(value=False))]
            if selected:
                conditions.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(selected))))
            for query, vector in zip(queries, vectors):
                hits = self.qdrant.query_points(self.active["collection"], query=vector, limit=limit,
                    query_filter=models.Filter(must=conditions), with_payload=False).points
                for rank, hit in enumerate(hits, 1):
                    i = int(hit.id)
                    cosines[i] = max(cosines[i], float(hit.score))
                    rrf[i] += 1 / (60 + rank)
                if strategy == "hybrid":
                    bhits = self._bm25(query, eligible)[:limit]
                    for rank, (i, score) in enumerate(bhits, 1):
                        lexical[i] = max(lexical[i], score)
                        rrf[i] += 1 / (60 + rank)
                    # BM25-only candidates still get an explicit cosine threshold check.
                    extra = [i for i, _ in bhits if i not in {int(h.id) for h in hits}]
                    if extra:
                        extra_vecs = self.embedder.embed([self.chunks[i]["text"] for i in extra])
                        for i, ev in zip(extra, extra_vecs):
                            cosines[i] = max(cosines[i], float(np.dot(vector, ev)))
            order = sorted(rrf, key=lambda i: (-rrf[i], -cosines[i], i))
            evidence = []
            for i in order:
                c = self.chunks[i]
                if cosines[i] < threshold:
                    continue
                # Remove near-duplicate overlapping chunks while retaining independent evidence.
                duplicate = False
                for prev in evidence:
                    if prev["doc_id"] == c["doc_id"] and prev["unit"] == c["unit"]:
                        common = max(0, min(prev["end"], c["end"]) - max(prev["start"], c["start"]))
                        if common / min(len(prev["text"]), len(c["text"])) > .65:
                            duplicate = True
                if duplicate:
                    continue
                evidence.append({**c, "id": f"E{len(evidence) + 1}", "cosine": round(cosines[i], 4),
                                 "rrf": round(rrf[i], 6), "bm25": round(lexical[i], 4)})
                if len(evidence) == top_k:
                    break
            return evidence, queries

    def ask(self, question, **options):
        question = question.strip()
        if not question or len(question) > 1500:
            raise UserError("请输入 1–1500 字的问题。")
        with self.lock:
            started = time.perf_counter()
            if SUSPICIOUS.search(question):
                return {"question": question, "status": "insufficient", "claims": [], "missing": ["此问题包含改变系统规则或索取密钥的指令，未执行。"],
                        "verified_quotes": 0, "evidence": [], "queries": [question], "timing": {"retrieval_ms": 0, "total_ms": 0},
                        "mode": self.config.answer_mode, "notice": "规则筛查只覆盖部分已知模式。"}
            evidence, queries = self.retrieve(question, **options)
            retrieved = time.perf_counter()
            weights = {t: math.log(1 + (len(self.chunks) + .5) / (self.df[t] + .5)) for t in set(tokens(question))}
            result = generate(self.config, question, evidence, term_weights=weights)
            return {**result, "question": question, "evidence": evidence, "queries": queries,
                    "mode": self.config.answer_mode,
                    "notice": ("当前为原文摘录模式，不进行跨资料推理；摘录可能只覆盖部分问题。" if self.config.answer_mode == "extractive" else
                               "已核验引用编号与引文匹配；结论是否由证据充分支持仍需人工核对。"),
                    "timing": {"retrieval_ms": round((retrieved - started) * 1000, 1), "total_ms": round((time.perf_counter() - started) * 1000, 1)}}

    def close(self):
        with self.lock:
            self.embedder.close()
            self.qdrant.close()
            self.db.close()
