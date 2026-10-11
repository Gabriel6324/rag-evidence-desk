"""Persistent evaluation runs; callers share the engine's SQLite lock."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import re
import uuid

from .config import UserError


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def request_snapshot(config, live=False, ablation=False):
    effective = replace(config, qdrant_url="", qdrant_memory=True, rerank_mode="none")
    if not live:
        effective = replace(effective, embedding_mode="hash", answer_mode="extractive")
    return {"live": live, "ablation": ablation, "models": effective.public(),
            "parameters": {"sizes": [240] if ablation else [420] if live else [240, 420, 700],
                           "top_k": [3] if ablation else [5] if live else [3, 5],
                           "strategies": ["hybrid"] if ablation else ["vector", "hybrid"],
                           "thresholds": [.08], "overlap": 60, "expand": True}}


def report_summary(result, files):
    summary = {k: v for k, v in result.items() if k != "runs"}
    summary["runs"] = [{k: v for k, v in run.items() if k != "details"} for run in result["runs"]]
    summary["files"] = files
    return summary


def valid_legacy_report(result):
    """Only restore records the history UI can render as completed evaluations."""
    if (not isinstance(result, dict) or type(result.get('question_count')) is not int
            or result['question_count'] < 1 or not isinstance(result.get('created_at'), str)
            or not isinstance(result.get('settings'), dict) or not isinstance(result.get('mode'), dict)
            or not isinstance(result.get('runs'), list) or not result['runs']):
        return False
    rates = ('evidence_recall', 'all_evidence_at_k', 'multi_all_evidence',
             'refusal_rate_on_unanswerable', 'quote_match_rate')
    for run in result['runs']:
        if (not isinstance(run, dict) or type(run.get('size')) is not int
                or type(run.get('top_k')) is not int or run['size'] < 1 or run['top_k'] < 1
                or run.get('strategy') not in ('hybrid', 'vector')
                or type(run.get('threshold')) not in (int, float)
                or not math.isfinite(run['threshold']) or not 0 <= run['threshold'] <= 1
                or not isinstance(run.get('metrics'), dict)):
            return False
        metrics = run['metrics']
        for key in rates:
            if key not in metrics:
                return False
            value = metrics[key]
            if value is not None and (type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 1):
                return False
        if (type(metrics.get('refused_answerable_count')) is not int
                or metrics['refused_answerable_count'] < 0
                or type(metrics.get('mean_latency_ms')) not in (int, float)
                or not math.isfinite(metrics['mean_latency_ms']) or metrics['mean_latency_ms'] < 0):
            return False
    return True


class EvaluationHistory:
    def __init__(self, db, lock):
        self.db, self.lock = db, lock
        with lock:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL, finished_at TEXT, kind TEXT NOT NULL,
                    state TEXT NOT NULL, message TEXT NOT NULL, request TEXT NOT NULL,
                    result TEXT, error TEXT
                );
                CREATE INDEX IF NOT EXISTS evaluations_created ON evaluation_runs(created_at DESC, seq DESC);
            """)

    def write(self, sql, parameters):
        with self.lock:
            try:
                self.db.execute(sql, parameters)
                self.db.commit()
            except Exception:
                # The connection is shared with documents and sessions. A later
                # unrelated commit must never publish this unsuccessful write.
                self.db.rollback()
                raise

    def create(self, key, kind, request):
        self.write("INSERT INTO evaluation_runs(id,created_at,kind,state,message,request) VALUES(?,?,?,?,?,?)",
                   (key, timestamp(), kind, "running", "开始评测…", json.dumps(request, ensure_ascii=False)))
        return self.get(key)

    def progress(self, key, message):
        self.write("UPDATE evaluation_runs SET message=? WHERE id=? AND state='running'", (message, key))

    def finish(self, key, result=None, error=None):
        self.write("UPDATE evaluation_runs SET finished_at=?,state=?,message=?,result=?,error=? WHERE id=?",
                            (timestamp(), "error" if error else "done", error["message"] if error else "评测已完成",
                             json.dumps(result, ensure_ascii=False) if result is not None else None,
                             json.dumps(error, ensure_ascii=False) if error is not None else None, key))

    @staticmethod
    def record(row):
        item = dict(zip(("number", "id", "created_at", "finished_at", "kind", "state", "message", "request", "result", "error"), row))
        for field in ("request", "result", "error"):
            item[field] = json.loads(item[field]) if item[field] else None
        return item

    def get(self, key):
        if not re.fullmatch(r"[a-f0-9]{32}", key):
            raise UserError("无效的评测编号。")
        with self.lock:
            row = self.db.execute("SELECT * FROM evaluation_runs WHERE id=?", (key,)).fetchone()
            if row is None:
                raise UserError("评测记录不存在。")
            return self.record(row)

    def list(self, offset=0, limit=20):
        with self.lock:
            rows = self.db.execute("SELECT * FROM evaluation_runs ORDER BY created_at DESC,seq DESC LIMIT ? OFFSET ?",
                                   (limit, offset)).fetchall()
            total = self.db.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0]
            return {"items": [self.record(row) for row in rows], "total": total}

    def recover_interrupted(self):
        self.write("UPDATE evaluation_runs SET state='interrupted',finished_at=?,message=? WHERE state='running'",
                            (timestamp(), "服务已重启，本次评测未完成。请查看引导后重新运行。"))

    def import_reports(self, directory):
        """Restore complete pre-history reports once, without treating partial files as successes."""
        for path in sorted(directory.glob("eval_*.json")):
            if not re.fullmatch(r"eval_[a-f0-9]{10}\.json", path.name):
                continue
            key = uuid.uuid5(uuid.NAMESPACE_URL, "zhiju-report:" + path.stem).hex
            with self.lock:
                if self.db.execute("SELECT 1 FROM evaluation_runs WHERE id=?", (key,)).fetchone():
                    continue
            if not all(path.with_suffix("." + ext).is_file() for ext in ("csv", "md")):
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                if not valid_legacy_report(result):
                    continue
                settings = result["settings"]
                live, ablation = bool(settings.get("live")), bool(settings.get("ablation"))
                created = datetime.fromisoformat(result["created_at"]).astimezone(timezone.utc).isoformat(timespec="microseconds")
                models = {k: v for k, v in result.get("mode", {}).items()
                          if k in {"embedding_mode", "embedding_model", "answer_mode", "llm_model"}}
                request = {"live": live, "ablation": ablation, "models": models, "imported": True,
                           "parameters": {"sizes": sorted({r["size"] for r in result["runs"]}),
                                          "top_k": sorted({r["top_k"] for r in result["runs"]}),
                                          "strategies": sorted({r["strategy"] for r in result["runs"]}),
                                          "thresholds": sorted({r["threshold"] for r in result["runs"]}),
                                          "overlap": settings.get("overlap"), "expand": settings.get("expand")}}
                files = {ext: path.with_suffix("." + ext).name for ext in ("json", "csv", "md")}
                summary = report_summary(result, files)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            kind = "live" if live else "ablation" if ablation else "parameters"
            self.write("INSERT OR IGNORE INTO evaluation_runs(id,created_at,finished_at,kind,state,message,request,result) "
                                "VALUES(?,?,?,?,?,?,?,?)", (key, created, created, kind, "done", "评测已完成（已恢复旧报表）",
                                json.dumps(request, ensure_ascii=False), json.dumps(summary, ensure_ascii=False)))
