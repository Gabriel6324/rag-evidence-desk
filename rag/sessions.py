"""Bounded SQLite conversations. Old answers are context, never fresh source evidence."""
from datetime import datetime, timezone
import json
import re
import uuid

from .config import UserError


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SessionStore:
    def __init__(self, db):
        self.db = db
        db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS turns (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            revision INTEGER NOT NULL, scope TEXT NOT NULL, question TEXT NOT NULL,
            response TEXT NOT NULL, created TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, seq);
        CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, trace TEXT NOT NULL, created TEXT NOT NULL);
        """)

    def require(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise UserError("无效的会话编号。")
        row = self.db.execute("SELECT id,title,created,updated FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise UserError("会话不存在或已删除，请新建会话。")
        return dict(zip(("id", "title", "created", "updated"), row))

    def create(self, title="新会话"):
        if self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= 200:
            raise UserError("最多保存 200 个会话，请先删除不需要的会话。")
        key, created = uuid.uuid4().hex, timestamp()
        self.db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (key, title[:80], created, created))
        self.db.commit()
        return self.require(key)

    def list(self):
        return [dict(zip(("id", "title", "created", "updated", "turn_count"), row)) for row in self.db.execute(
            "SELECT s.id,s.title,s.created,s.updated,COUNT(t.seq) FROM sessions s LEFT JOIN turns t "
            "ON t.session_id=s.id GROUP BY s.id ORDER BY s.updated DESC LIMIT 200")]

    def detail(self, session_id):
        session = self.require(session_id)
        rows = self.db.execute("SELECT question,response,created FROM turns WHERE session_id=? ORDER BY seq", (session_id,))
        session["turns"] = [{"question": q, "response": json.loads(r), "created": created} for q, r, created in rows]
        return session

    def remove(self, session_id):
        self.require(session_id)
        try:
            self.db.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
            self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    @staticmethod
    def scope(doc_ids):
        return json.dumps(sorted(set(doc_ids or [])))

    def context(self, session_id, revision, doc_ids, fingerprint, budget=1600):
        if not session_id:
            return [], 0
        self.require(session_id)
        rows = self.db.execute("SELECT question,response FROM turns WHERE session_id=? AND revision=? AND scope=? "
                               "ORDER BY seq DESC LIMIT 3", (session_id, revision, self.scope(doc_ids))).fetchall()
        recent, chars = [], 0
        for question, stored in rows:
            response = json.loads(stored)
            if response.get("index_fingerprint") != fingerprint:
                continue
            text = "\n".join(c["text"] for c in response.get("claims", []))
            if not text:
                continue
            entry = {"question": question[:500], "answer": text[:700]}
            size = len(json.dumps(entry, ensure_ascii=False))
            if chars + size > budget:
                break
            chars += size
            recent.append(entry)
        return list(reversed(recent)), chars

    def append(self, session_id, revision, doc_ids, response):
        self.require(session_id)
        created = timestamp()
        try:
            self.db.execute("INSERT INTO turns(session_id,revision,scope,question,response,created) VALUES (?,?,?,?,?,?)",
                            (session_id, revision, self.scope(doc_ids), response["question"],
                             json.dumps(response, ensure_ascii=False), created))
            # Retain the newest 100 turns per session, including their evidence snapshots.
            self.db.execute("DELETE FROM turns WHERE session_id=? AND seq NOT IN "
                            "(SELECT seq FROM turns WHERE session_id=? ORDER BY seq DESC LIMIT 100)", (session_id, session_id))
            self.db.execute("UPDATE sessions SET updated=?,title=CASE WHEN title='新会话' THEN ? ELSE title END WHERE id=?",
                            (created, response["question"][:80], session_id))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def save_trace(self, trace):
        try:
            self.db.execute("INSERT OR REPLACE INTO traces VALUES (?,?,?)",
                            (trace["id"], json.dumps(trace, ensure_ascii=False), trace["created_at"]))
            self.db.execute("DELETE FROM traces WHERE id NOT IN (SELECT id FROM traces ORDER BY created DESC,rowid DESC LIMIT 200)")
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def trace(self, trace_id):
        row = self.db.execute("SELECT trace FROM traces WHERE id=?", (trace_id,)).fetchone()
        if row is None:
            raise UserError("运行记录不存在或已超出保留数量。")
        return json.loads(row[0])

    def traces(self):
        traces = [json.loads(row[0]) for row in self.db.execute("SELECT trace FROM traces ORDER BY created DESC,rowid DESC LIMIT 50")]
        return [{k: t[k] for k in ("id", "workflow", "created_at", "state", "total_ms")} for t in traces]
