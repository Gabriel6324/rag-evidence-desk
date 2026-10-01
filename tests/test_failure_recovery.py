"""Unsuccessful writes must remain unsuccessful after the next operation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from rag.app import create_app
from rag.config import Config, UserError
from rag.engine import Engine
from rag.models import Embeddings, validate_answer
from rag.evaluation import run_evaluation


class FailCommitOnce:
    def __init__(self, connection):
        self.connection = connection
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected commit failure")
        self.connection.commit()


class FailureRecoveryTests(unittest.TestCase):
    def test_invalid_env_values_and_url_ports_have_actionable_errors(self):
        for values in [{"API_TIMEOUT_SECONDS": "ten"}, {"LLM_JSON_MODE": "maybe"}, {"RAG_EMBEDDING_MODE": "api", "EMBEDDING_BASE_URL": "http://localhost:bad/v1"}]:
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True), self.assertRaises(UserError):
                Config.from_env()

    def test_empty_or_missing_evaluation_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "notes.md").write_text("# 要求\n课程项目需要提交实验报告。", encoding="utf-8")
            dataset = root / "questions.json"
            for atom in [{"source": "notes.md", "quote": ""}, {"source": "missing.md", "quote": "课程项目需要提交实验报告。"}, {"source": "notes.md", "quote": "不存在的实验规定"}]:
                dataset.write_text(json.dumps([{"id": "test", "question": "需要提交什么？", "type": "single", "evidence": [atom]}], ensure_ascii=False), encoding="utf-8")
                with self.subTest(atom=atom), self.assertRaises(UserError):
                    run_evaluation(dataset=dataset, samples=root, sizes=[420], ks=[5], strategies=["hybrid"])

    def test_malformed_status_is_a_controlled_model_error(self):
        for value in [[], {}, 1, None, True]:
            with self.subTest(status=value), self.assertRaises(UserError):
                validate_answer({"status": value}, [])

    def test_extreme_api_vectors_keep_unit_norm(self):
        with tempfile.TemporaryDirectory() as d:
            embedder = Embeddings(Config(Path(d), embedding_mode="api"))
            try:
                for value in [1e38, 1e-38, 1e300, 1e-300]:
                    with self.subTest(value=value), patch("rag.models.post_json", return_value={"data": [{"index": 0, "embedding": [value] * 8}]}):
                        vector = embedder.embed([str(value)])[0]
                        self.assertTrue(np.isfinite(vector).all())
                        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
            finally:
                embedder.close()

    def test_failed_ingest_is_rolled_back_before_an_unrelated_commit(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(Path(d))
            engine = Engine(cfg)
            try:
                original = engine.db
                engine.db = FailCommitOnce(original)
                with self.assertRaises(RuntimeError):
                    engine.ingest("failed.txt", "不应被错误提交的内容。".encode())
                self.assertFalse(original.in_transaction)
                self.assertEqual(engine.documents(), [])
                engine.set_meta("unrelated", True)
                self.assertEqual(engine.meta("revision"), 0)
            finally:
                engine.close()
            engine = Engine(cfg)
            try:
                self.assertEqual(engine.documents(), [])
            finally:
                engine.close()

    def test_failed_cache_commit_does_not_publish_uncommitted_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            embedder = Embeddings(Config(Path(d), embedding_mode="api"))
            try:
                original = embedder.cache
                embedder.cache = FailCommitOnce(original)
                payload = {"data": [{"index": 0, "embedding": [1.] + [0.] * 7}]}
                with patch("rag.models.post_json", return_value=payload):
                    with self.assertRaises(RuntimeError):
                        embedder.embed(["failed"])
                    self.assertFalse(original.in_transaction)
                    self.assertEqual(original.execute("SELECT COUNT(*) FROM cache").fetchone()[0], 0)
                    embedder.embed(["success"])
                    self.assertEqual(original.execute("SELECT COUNT(*) FROM cache").fetchone()[0], 1)
            finally:
                embedder.close()

    def test_failed_delete_restores_document_and_revision(self):
        with tempfile.TemporaryDirectory() as d:
            engine = Engine(Config(Path(d)))
            try:
                doc = engine.ingest("keep.txt", "删除失败时仍然保留的资料。".encode())
                revision = engine.meta("revision")
                engine.db = FailCommitOnce(engine.db)
                with self.assertRaises(RuntimeError):
                    engine.remove(doc["id"])
                engine.set_meta("unrelated", True)
                self.assertEqual(engine.documents()[0]["id"], doc["id"])
                self.assertEqual(engine.meta("revision"), revision)
            finally:
                engine.close()

    def test_failed_bootstrap_releases_qdrant_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(Path(d))
            created = []
            def create_engine(config):
                engine = Engine(config)
                created.append(engine)
                return engine
            try:
                with patch("rag.app.Engine", side_effect=create_engine), patch.object(Engine, "seed", side_effect=UserError("injected bootstrap failure")):
                    with self.assertRaises(UserError):
                        with TestClient(create_app(cfg)):
                            pass
                with TestClient(create_app(cfg)) as client:
                    self.assertEqual(client.get("/api/status").status_code, 200)
            finally:
                for engine in created:
                    try:
                        engine.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
