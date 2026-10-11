import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.app import create_app
from rag.config import Config, ROOT
from rag.evaluation import save_report
from rag.evaluation_history import request_snapshot


class EvaluationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(Path(self.tmp.name))
        self.client = None
        self.restart()
        self.report = json.loads((ROOT / "reports/evaluation.json").read_text(encoding="utf-8"))

    def tearDown(self):
        if self.client:
            self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def restart(self):
        if self.client:
            self.client.__exit__(None, None, None)
        self.app = create_app(self.cfg, bootstrap=False)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def finish(self, key):
        for _ in range(200):
            job = self.client.get("/api/jobs/" + key).json()
            if job["state"] != "running":
                return job
            time.sleep(.01)
        self.fail("evaluation did not finish")

    def start_success(self, ablation=False):
        report = copy.deepcopy(self.report)
        if ablation:
            report["settings"]["ablation"] = True
            report["runs"] = report["runs"][:4]
        target = "rag.app.run_ablation" if ablation else "rag.app.run_evaluation"
        with patch(target, return_value=report):
            response = self.client.post("/api/evaluate", json={"ablation": ablation})
            self.assertEqual(response.status_code, 200)
            started = response.json()
            self.assertEqual(started["evaluation"]["state"], "running")
            self.assertEqual(started["evaluation"]["id"], started["job_id"])
            self.assertEqual(self.finish(started["job_id"])["state"], "done")
        return self.client.get("/api/evaluations/" + started["job_id"]).json()

    def test_newest_first_distinct_exports_and_restart(self):
        first = self.start_success()
        second = self.start_success(ablation=True)
        items = self.client.get("/api/evaluations").json()["items"]
        self.assertEqual([i["id"] for i in items], [second["id"], first["id"]])
        self.assertEqual([i["kind"] for i in items], ["ablation", "parameters"])
        self.assertNotEqual(first["result"]["files"], second["result"]["files"])
        for record in items:
            self.assertNotIn("details", record["result"]["runs"][0])
            for ext, name in record["result"]["files"].items():
                download = self.client.get("/api/reports/" + name)
                self.assertEqual(download.status_code, 200)
                self.assertIn(name, download.headers["content-disposition"])
                if ext == "json":
                    self.assertIn("details", download.json()["runs"][0])
        self.restart()
        self.assertEqual(self.client.get("/api/evaluations").json()["items"], items)

    def test_failure_retains_completed_record_and_its_reports(self):
        first = self.start_success()
        with patch("rag.app.save_report", side_effect=PermissionError("reports access denied")), \
             patch("rag.app.run_evaluation", return_value=self.report):
            key = self.client.post("/api/evaluate", json={}).json()["job_id"]
            failed_job = self.finish(key)
        items = self.client.get("/api/evaluations").json()["items"]
        self.assertEqual(items[0]["state"], "error")
        self.assertIsNone(items[0]["result"])
        self.assertEqual(items[0]["error"], failed_job["error"])
        self.assertIn("保存评测报告", items[0]["error"]["stage"])
        self.assertEqual(items[1], first)
        self.assertEqual(self.client.get("/api/reports/" + first["result"]["files"]["json"]).status_code, 200)
        self.restart()
        self.assertEqual(self.client.get("/api/evaluations").json()["items"][0]["error"], failed_job["error"])

    def test_concurrent_rejection_does_not_create_phantom_run(self):
        release = threading.Event()
        def blocked(**options):
            release.wait(timeout=5)
            return self.report
        with patch("rag.app.run_evaluation", side_effect=blocked):
            key = self.client.post("/api/evaluate", json={}).json()["job_id"]
            try:
                self.assertEqual(self.client.post("/api/evaluate", json={}).status_code, 400)
                self.assertEqual(self.client.get("/api/evaluations").json()["total"], 1)
            finally:
                release.set()
            self.assertEqual(self.finish(key)["state"], "done")

    def test_snapshot_uses_run_mode_without_keys_or_urls(self):
        self.cfg.answer_mode, self.cfg.embedding_mode = "llm", "api"
        self.cfg.llm_model, self.cfg.embedding_model = "test-chat", "test-vector"
        self.cfg.llm_key, self.cfg.embedding_key = "private-chat-key", "private-vector-key"
        self.cfg.rerank_key, self.cfg.qdrant_key = "private-rerank-key", "private-db-key"
        for live in (False, True):
            with patch("rag.app.run_evaluation", return_value=self.report):
                started = self.client.post("/api/evaluate", json={"live": live}).json()
                self.finish(started["job_id"])
            snapshot = started["evaluation"]["request"]
            serialized = json.dumps(snapshot)
            for secret in (self.cfg.llm_key, self.cfg.embedding_key, self.cfg.rerank_key, self.cfg.qdrant_key):
                self.assertNotIn(secret, serialized)
            self.assertNotIn("https://", serialized)
            self.assertEqual(snapshot["models"]["answer_mode"], "llm" if live else "extractive")
            self.assertEqual(snapshot["models"]["embedding_mode"], "api" if live else "hash")

    def test_restart_marks_running_interrupted_without_model_calls(self):
        key = "f" * 32
        self.app.state.evaluations.create(key, "live", request_snapshot(self.cfg, live=True))
        with patch("rag.app.run_evaluation") as evaluate:
            self.restart()
            record = self.client.get("/api/evaluations/" + key).json()
            self.assertEqual(record["state"], "interrupted")
            self.assertIsNone(record["result"])
            evaluate.assert_not_called()

    def test_complete_legacy_reports_restore_once_and_keep_exports(self):
        directory = self.cfg.data_dir / "reports"
        save_report(self.report, directory, "eval_1234567890")
        (directory / "eval_aaaaaaaaaa.json").write_text("{", encoding="utf-8")
        (directory / "eval_bbbbbbbbbb.json").write_text(json.dumps(self.report), encoding="utf-8")
        self.restart()
        records = self.client.get("/api/evaluations").json()
        self.assertEqual(records["total"], 1)
        self.assertTrue(records["items"][0]["request"]["imported"])
        self.assertEqual(records["items"][0]["result"]["files"]["json"], "eval_1234567890.json")
        self.restart()
        self.assertEqual(self.client.get("/api/evaluations").json()["total"], 1)
        self.assertEqual(self.client.get("/api/reports/eval_1234567890.json").status_code, 200)

    def test_pagination_and_invalid_ids(self):
        first, second = self.start_success(), self.start_success()
        self.assertEqual(self.client.get("/api/evaluations?limit=1").json()["items"][0]["id"], second["id"])
        self.assertEqual(self.client.get("/api/evaluations?limit=1&offset=1").json()["items"][0]["id"], first["id"])
        for query in ("limit=0", "limit=51", "offset=-1"):
            self.assertEqual(self.client.get("/api/evaluations?" + query).status_code, 422)
        self.assertEqual(self.client.get("/api/evaluations/not-a-run").status_code, 400)
        self.assertEqual(self.client.get("/api/reports/eval_" + "a" * 31 + ".json").status_code, 400)


if __name__ == "__main__":
    unittest.main()
