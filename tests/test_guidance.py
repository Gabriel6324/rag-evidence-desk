"""Regression checks for specific diagnostics, isolation, and public configuration."""
import ast
import errno
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from rag.app import Jobs, create_app
from rag.config import Config, ROOT, UserError
from rag.errors import diagnose, RULES, save_diagnostic
from rag.evaluation import run_ablation, run_evaluation
from rag.engine import Engine
from rag.settings import SettingsInput, SettingsStore


class DiagnosticTests(unittest.TestCase):
    def test_http_statuses_have_distinct_actionable_guides(self):
        codes = set()
        for status in [400,401,402,403,404,408,413,415,422,429,503,302]:
            with self.subTest(status=status):
                d = diagnose(UserError(f"模型 API 请求失败（HTTP {status}）。"), stage="测试向量接口")
                self.assertEqual(d["http_status"], status)
                self.assertGreaterEqual(len(d["steps"]), 2)
                self.assertEqual(d["stage"], "测试向量接口")
                codes.add(d["code"])
        self.assertEqual(len(codes), 12)

    def test_permission_error_keeps_resource_and_stage(self):
        error = PermissionError(errno.EACCES, "Permission denied", "/reports/report.json")
        d = diagnose(error, stage="保存评测报告", operation="离线参数评测")
        self.assertEqual(d["code"], "FILESYSTEM_PERMISSION")
        self.assertEqual(d["resource"], "/reports/report.json")
        self.assertIn("保存评测报告", d["details"])
        self.assertNotIn("检查依赖、配置或 Qdrant", d["message"])

    def test_windows_file_lock_differs_from_permission(self):
        error = PermissionError(13, "file in use", "storage.sqlite")
        error.winerror = 32
        d = diagnose(error, stage="清理评测临时文件")
        self.assertEqual(d["code"], "FILE_IN_USE")

    def test_caused_network_timeout_is_not_json_error(self):
        error = UserError("无法连接模型 API 或返回格式不是 JSON，请检查地址与网络。")
        error.__cause__ = httpx.ReadTimeout("not exposed")
        self.assertEqual(diagnose(error)["code"], "NETWORK_TIMEOUT")
        error.__cause__ = httpx.ConnectError("not exposed")
        self.assertEqual(diagnose(error)["code"], "NETWORK_CONNECTION")
        error.__cause__ = json.JSONDecodeError("bad", "", 0)
        self.assertEqual(diagnose(error)["code"], "MODEL_JSON")

    def test_secrets_and_provider_bodies_never_appear_in_details(self):
        secret = "private-credential-123"
        error = UserError("请检查配置 " + secret)
        error.__cause__ = RuntimeError("response containing " + secret + " and other private data")
        d = diagnose(error, stage=secret, secrets=[secret])
        self.assertNotIn(secret, json.dumps(d, ensure_ascii=False))
        self.assertNotIn("other private data", d["details"])

    def test_unknown_failure_does_not_invent_cause(self):
        d = diagnose(RuntimeError("untrusted provider response"))
        self.assertEqual(d["code"], "UNCLASSIFIED")
        self.assertIn("暂时无法确定", d["cause"])
        self.assertNotIn("untrusted provider response", d["details"])
        self.assertIn("RuntimeError", d["details"])

    def test_invalid_evaluation_json_has_dataset_guide(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "questions.json"
            dataset.write_text("{broken", encoding="utf-8")
            try:
                run_evaluation(dataset=dataset)
            except UserError as error:
                diagnostic = diagnose(error, stage="读取评测问题与证据标注")
            else:
                self.fail("invalid evaluation JSON was accepted")
            self.assertEqual(diagnostic["code"], "EVAL_DATASET")
            self.assertTrue(any("JSON" in step for step in diagnostic["steps"]))
            self.assertNotIn("模型返回", diagnostic["title"])

    def test_filesystem_and_dependency_types(self):
        cases = [
            (OSError(errno.ENOSPC, "full"), "DISK_FULL"),
            (FileNotFoundError("missing"), "FILE_NOT_FOUND"),
            (ModuleNotFoundError("missing"), "DEPENDENCY_MISSING"),
        ]
        for error, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(diagnose(error)["code"], expected)

    def test_all_explicit_user_errors_have_guide(self):
        for path in (ROOT / "rag").glob("*.py"):
            if path.name == "errors.py":
                continue
            for item in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(item, ast.Raise) and isinstance(item.exc, ast.Call)
                        and isinstance(item.exc.func, ast.Name) and item.exc.func.id == "UserError" and item.exc.args):
                    continue
                argument = item.exc.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    message = argument.value
                elif isinstance(argument, ast.JoinedStr):
                    message = "".join(v.value if isinstance(v, ast.Constant) else "401" for v in argument.values)
                else:
                    continue
                with self.subTest(path=path.name, line=item.lineno, message=message):
                    self.assertNotEqual(diagnose(UserError(message))["code"], "UNCLASSIFIED")

    def test_diagnostic_write_failure_preserves_original(self):
        d = diagnose(PermissionError(13, "denied", "/reports/output.json"))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "file"
            file.write_text("occupied")
            self.assertIsNone(save_diagnostic(d, file))
            self.assertEqual(d["code"], "FILESYSTEM_PERMISSION")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(data_dir=Path(self.tmp.name))
        self.client = TestClient(create_app(self.cfg))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def wait_job(self, key):
        for _ in range(200):
            item = self.client.get("/api/jobs/" + key).json()
            if item["state"] != "running":
                return item
            time.sleep(.01)
        self.fail("job did not complete")

    def test_failed_evaluation_api_and_sse_keep_stage_and_redact(self):
        def failure(progress, **kwargs):
            progress("保存评测报告")
            raise PermissionError(13, "denied", "/reports/output.json")
        with patch("rag.app.run_evaluation", failure):
            job = self.client.post("/api/evaluate", json={}).json()["job_id"]
            item = self.wait_job(job)
        self.assertEqual(item["state"], "error")
        self.assertEqual(item["error"]["stage"], "保存评测报告")
        self.assertEqual(item["error"]["code"], "FILESYSTEM_PERMISSION")
        with self.client.stream("GET", "/api/jobs/" + job + "/events") as response:
            content = b"".join(response.iter_bytes()).decode()
        self.assertIn("event: failed", content)
        self.assertIn("FILESYSTEM_PERMISSION", content)
        self.assertTrue((self.cfg.data_dir / "diagnostics" / item["error"]["diagnostic_file"]).is_file())

    def test_validation_and_upload_errors_are_structured(self):
        result = self.client.post("/api/ask", json={"question":"x","top_k":0}).json()
        self.assertEqual(result["error"]["code"], "SETTING_VALIDATION")
        result = self.client.post("/api/documents", files={"file":("empty.txt", b"", "text/plain")}).json()
        self.assertEqual(result["error"]["code"], "FILE_SIZE")
        self.assertIn("导入资料", result["error"]["stage"])

    def test_local_request_guard_has_distinct_structured_guides(self):
        cases = [
            ({"origin": "https://other-site.example"}, "LOCAL_REQUEST_BLOCKED", 403),
            ({"content-length": "invalid"}, "REQUEST_LENGTH_INVALID", 400),
            ({"content-length": "-1"}, "REQUEST_LENGTH_INVALID", 400),
            ({"content-length": str(12 * 1024 * 1024)}, "REQUEST_TOO_LARGE", 413),
        ]
        for headers, code, status in cases:
            with self.subTest(code=code, headers=headers):
                response = self.client.post("/api/evaluate", json={}, headers=headers)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], code)
                self.assertTrue(response.json()["error"]["steps"])

    def test_saved_keys_write_only_and_hot_reload(self):
        payload = self.client.get("/api/settings").json()
        key = "not-a-login-password-private-123"
        payload["llm"]["key"] = key
        payload["llm"].pop("has_key")
        for kind in ("embedding", "rerank"):
            payload[kind].pop("has_key")
        response = self.client.post("/api/settings", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(key, response.text)
        state = self.client.get("/api/settings")
        self.assertNotIn(key, state.text)
        self.assertTrue(state.json()["llm"]["has_key"])
        self.assertEqual(len(self.client.get("/api/status").json()["documents"]), 5)

    def test_guides_cover_fields_and_preserve_public_sentence(self):
        guides = self.client.get("/api/guides").json()
        for kind in ("llm", "embedding", "rerank"):
            for field in ("url","key","model","enabled","show","clear","models","test"):
                self.assertTrue(guides[kind + "_" + field]["steps"])
        self.assertIn("这个是个人的，我不提供", guides["key"]["note"])

    def test_agent_and_session_remain_usable(self):
        job = self.client.post("/api/ask", json={"question":"RRF 怎样融合不同检索结果？","workflow":"agent"}).json()["job_id"]
        item = self.wait_job(job)
        self.assertEqual(item["state"], "done")
        self.assertEqual(item["result"]["status"], "answered")
        self.assertGreater(item["result"]["verified_quotes"], 0)
        self.assertEqual(item["result"]["agent"]["framework"], "LangGraph")


class StorageRegressionTests(unittest.TestCase):
    def test_repeated_local_builds_close_discarded_collections(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(Config(data_dir=Path(folder)))
            try:
                engine.seed()
                engine.build()
                old = engine.qdrant._client.collections[engine.active["collection"]]
                engine.build()
                with self.assertRaises(Exception):
                    old.storage.storage.execute("SELECT 1")
            finally:
                engine.close()
            self.assertTrue(Path(folder).is_dir())

    def test_evaluation_uses_memory_and_keeps_user_data(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Config(data_dir=Path(folder))
            marker = Path(folder) / "keep.txt"
            marker.write_text("user data")
            result = run_evaluation(cfg, sizes=(420,), ks=(5,), strategies=("hybrid",))
            self.assertEqual(result["mode"]["qdrant_mode"], "评测内存隔离")
            self.assertEqual(marker.read_text(), "user data")
            self.assertFalse((Path(folder) / "qdrant").exists())
            self.assertEqual(result["runs"][0]["metrics"]["refused_answerable_count"], 0)

    def test_four_variants_match_recorded_baseline(self):
        result = run_ablation()
        a, b, c, d = [v["metrics"] for v in result["runs"]]
        self.assertEqual(a["all_evidence_at_k"], .9286)
        self.assertEqual(d["all_evidence_at_k"], 1.0)
        self.assertEqual(a["refusal_rate_on_unanswerable"], .5)
        self.assertEqual(d["refusal_rate_on_unanswerable"], 1.0)
        self.assertEqual(d["refused_answerable_count"], 0)


if __name__ == "__main__":
    unittest.main()

