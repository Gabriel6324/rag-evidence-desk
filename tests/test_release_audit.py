"""Additional acceptance and fault-injection checks for the frozen release."""
import copy
import json
import os
from pathlib import Path
import shlex
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

from rag.app import create_app
from rag.config import Config, ROOT, UserError
from rag.engine import Engine
from rag.evaluation_history import EvaluationHistory
from rag.errors import RULES
from rag.settings import SettingsInput, SettingsStore, connection_test, available_models
from rag.reranking import rerank


class FailCommitOnce:
    def __init__(self, connection):
        self.connection, self.failed = connection, False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        if not self.failed:
            self.failed = True
            raise sqlite3.OperationalError('database or disk is full')
        self.connection.commit()


class ReleaseAuditTests(unittest.TestCase):
    def test_rejected_history_creation_cannot_be_committed_later(self):
        with tempfile.TemporaryDirectory() as d:
            db = sqlite3.connect(Path(d) / 'catalog.sqlite')
            try:
                history = EvaluationHistory(db, threading.RLock())
                history.db = FailCommitOnce(db)
                with self.assertRaises(sqlite3.OperationalError):
                    history.create(uuid.uuid4().hex, 'parameters', {})
                db.execute('CREATE TABLE unrelated(value)')
                db.commit()
                self.assertEqual(history.list()['total'], 0)
                self.assertFalse(db.in_transaction)
            finally:
                db.close()

    def test_failed_progress_and_completion_keep_last_saved_state(self):
        for operation in ('progress', 'finish', 'recover_interrupted'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as d:
                db = sqlite3.connect(Path(d) / 'catalog.sqlite')
                try:
                    history = EvaluationHistory(db, threading.RLock())
                    key = uuid.uuid4().hex
                    initial = history.create(key, 'parameters', {})
                    history.db = FailCommitOnce(db)
                    with self.assertRaises(sqlite3.OperationalError):
                        if operation == 'progress':
                            history.progress(key, 'This update failed')
                        elif operation == 'finish':
                            history.finish(key, result={'runs': []})
                        else:
                            history.recover_interrupted()
                    db.commit()  # A later unrelated commit must not resurrect failed writes.
                    self.assertEqual(history.get(key), initial)
                finally:
                    db.close()

    def test_malformed_legacy_reports_do_not_block_startup_or_history(self):
        valid = json.loads((ROOT / 'reports/evaluation.json').read_text(encoding='utf-8'))
        cases = [[], None, 'broken', 42, {'runs': [None], 'question_count': 18},
                 {**valid, 'settings': []}, {**valid, 'mode': []},
                 {**valid, 'runs': [{**valid['runs'][0], 'metrics': []}]},
                 {**valid, 'runs': [{**valid['runs'][0], 'metrics': {}}]},
                 {**valid, 'runs': [{**valid['runs'][0], 'metrics': {
                     **valid['runs'][0]['metrics'], 'all_evidence_at_k': 'unknown'}}]}]
        for report in cases:
            with self.subTest(report_type=type(report).__name__), tempfile.TemporaryDirectory() as d:
                reports = Path(d) / 'reports'
                reports.mkdir()
                (reports / 'eval_1234567890.json').write_text(json.dumps(report), encoding='utf-8')
                (reports / 'eval_1234567890.csv').write_text('placeholder', encoding='utf-8')
                (reports / 'eval_1234567890.md').write_text('placeholder', encoding='utf-8')
                with TestClient(create_app(Config(Path(d)), bootstrap=False)) as client:
                    self.assertEqual(client.get('/api/health').status_code, 200)
                    self.assertEqual(client.get('/api/evaluations').json()['total'], 0)
                self.assertTrue((reports / 'eval_1234567890.json').exists())

    def test_docker_default_configuration_allows_model_settings(self):
        docker = (ROOT / 'Dockerfile').read_text()
        env = {}
        for line in docker.splitlines():
            if line.startswith('ENV '):
                env.update(token.split('=', 1) for token in shlex.split(line[4:]))
        with tempfile.TemporaryDirectory() as d:
            env.update(RAG_CONFIG_DIR=d, RAG_DATA_DIR=str(Path(d) / 'knowledge'))
            with patch.dict(os.environ, env, clear=True):
                cfg = Config.from_env()
                store = SettingsStore(cfg, Path(d) / '.env')
                payload = store.public()
                for kind in ('llm', 'embedding', 'rerank'):
                    payload[kind].pop('has_key')
                payload['llm'].update(enabled=True, url='http://127.0.0.1:9999/v1', model='fixture-model')
                store.save(SettingsInput.model_validate(payload))
                self.assertEqual(Config.from_env().answer_mode, 'llm')

    def test_mcp_protocol_tools_and_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as d, TestClient(create_app(Config(Path(d))), base_url='http://127.0.0.1:8765') as client:
            def rpc(method, params=None):
                response = client.post('/mcp/', json={'jsonrpc': '2.0', 'id': 1,
                    'method': method, **({'params': params} if params is not None else {})},
                    headers={'Accept': 'application/json, text/event-stream'})
                self.assertEqual(response.status_code, 200, response.text)
                return response.json()['result']
            self.assertIn('protocolVersion', rpc('initialize', {'protocolVersion':'2025-03-26',
                'capabilities':{}, 'clientInfo':{'name':'release-audit','version':'1'}}))
            tools = rpc('tools/list')['tools']
            self.assertEqual({t['name'] for t in tools}, {'list_documents','search_documents','read_source'})
            self.assertTrue(all(t['annotations']['readOnlyHint'] for t in tools))
            docs = rpc('tools/call', {'name':'list_documents','arguments':{}})
            self.assertFalse(docs.get('isError'))
            doc_id = docs['structuredContent']['documents'][0]['id']
            search = rpc('tools/call', {'name':'search_documents',
                'arguments':{'query':'RRF 怎样融合 BM25 和向量检索？','top_k':5}})
            self.assertTrue(search['structuredContent']['evidence'])
            source = rpc('tools/call', {'name':'read_source','arguments':{'doc_id':doc_id,'length':200}})
            self.assertTrue(source['structuredContent']['text'])
            self.assertLessEqual(len(source['structuredContent']['text']), 200)
            for name, args in [('search_documents', {'query':'x','top_k':99}),
                               ('read_source', {'doc_id':doc_id,'unit':999}),
                               ('read_source', {'doc_id':'../.env'}),
                               ('delete_documents', {})]:
                self.assertTrue(rpc('tools/call', {'name':name,'arguments':args})['isError'])

    def test_sessions_traces_evidence_survive_restart_and_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(Path(d))
            engine = Engine(cfg)
            try:
                engine.seed(); engine.build()
                sid = engine.sessions.create()['id']
                first = engine.ask('RRF 怎样融合 BM25 和向量检索结果？', session_id=sid)
                follow = engine.ask('它具体怎么算？', session_id=sid)
                self.assertTrue(follow['retrieval_question'].startswith(first['question']))
                turns = engine.sessions.detail(sid)['turns']
                trace_id = first['trace']['id']
                engine.remove(first['evidence'][0]['doc_id'])
                self.assertEqual(engine.sessions.detail(sid)['turns'], turns)
            finally:
                engine.close()
            engine = Engine(cfg)
            try:
                self.assertEqual(engine.sessions.detail(sid)['turns'], turns)
                self.assertEqual(engine.sessions.trace(trace_id)['id'], trace_id)
                engine.sessions.remove(sid)
                self.assertEqual(engine.sessions.list(), [])
                self.assertEqual(engine.db.execute('SELECT COUNT(*) FROM turns').fetchone()[0], 0)
            finally:
                engine.close()

    def test_session_retention_and_context_scope(self):
        with tempfile.TemporaryDirectory() as d:
            engine = Engine(Config(Path(d)))
            try:
                store = engine.sessions
                sid = store.create()['id']
                response = {'question':'retention', 'claims':[{'text':'retained answer'}],
                            'index_fingerprint':'fingerprint', 'evidence':[]}
                for n in range(103):
                    store.append(sid, 1, ['a'], {**response, 'question':str(n)})
                detail = store.detail(sid)
                self.assertEqual(len(detail['turns']), 100)
                self.assertEqual(detail['turns'][0]['question'], '3')
                history, chars = store.context(sid, 1, ['a'], 'fingerprint')
                self.assertEqual(len(history), 3); self.assertLessEqual(chars,1600)
                for revision, docs, fingerprint in [(2,['a'],'fingerprint'),(1,['b'],'fingerprint'),(1,['a'],'other')]:
                    self.assertEqual(store.context(sid,revision,docs,fingerprint)[0], [])
            finally:
                engine.close()

    def test_connection_probes_lists_and_key_destination_protection(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(Path(d), llm_url='http://127.0.0.1:9999/v1', llm_model='fixture',
                         embedding_url='http://127.0.0.1:9999/v1', embedding_model='fixture',
                         rerank_url='http://127.0.0.1:9999/v1', rerank_model='fixture', llm_key='audit-key')
            with patch('rag.settings.post_json', side_effect=[
                {'choices':[{'message':{'content':'{"ok":true}'}}]},
                {'choices':[{'message':{'tool_calls':[{'function':{'name':'connection_probe','arguments':'{"ok":true}'}}]}}]}]):
                self.assertTrue(connection_test(cfg,'llm')['ok'])
            with patch('rag.settings.post_json', return_value={'data':[{'index':0,'embedding':[1]*8}]}):
                self.assertTrue(connection_test(cfg,'embedding')['ok'])
            with patch('rag.settings.post_json', return_value={'results':[{'index':0,'relevance_score':1},{'index':1,'relevance_score':0}]}):
                self.assertTrue(connection_test(cfg,'rerank')['ok'])
            with patch('rag.settings.httpx.Client') as mock:
                response = mock.return_value.__enter__.return_value.get.return_value
                response.status_code = 200
                response.json.return_value = {'data':[{'id':'z'},{'id':'a'},{'id':'a'}]}
                self.assertEqual(available_models(cfg,'llm')['models'], ['a','z'])
            store=SettingsStore(cfg,Path(d)/'.env'); payload=store.public()
            for kind in ('llm','embedding','rerank'): payload[kind].pop('has_key')
            payload['llm']['url']='https://unrelated.example/v1'
            with self.assertRaises(UserError): store.candidate(SettingsInput.model_validate(payload))

    def test_reranking_order_and_failure_fallback(self):
        cfg=Config(Path('/tmp'),rerank_mode='api',rerank_model='fixture')
        candidates=[{'text':'a','id':'E1'},{'text':'b','id':'E2'}]
        with patch('rag.reranking.post_json', return_value={'results':[{'index':0,'relevance_score':.1},{'index':1,'relevance_score':.9}]}):
            ordered, info=rerank(cfg,'query',candidates,1)
            self.assertEqual(ordered[0]['id'],'E2'); self.assertTrue(info['applied'])
        for result in [{'results':[]},{'results':[{'index':0,'relevance_score':float('nan')},{'index':1,'relevance_score':1}]}]:
            with patch('rag.reranking.post_json',return_value=result):
                ordered, info=rerank(cfg,'query',candidates,1)
                self.assertEqual(ordered,candidates[:1]); self.assertFalse(info['applied'])
                self.assertIn('回退',info['warning'])

    def test_llm_agent_loop_and_tool_budget(self):
        with tempfile.TemporaryDirectory() as d:
            engine=Engine(Config(Path(d)))
            try:
                engine.seed(); engine.build()
                engine.config.answer_mode='llm'; engine.config.llm_model='fixture'
                def model(*args):
                    payload=args[3]
                    if 'tools' in payload:
                        observations=[m for m in payload['messages'] if m['role']=='tool']
                        n=len(observations)+1
                        return {'choices':[{'message':{'tool_calls':[{'id':str(n),'type':'function',
                            'function':{'name':'search_documents','arguments':json.dumps({'query':'RRF 怎样融合 BM25 和向量检索？'})}}]}}]}
                    evidence=json.loads(payload['messages'][-1]['content'])['evidence']
                    item=evidence[0]
                    return {'choices':[{'message':{'content':json.dumps({'status':'answered','claims':[
                        {'text':'Fixture claim','evidence':[{'id':item['id'],'quote':item['text'][:30]}]}],'missing':[]})}}]}
                with patch('rag.agent.post_json',side_effect=model), patch('rag.models.post_json',side_effect=model):
                    result=engine.ask('RRF 怎样融合检索结果？',workflow='agent')
                self.assertEqual(result['termination'],'budget_reached')
                self.assertLessEqual(result['agent']['tool_calls'],6)
                self.assertLessEqual(result['agent']['decision_rounds'],4)
                self.assertGreater(result['verified_quotes'],0)
            finally:
                engine.close()

    def test_static_help_error_routes_and_resource_references(self):
        guides=json.loads((ROOT/'web/help-content.json').read_text(encoding='utf-8'))
        for rule in RULES:
            self.assertIn(rule['topic'],guides,rule['code'])
            self.assertIn(rule['page'],['chat','library','experiments','settings'])
            self.assertTrue(rule['steps'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
