"""Two clients: a new evaluation arrives while a history page is already open."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
from playwright.sync_api import expect, sync_playwright

ROOT=Path(os.environ.get('RAG_AUDIT_TARGET', str(Path(__file__).resolve().parent.parent))).resolve()
sys.path.insert(0,str(ROOT))
from rag.config import Config
from rag.engine import Engine
from rag.evaluation_history import EvaluationHistory, request_snapshot, report_summary


def main():
    with tempfile.TemporaryDirectory(prefix='zhiju_paging_') as d:
        data=Path(d)/'data'; cfg=Config(data)
        engine=Engine(cfg)
        history=EvaluationHistory(engine.db,engine.lock)
        report=json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'))
        report['runs']=report['runs'][:1]
        for _ in range(42):
            key=uuid.uuid4().hex
            history.create(key,'parameters',request_snapshot(cfg))
            history.finish(key,result=report_summary(report,{}))
        engine.close()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        env=dict(os.environ,RAG_DATA_DIR=str(data),RAG_CONFIG_DIR=str(Path(d)/'settings'),
                 RAG_EMBEDDING_MODE='hash',RAG_ANSWER_MODE='extractive',RAG_RERANK_MODE='none',QDRANT_URL='')
        with (Path(d)/'server.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'--no-browser','--strict-port','--port',str(port)],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                url=f'http://127.0.0.1:{port}'
                with httpx.Client(base_url=url,trust_env=False,timeout=5) as client:
                    for _ in range(150):
                        try:
                            if client.get('/api/health').status_code==200: break
                        except httpx.TransportError: pass
                        if proc.poll() is not None: raise RuntimeError('server failed')
                        time.sleep(.1)
                    else: raise RuntimeError('server not ready')
                    with sync_playwright() as p:
                        browser=p.chromium.launch(executable_path=os.environ.get('RAG_BROWSER_BIN'), headless=True,args=['--no-sandbox'])
                        page=browser.new_page()
                        errors=[]; page.on('pageerror',lambda e:errors.append(str(e)))
                        page.goto(url,wait_until='networkidle')
                        page.locator('#welcome-start').click()
                        page.locator('[data-page=experiments]').click()
                        expect(page.locator('.evaluation-run')).to_have_count(20)
                        started=client.post('/api/evaluate',json={'ablation':True}).json()
                        key=started['job_id']
                        for _ in range(200):
                            job=client.get('/api/jobs/'+key).json()
                            if job['state']!='running': break
                            time.sleep(.05)
                        assert job['state']=='done',job
                        for _ in range(4):
                            if not page.locator('#eval-load-more').is_visible(): break
                            page.locator('#eval-load-more').click()
                            expect(page.locator('#eval-load-more')).to_be_enabled()
                        result={'target':str(ROOT),'loaded':page.locator('.evaluation-run').count(),
                                'total':page.evaluate('state.evaluationTotal'),
                                'load_more_visible':page.locator('#eval-load-more').is_visible(),
                                'latest_external_record_present':page.locator('#evaluation-'+key).count()==1,
                                'page_errors':errors}
                        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
                        browser.close()
                        assert result['loaded']==43 and result['total']==43
                        assert not result['load_more_visible'] and result['latest_external_record_present']
                        assert not errors
            finally:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)


if __name__=='__main__': main()
