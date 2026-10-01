"""Optional real-browser QA. Install playwright==1.51.0 and its chromium first."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parent.parent


def main():
    out=ROOT/'docs/screenshots';out.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[]
    with tempfile.TemporaryDirectory() as d:
        env=dict(os.environ,RAG_DATA_DIR=d,RAG_EMBEDDING_MODE='hash',RAG_ANSWER_MODE='extractive',QDRANT_URL='')
        proc=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'--no-browser','--port','8876'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            ready=False
            for _ in range(100):
                try:
                    with socket.create_connection(('127.0.0.1',8876),timeout=.1):ready=True;break
                except OSError:time.sleep(.1)
            if not ready:raise RuntimeError('server did not start')
            with sync_playwright() as p:
                b=p.chromium.launch(headless=True,args=['--no-sandbox'])
                page=b.new_page(viewport={'width':1440,'height':1020})
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
                page.goto('http://127.0.0.1:8876',wait_until='networkidle')
                expect(page.locator('#stat-docs')).to_have_text('5')
                page.screenshot(path=str(out/'01_home.png'),full_page=True);checks.append('desktop initial rendering')
                page.locator('[data-question]').first.click()
                expect(page.locator('.answer-claim').first).to_contain_text('RRF')
                expect(page.locator('.answer-claim').first).to_contain_text('名次')
                page.locator('.citation').first.click()
                expect(page.locator('.evidence-item.highlight')).to_be_visible()
                page.screenshot(path=str(out/'02_answer.png'),full_page=True);checks.append('question, citation highlight, evidence cards')
                page.locator('.evidence-item .text-button').first.click()
                expect(page.locator('#source-dialog')).to_be_visible()
                expect(page.locator('#source-content mark')).to_be_visible()
                page.locator('#close-source').click();checks.append('source dialog and original span highlight')
                with page.expect_download() as download:
                    page.get_by_text('导出回答 .md',exact=True).click()
                assert download.value.suggested_filename.endswith('.md');checks.append('answer markdown export')
                page.locator('[data-page=library]').click()
                page.locator('#file-input').set_input_files({'name':'ui_测试.md','mimeType':'text/markdown','buffer':'# 浏览器回归\n云杉传感器的校准周期为七天。\n<img src=x onerror=window.XSS_EXECUTED=1>'.encode()})
                expect(page.locator('#doc-summary')).to_contain_text('6 份')
                page.locator('#chunk-size').fill('300');page.locator('#chunk-overlap').fill('50');page.locator('#rebuild').click()
                expect(page.locator('#index-pill')).to_have_text('已就绪',timeout=30000)
                assert page.evaluate('window.XSS_EXECUTED') is None;checks.append('document upload, reindex, inert HTML content')
                page.screenshot(path=str(out/'03_library.png'),full_page=True)
                page.locator('[data-page=chat]').click()
                page.locator('#doc-filter').select_option(label='ui_测试.md')
                page.locator('#question').fill('云杉传感器的校准周期是多久？')
                page.locator('#ask-button').click()
                expect(page.locator('.answer-claim').first).to_contain_text('七天',timeout=30000)
                expect(page.locator('.evidence-name')).to_have_count(1)
                assert page.evaluate('window.XSS_EXECUTED') is None;checks.append('document-filtered retrieval on uploaded content')
                page.locator('[data-page=library]').click()
                page.once('dialog',lambda d:d.accept())
                page.locator('.document-row').filter(has_text='ui_测试.md').get_by_text('移除',exact=True).click()
                expect(page.locator('#doc-summary')).to_contain_text('5 份')
                expect(page.locator('#index-pill')).to_have_text('需重建');checks.append('remove document and stale index indication')
                page.locator('#rebuild').click();expect(page.locator('#index-pill')).to_have_text('已就绪',timeout=30000)
                page.locator('[data-page=chat]').click();page.locator('#doc-filter').select_option('')
                page.locator('#question').fill('明天北京天气怎么样？');page.locator('#ask-button').click()
                expect(page.locator('.answer-header h2')).to_have_text('当前资料不足',timeout=30000);checks.append('out-of-corpus abstention')
                page.locator('[data-page=experiments]').click();page.locator('#run-eval').click()
                expect(page.locator('#eval-status')).to_have_text('12 组 · 18 题 / 组',timeout=60000)
                expect(page.locator('.eval-table tbody tr')).to_have_count(12)
                with page.expect_download() as dl:page.get_by_text('下载 JSON',exact=True).click()
                assert dl.value.suggested_filename.endswith('.json')
                page.screenshot(path=str(out/'04_evaluation.png'),full_page=True);checks.append('12 configuration evaluation and result export')
                page.locator('[data-page=settings]').click();expect(page.locator('#connection-info')).to_contain_text('特征哈希')
                page.screenshot(path=str(out/'05_settings.png'),full_page=True);checks.append('model configuration visibility')
                page.set_viewport_size({'width':390,'height':844});page.locator('[data-page=chat]').click()
                page.locator('#clear-chat').click()
                page.locator('#question').blur()
                expect(page.locator('#toast')).to_be_hidden(timeout=8000)
                page.screenshot(path=str(out/'06_mobile.png'),full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'horizontal overflow'
                checks.append('390px responsive layout without horizontal overflow')
                b.close()
            if errors:raise AssertionError(errors)
        finally:
            proc.terminate();server_output=proc.communicate(timeout=15)[0].decode(errors='replace')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'browser':'Chromium 134 / Playwright 1.51.0','checks':checks,'console_errors':errors,'server_output':server_output.strip(),'status':'passed'}
    (ROOT/'reports/browser_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
