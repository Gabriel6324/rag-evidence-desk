"""Real local-browser regression; server and browser share one process namespace.

Install playwright==1.51.0 and run `python -m playwright install chromium`.
Screenshots and the result are internal QA artifacts, outside release packages.
"""
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.config import UserError
from rag.errors import diagnose


def main():
    out = Path(os.environ.get('RAG_UI_QA_OUT', str(ROOT.parent / 'qa-browser'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, console = [], [], []
    report = {'status': 'failed', 'checks': checks, 'page_errors': errors, 'console': console}
    with tempfile.TemporaryDirectory(prefix='zhiju_ui_') as d:
        env = dict(os.environ, RAG_DATA_DIR=str(Path(d) / 'data'), RAG_CONFIG_DIR=str(Path(d) / 'config'),
                   RAG_EMBEDDING_MODE='hash', RAG_ANSWER_MODE='extractive', RAG_RERANK_MODE='none',
                   QDRANT_URL='', QDRANT_API_KEY='', LLM_API_KEY='', EMBEDDING_API_KEY='', RERANK_API_KEY='')
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), '--no-browser', '--strict-port', '--port', str(port)],
                                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if proc.poll() is not None:
                    raise RuntimeError('local service exited before readiness')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError('local service did not start')
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path=os.environ.get('RAG_BROWSER_BIN'), headless=True, args=['--no-sandbox'])
                page = browser.new_page(viewport={'width': 1440, 'height': 1020})
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('console', lambda m: console.append(m.text) if m.type == 'error' else None)
                page.goto(f'http://127.0.0.1:{port}', wait_until='networkidle')
                expect(page.locator('#stat-docs')).to_have_text('5')
                expect(page.locator('#welcome-dialog')).to_be_visible()
                expect(page.locator('#welcome-dialog')).to_contain_text('知据帮你从文档里找答案')
                page.locator('#welcome-start').click()
                expect(page.locator('#delete-session')).to_be_disabled()
                expect(page.locator('#delete-session')).to_have_attribute('title', '请先选择一条已保存的会话。')
                assert page.locator('#delete-session').evaluate('(e)=>getComputedStyle(e).cursor') == 'not-allowed'
                checks.append('disabled session delete uses not-allowed cursor and explains why')
                page.locator('#open-help').click()
                expect(page.locator('#help-dialog')).to_contain_text('知据是做什么的')
                assert 'DeepSeek 开源' not in page.locator('#help-dialog').inner_text()
                assert 'ChatGPT Plus' not in page.locator('#help-dialog').inner_text()
                page.locator('#help-done').click()
                checks.append('app purpose, separate tutorial and removed personal FAQs')
                page.screenshot(path=str(out / '01_home.png'), full_page=True)
                page.locator('[data-question]').first.click()
                expect(page.locator('.answer-claim').first).to_contain_text('RRF', timeout=30000)
                page.locator('.citation').first.click()
                expect(page.locator('.evidence-item.highlight')).to_be_visible()
                page.locator('.evidence-item .text-button').first.click()
                expect(page.locator('#source-content mark')).to_be_visible()
                page.locator('#close-source').click()
                with page.expect_download() as dl:
                    page.get_by_text('导出回答 .md', exact=True).click()
                assert dl.value.suggested_filename.endswith('.md')
                checks.append('question, citations, source highlight and answer export')
                page.locator('#workflow').select_option('agent')
                page.locator('#question').fill('RRF 怎样融合 BM25 和向量检索结果？')
                page.locator('#ask-button').click()
                expect(page.locator('.answer-claim').first).to_contain_text('RRF', timeout=30000)
                assert page.evaluate('state.answer.agent.framework') == 'LangGraph'
                assert page.evaluate('state.answer.trace.steps.length') > 0
                checks.append('LangGraph agent trace and persisted conversation')
                page.locator('[data-page=library]').click()
                page.locator('#file-input').set_input_files({'name': 'ui_测试.md', 'mimeType': 'text/markdown',
                    'buffer': '# 浏览器回归\n云杉传感器的校准周期为七天。\n<img src=x onerror=window.XSS_EXECUTED=1>'.encode()})
                expect(page.locator('#doc-summary')).to_contain_text('6 份', timeout=30000)
                expect(page.locator('#index-pill')).to_have_text('已就绪', timeout=30000)
                page.locator('[data-page=chat]').click()
                page.locator('#workflow').select_option('rag')
                page.locator('#doc-filter').select_option(label='ui_测试.md')
                page.locator('#question').fill('云杉传感器的校准周期是多久？')
                page.locator('#ask-button').click()
                expect(page.locator('.answer-claim').first).to_contain_text('七天', timeout=30000)
                assert page.evaluate('window.XSS_EXECUTED') is None
                checks.append('upload, automatic index, document filtering and inert imported HTML')
                page.locator('[data-page=settings]').click()
                expect(page.locator('#page-settings')).to_contain_text('这个是个人的，我不提供')
                expect(page.locator('#connection-info')).to_contain_text('特征哈希')
                page.locator('.connection-section').evaluate_all('(items)=>items.forEach(e=>e.open=true)')
                for kind in ['llm', 'embedding', 'rerank']:
                    for field in ['url', 'key', 'model', 'enabled', 'show', 'clear', 'models', 'test']:
                        page.locator(f'[data-help={kind}_{field}]').click()
                        expect(page.locator('#field-help-dialog')).to_be_visible()
                        assert len(page.locator('#field-help-content').inner_text()) > 80
                        expect(page.locator('#help-dialog')).to_be_hidden()
                        page.locator('#field-help-done').click()
                page.locator('.model-advanced').evaluate('(e)=>e.open=true')
                for topic in ['model_types', 'advanced', 'timeout', 'json_mode', 'offline', 'save', 'mcp', 'mcp_url', 'copy_mcp', 'database', 'deepseek']:
                    page.locator(f'#page-settings [data-help={topic}]').first.click()
                    expect(page.locator('#field-help-dialog')).to_be_visible()
                    assert len(page.locator('#field-help-content').inner_text()) > 80
                    page.locator('#field-help-done').click()
                checks.append('all 24 model field/action guides and 11 other contextual guides')
                key = page.locator('#model-llm-key')
                expect(key).to_have_attribute('type', 'text')
                expect(key).to_have_attribute('autocomplete', 'off')
                key.fill('qa-only-no-real-key')
                assert page.locator('.api-key-wrap').first.evaluate('(e)=>e.classList.contains("masked-key")')
                assert key.evaluate('(e)=>getComputedStyle(e).color') == 'rgba(0, 0, 0, 0)'
                expect(page.locator('.key-mask').first).to_have_text('•' * 19)
                page.locator('#model-llm-show').click()
                assert not page.locator('.api-key-wrap').first.evaluate('(e)=>e.classList.contains("masked-key")')
                page.locator('#model-llm-show').click()
                key.fill('')
                page.locator('#model-deepseek-preset').click()
                expect(page.locator('#model-llm-url')).to_have_value('https://api.deepseek.com')
                page.locator('#model-demo').click()
                expect(page.locator('#model-feedback')).to_contain_text('设置已保存', timeout=30000)
                expect(page.locator('#model-llm-enabled')).not_to_be_checked()
                checks.append('API credential masking/show, disabled password autofill and offline save')
                page.screenshot(path=str(out / '02_settings.png'), full_page=True)
                permission = diagnose(PermissionError(errno.EACCES, 'denied', 'C:/Users/Demo/AppData/知据/reports'),
                                      stage='保存评测 JSON / CSV / Markdown 报告', operation='离线对照评测')
                failed_id = 'a' * 32
                failed_record = {'id': failed_id, 'number': 99, 'kind': 'parameters', 'state': 'running',
                                 'created_at': datetime.now(timezone.utc).isoformat(), 'message': '开始评测…',
                                 'request': {'live': False, 'parameters': {}, 'models': {}}, 'result': None, 'error': None}
                def mock_failed_evaluation():
                    page.route('**/api/evaluate', lambda route: route.fulfill(json={'job_id': failed_id, 'evaluation': failed_record}))
                    page.route('**/api/evaluations/' + failed_id, lambda route: route.fulfill(json={**failed_record, 'state': 'error', 'message': permission['message'], 'error': permission}))
                    page.route('**/api/jobs/' + failed_id + '/events', lambda route: route.fulfill(content_type='text/event-stream',
                           body='event: failed\ndata: ' + json.dumps({'message': permission['message'], 'error': permission}, ensure_ascii=False) + '\n\n'))
                def unmock_failed_evaluation():
                    for path in ('**/api/evaluate', '**/api/evaluations/' + failed_id, '**/api/jobs/' + failed_id + '/events'):
                        page.unroute(path)
                mock_failed_evaluation()
                page.locator('[data-page=experiments]').click()
                page.locator('#run-eval').click()
                expect(page.locator('#global-alert')).to_contain_text('文件或目录访问被拒绝')
                page.locator('[data-page=chat]').click()
                expect(page.locator('#global-alert')).to_be_visible()
                page.locator('#error-help').click()
                expect(page.locator('#error-content')).to_contain_text('保存评测')
                expect(page.locator('#error-content')).to_contain_text('C:/Users/Demo/AppData/知据/reports')
                expect(page.locator('#error-content')).to_contain_text('无需先修改模型 Key')
                page.screenshot(path=str(out / '03_failure_guide.png'), full_page=True)
                page.locator('#close-error-dialog').click()
                page.locator('#dismiss-error').click()
                page.locator('[data-page=experiments]').click()
                expect(page.locator('#global-alert')).to_be_hidden()
                unmock_failed_evaluation()
                key_error = diagnose(UserError('模型接口 HTTP 401'), stage='测试对话模型', operation='测试模型连接')
                page.route('**/api/settings/test', lambda route: route.fulfill(status=400, json={'detail': key_error['message'], 'error': key_error}))
                page.locator('[data-page=settings]').click()
                page.locator('#model-llm-test').click()
                expect(page.locator('#global-alert')).to_contain_text('API Key 未通过验证')
                page.locator('#error-help').click()
                expect(page.locator('#error-content')).to_contain_text('重新复制完整 Key')
                page.locator('#close-error-dialog').click()
                page.locator('#dismiss-error').click()
                page.unroute('**/api/settings/test')
                checks.append('SSE failure stage/path, corresponding guide, cross-page persistence, X dismissal and distinct new failure')
                page.locator('[data-page=experiments]').click()
                for topic in ['evaluation', 'eval_parameters', 'eval_ablation', 'eval_live', 'eval_history', 'metrics']:
                    page.locator(f'#page-experiments [data-help={topic}]').first.click()
                    expect(page.locator('#field-help-dialog')).to_be_visible()
                    page.locator('#field-help-done').click()
                page.locator('#run-eval').click()
                expect(page.locator('#run-eval')).to_have_attribute('aria-busy', 'true')
                assert page.locator('#run-eval').evaluate('(e)=>getComputedStyle(e).cursor') == 'wait'
                assert page.locator('#run-ablation').evaluate('(e)=>getComputedStyle(e).cursor') == 'not-allowed'
                latest = page.locator('.evaluation-run').first
                expect(latest.locator('.evaluation-progress')).to_have_text('12 组 · 18 题 / 组', timeout=60000)
                expect(latest.locator('.eval-table tbody tr')).to_have_count(12)
                first_id = latest.get_attribute('data-evaluation-id')
                first_file = latest.get_by_text('下载 JSON', exact=True).get_attribute('href')
                with page.expect_download() as dl:
                    latest.get_by_text('下载 JSON', exact=True).click()
                assert dl.value.suggested_filename.endswith('.json')
                page.screenshot(path=str(out / '04_evaluation.png'), full_page=True)
                page.locator('#run-ablation').click()
                expect(latest.locator('.evaluation-progress')).to_have_text('4 组 · 18 题 / 组', timeout=60000)
                expect(latest.locator('.eval-table tbody tr')).to_have_count(4)
                expect(latest.locator('.eval-table tbody tr').last).to_contain_text('100.0%')
                previous = page.locator('#evaluation-' + first_id)
                assert not previous.evaluate('(e)=>e.open')
                previous.locator('summary').click()
                expect(previous.locator('.eval-table tbody tr')).to_have_count(12)
                assert previous.get_by_text('下载 JSON', exact=True).get_attribute('href') == first_file
                assert latest.get_by_text('下载 JSON', exact=True).get_attribute('href') != first_file
                with page.expect_download() as dl:
                    previous.get_by_text('下载 JSON', exact=True).click()
                assert dl.value.suggested_filename in first_file
                page.reload(wait_until='networkidle')
                page.locator('[data-page=experiments]').click()
                expect(page.locator('.evaluation-run')).to_have_count(2)
                expect(latest.locator('.evaluation-progress')).to_have_text('4 组 · 18 题 / 组')
                expect(page.locator('#evaluation-' + first_id)).to_have_attribute('data-state', 'done')
                mock_failed_evaluation()
                page.locator('#run-eval').click()
                expect(latest).to_have_attribute('data-state', 'error')
                expect(page.locator('.evaluation-run')).to_have_count(3)
                expect(page.locator('#evaluation-' + first_id)).to_have_attribute('data-state', 'done')
                page.locator('#dismiss-error').click()
                latest.get_by_text('查看原因与解决方法', exact=True).click()
                expect(page.locator('#error-content')).to_contain_text('保存评测')
                page.locator('#close-error-dialog').click()
                unmock_failed_evaluation()
                page.screenshot(path=str(out / '06_evaluation_history.png'), full_page=True)
                page.locator('#run-live-eval').click()
                expect(page.locator('#toast')).to_contain_text('请先勾选')
                checks.append('12-group and 4-group independent cards, newest first, collapse, separate exports, refresh recovery, per-run failure guide, running/disabled cursors and API consent')
                page.set_viewport_size({'width': 390, 'height': 844})
                for name in ['chat', 'library', 'experiments', 'settings']:
                    page.locator(f'[data-page={name}]').click()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), name + ' horizontal overflow'
                page.locator('[data-help=model_types]').click()
                expect(page.locator('#field-help-dialog')).to_be_visible()
                assert page.locator('#field-help-dialog').evaluate('(e)=>e.getBoundingClientRect().width<=innerWidth')
                page.screenshot(path=str(out / '05_mobile_help.png'), full_page=True)
                checks.append('390px layout and readable contextual modal')
                if errors:
                    raise AssertionError(errors)
                browser.close()
                report['status'] = 'passed'
        except Exception as exc:
            report['failure'] = str(exc)
            raise
        finally:
            proc.terminate()
            try:
                server_output = proc.communicate(timeout=15)[0].decode(errors='replace')
            except subprocess.TimeoutExpired:
                proc.kill()
                server_output = proc.communicate()[0].decode(errors='replace')
            report.update(created_at=datetime.now(timezone.utc).isoformat(), server_output=server_output[-8000:], browser='Chromium / Playwright 1.51.0')
            (out / 'browser_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': checks, 'page_errors': errors}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
