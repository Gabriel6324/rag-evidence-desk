import contextlib
import io
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import ProxyHandler, build_opener

from run import announce_when_ready, bind_local_socket


class StartupTests(unittest.TestCase):
    def test_occupied_port_uses_another_held_socket(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0)); occupied.listen()
            port = occupied.getsockname()[1]
            chosen = bind_local_socket(port, notify=lambda _: None)
            try:
                self.assertNotEqual(chosen.getsockname()[1], port)
                self.assertEqual(chosen.getsockname()[0], '127.0.0.1')
                with socket.socket() as competing, self.assertRaises(OSError):
                    competing.bind(chosen.getsockname())
            finally:
                chosen.close()

    def test_windows_access_denied_falls_back(self):
        denied, available = Mock(), Mock()
        failure = PermissionError(13, 'Access denied')
        failure.winerror = 10013
        denied.bind.side_effect = failure
        logs = []
        with patch('run.socket.socket', side_effect=[denied, available]):
            self.assertIs(bind_local_socket(8765, notify=logs.append), available)
        denied.close.assert_called_once()
        available.bind.assert_called_once_with(('127.0.0.1', 18080))
        self.assertIn('10013', logs[0])

    def test_all_ports_denied_have_actionable_failure(self):
        sockets = []
        def denied(*_):
            s = Mock(); s.bind.side_effect = PermissionError(13, 'Access denied'); sockets.append(s); return s
        with patch('run.socket.socket', side_effect=denied), self.assertRaisesRegex(OSError, 'No accessible local port'):
            bind_local_socket(8765, notify=lambda _: None)
        self.assertTrue(all(s.close.called for s in sockets))

    def test_os_assigned_port_and_strict_port(self):
        chosen = bind_local_socket(0)
        try:
            port = chosen.getsockname()[1]
            self.assertGreater(port, 0)
            chosen.listen()
            with self.assertRaises(OSError):
                bind_local_socket(port, strict=True)
        finally:
            chosen.close()

    def test_browser_waits_for_readiness(self):
        server = SimpleNamespace(started=False, should_exit=False)
        stopped, opened = threading.Event(), threading.Event()
        urls, logs = [], []
        def opener(url): urls.append(url); opened.set()
        thread = threading.Thread(target=announce_when_ready,
            args=(server, 'http://127.0.0.1:18080', stopped), kwargs={'opener': opener, 'notify': logs.append})
        thread.start()
        try:
            self.assertFalse(opened.wait(.15))
            self.assertEqual(logs, [])
            server.started = True
            self.assertTrue(opened.wait(2))
        finally:
            stopped.set(); thread.join(2)
        self.assertEqual(urls, ['http://127.0.0.1:18080'])
        self.assertIn('READY', logs[0])

    def test_startup_failure_never_opens_browser(self):
        server = SimpleNamespace(started=False, should_exit=True)
        opener, log = Mock(), Mock()
        announce_when_ready(server, 'http://127.0.0.1:8765', threading.Event(), opener=opener, notify=log)
        opener.assert_not_called(); log.assert_not_called()

    def test_full_server_reaches_ready_with_preferred_port_occupied(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as data_dir, socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0)); occupied.listen()
            port = occupied.getsockname()[1]
            env = dict(os.environ, RAG_DATA_DIR=data_dir, RAG_EMBEDDING_MODE='hash',
                       RAG_ANSWER_MODE='extractive', QDRANT_URL='')
            child = subprocess.Popen([sys.executable, str(root/'run.py'), '--no-browser', '--port', str(port)],
                cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            lines, output = queue.Queue(), []
            def read_output():
                for line in child.stdout: lines.put(line)
            reader = threading.Thread(target=read_output, daemon=True); reader.start()
            try:
                deadline, url = time.monotonic() + 30, None
                while time.monotonic() < deadline:
                    try: line = lines.get(timeout=.2)
                    except queue.Empty:
                        if child.poll() is not None: break
                        continue
                    output.append(line)
                    match = re.search(r'Open: (http://127\.0\.0\.1:\d+)', line)
                    if match: url = match.group(1); break
                self.assertIsNotNone(url, ''.join(output))
                self.assertNotEqual(url, f'http://127.0.0.1:{port}')
                # Local test request; no network or proxy used.
                opener = build_opener(ProxyHandler({}))
                with opener.open(url + '/api/status', timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'1.0.1', response.read())
                self.assertIn('READY', ''.join(output))
                self.assertIn('trying another local port', ''.join(output))
            finally:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
                reader.join(timeout=2); child.stdout.close()


if __name__ == '__main__': unittest.main(verbosity=2)
