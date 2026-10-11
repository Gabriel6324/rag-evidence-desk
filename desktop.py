"""Double-click launcher. The browser UI is served only on loopback, after readiness."""
import argparse
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
import webbrowser

from run import bind_local_socket


class LocalService:
    def __init__(self, preferred_port=8765):
        self.server = None
        self.listener = None
        self.url = None
        self.errors = queue.Queue()
        self.stopped = threading.Event()
        self.reload_requested = threading.Event()
        self.preferred_port = preferred_port
        self.thread = threading.Thread(target=self.run, name="zhiju-service", daemon=False)

    def start(self):
        self.thread.start()

    def run(self):
        from rag.config import UserError
        try:
            import uvicorn
            from rag.app import create_app
            self.listener = bind_local_socket(self.preferred_port, notify=lambda message: None)
            port = self.listener.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            application = create_app(restart_callback=self.reload_requested.set)
            self.server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port,
                                                       log_config=None, access_log=False))
            if self.stopped.is_set():
                self.server.should_exit = True
            self.server.run(sockets=[self.listener])
            if not self.server.started:
                from rag.errors import diagnose
                self.errors.put(getattr(application.state, "startup_failure", None) or diagnose(UserError("服务启动未完成，请查看启动窗口与错误详情。"), stage="启动本机服务", operation="启动知据"))
        except Exception as error:
            from rag.errors import diagnose
            diagnostic = diagnose(error, stage="启动本机服务并打开资料索引", operation="启动知据")
            logging.error("Local service failed: %s", diagnostic["details"])
            self.errors.put(diagnostic)
        finally:
            if self.listener is not None:
                self.listener.close()

    @property
    def ready(self):
        return bool(self.server and self.server.started and self.thread.is_alive())

    def stop(self):
        self.stopped.set()
        if self.server is not None:
            self.server.should_exit = True


def smoke_test(report_path):
    """Run the *packaged executable* against real local HTTP and bundled assets."""
    import httpx
    service = LocalService()
    result = {"frozen": bool(getattr(sys, "frozen", False)), "status": "error", "checks": []}
    service.start()
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and service.thread.is_alive() and not service.ready:
            time.sleep(.1)
        if not service.ready:
            raise RuntimeError("packaged service did not become ready")
        with httpx.Client(base_url=service.url, trust_env=False, timeout=30) as client:
            assert client.get("/api/health").json()["status"] == "ready"
            result["checks"].append("health endpoint and startup")
            assert "id=\"workflow\"" in client.get("/").text
            assert "EventSource" in client.get("/static/app.js").text
            result["checks"].append("bundled web assets")
            status = client.get("/api/status").json()
            assert len(status["documents"]) == 5 and not status["needs_rebuild"]
            result["checks"].append("bundled samples and local Qdrant")
            session = client.post("/api/sessions", json={}).json()
            job = client.post("/api/ask", json={"question": "RRF 怎样融合 BM25 和向量检索结果？",
                                                "workflow": "agent", "session_id": session["id"]}).json()
            while time.monotonic() < deadline:
                state = client.get("/api/jobs/" + job["job_id"]).json()
                if state["state"] != "running":
                    break
                time.sleep(.1)
            assert state["state"] == "done", state
            answer = state["result"]
            assert answer["status"] == "answered" and answer["verified_quotes"] > 0
            assert answer["agent"]["framework"] == "LangGraph"
            assert client.get("/api/sessions/" + session["id"]).json()["turns"]
            result["checks"].append("agent graph, citations and persisted conversation")
            result["status"] = "passed"
    except Exception as error:
        result["error_type"] = type(error).__name__
        logging.exception("Packaged smoke test failed")
    finally:
        service.stop()
        service.thread.join(timeout=35)
        if service.thread.is_alive():
            result["status"] = "error"
            result["error_type"] = "ShutdownTimeout"
        Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description="知据一键启动")
    parser.add_argument("--smoke-test", metavar="REPORT_PATH")
    args = parser.parse_args()
    from rag.config import ROOT, runtime_paths
    settings, data = runtime_paths()
    settings.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(data / "startup.log"), level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    if args.smoke_test:
        return smoke_test(args.smoke_test)
    template = ROOT / ".env.example"
    if not (settings / ".env").exists() and template.exists():
        shutil.copyfile(template, settings / ".env")
    import tkinter as tk
    from tkinter import messagebox, ttk
    window = tk.Tk()
    window.title("知据 · 一键启动")
    window.geometry("630x420")
    window.minsize(520, 380)
    window.configure(bg="#f5f5ef")
    style = ttk.Style(window)
    style.configure("TButton", padding=(12, 7))
    frame = tk.Frame(window, bg="#f5f5ef", padx=28, pady=22)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text="知据 Evidence Desk", font=("Microsoft YaHei UI", 20, "bold"),
             bg="#f5f5ef", fg="#243c35").pack(anchor="w")
    tk.Label(frame, text="资料问答 · 分步核查 · 可追溯引用", font=("Microsoft YaHei UI", 10),
             bg="#f5f5ef", fg="#65746d").pack(anchor="w", pady=(5, 14))
    status = tk.StringVar(value="正在启动，首次运行会建立演示资料索引…")
    tk.Label(frame, textvariable=status, font=("Microsoft YaHei UI", 10), wraplength=470,
             justify="left", bg="#f5f5ef", fg="#243c35").pack(anchor="w", pady=8)
    row = tk.Frame(frame, bg="#f5f5ef")
    row.pack(anchor="w", pady=14)
    service = LocalService()
    opened, closing, restart_requested, open_browser_on_ready = False, False, False, True
    config_dialog = None
    last_error = None
    error_row = ttk.Frame(frame)
    error_text = tk.StringVar()
    ttk.Label(error_row, textvariable=error_text, wraplength=470).pack(anchor="w")
    def error_help():
        from desktop_help import show_failure
        if last_error:
            show_failure(window, last_error)
    ttk.Button(error_row, text="查看原因与解决方法", command=error_help).pack(side="left")
    def dismiss_error():
        nonlocal last_error
        error_row.pack_forget()
        last_error = None
    ttk.Button(error_row, text="×", width=3, command=dismiss_error).pack(side="right")

    def open_workbench():
        try:
            if not webbrowser.open(service.url):
                raise RuntimeError("no browser")
        except Exception:
            messagebox.showinfo("打开工作台", "未能唤起默认浏览器。请复制这个地址打开：\n" + service.url)

    def configure_model():
        nonlocal config_dialog
        if config_dialog is not None and config_dialog.winfo_exists():
            config_dialog.lift()
            return
        from desktop_settings import SettingsDialog
        def saved(result):
            nonlocal restart_requested
            status.set(result["message"])
            if not service.ready:
                restart_requested = True
                service.stop()
        config_dialog = SettingsDialog(window, service, saved)

    launch = ttk.Button(row, text="打开工作台", command=open_workbench, state="disabled")
    launch.pack(side="left", padx=(0, 8))
    ttk.Button(row, text="配置模型接口", command=configure_model).pack(side="left", padx=(0, 8))
    def open_help():
        if service.ready:
            webbrowser.open(service.url + "/?help=1")
        else:
            messagebox.showinfo("使用说明", "1. 完整解压后双击 EXE。\n2. 工作台会自动打开，保留启动窗口。\n3. 没有 Key 也能体验演示。\n4. 在“配置模型接口”填写并测试；保存后自动重启。\n5. 工作台右上角有完整使用说明和常见问题。")
    ttk.Button(frame, text="? 使用说明", command=open_help).place(relx=1, y=0, anchor="ne")
    launch_note = tk.Label(frame, text="启动完成会自动打开工作台。保留此窗口即可持续使用。\n没有 Key 也能体验演示；保存模型配置后自动重启服务，资料与会话会保留。\n也可以进入工作台，在“接口设置”中配置模型；两处入口的设置完全一致。",
             font=("Microsoft YaHei UI", 9), justify="left", anchor="w", wraplength=570,
             bg="#f5f5ef", fg="#65746d")
    launch_note.pack(anchor="w", fill="x")
    frame.bind("<Configure>", lambda event: launch_note.configure(wraplength=max(200, event.width - 56)))

    def poll():
        nonlocal opened, service, restart_requested, open_browser_on_ready, last_error
        if closing:
            if not service.thread.is_alive():
                window.destroy()
                return
            window.after(100, poll)
            return
        if service.reload_requested.is_set() and not restart_requested:
            restart_requested = True
            status.set("设置已保存，正在自动重启服务，资料与会话会保留…")
            service.stop()
        if restart_requested and not service.thread.is_alive():
            restart_requested = False
            port = int(service.url.rsplit(":", 1)[1]) if service.url else 8765
            service = LocalService(preferred_port=port)
            if config_dialog is not None and config_dialog.winfo_exists():
                config_dialog.service = service
            open_browser_on_ready = not opened
            opened = False
            service.start()
        if not service.errors.empty():
            last_error = service.errors.get()
            status.set(last_error["message"])
            error_text.set(last_error["title"] + " · " + last_error["stage"])
            error_row.pack(fill="x", pady=(8, 0))
            launch.configure(state="disabled")
        if service.ready and not opened:
            opened = True
            status.set("工作台已就绪，资料与会话保存在这台电脑。")
            launch.configure(state="normal")
            if open_browser_on_ready:
                open_workbench()
        window.after(100, poll)

    def shutdown():
        nonlocal closing
        closing = True
        status.set("正在关闭服务，请等待当前操作结束…")
        launch.configure(state="disabled")
        service.stop()

    window.protocol("WM_DELETE_WINDOW", shutdown)
    service.start()
    window.after(100, poll)
    window.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        from rag.errors import diagnose
        from desktop_help import show_failure
        import tkinter as tk
        diagnostic = diagnose(error, stage="准备启动配置与数据目录", operation="启动知据")
        root = tk.Tk()
        root.withdraw()
        detail_window = show_failure(root, diagnostic)
        root.wait_window(detail_window)
        root.destroy()
        raise SystemExit(1)
