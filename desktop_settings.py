"""Native model settings. All network work stays off the Tk event loop."""
import queue
import threading
import tkinter as tk
from tkinter import ttk, font as tkfont

import httpx

from rag.config import Config, UserError, runtime_paths
from rag.errors import diagnose
from desktop_help import show_guide, show_failure
from rag.settings import SettingsInput, ProbeInput, SettingsStore, connection_test, available_models


def native_font(parent):
    available = set(tkfont.families(parent))
    family = next((name for name in ("Microsoft YaHei UI", "Noto Sans CJK SC", "PingFang SC") if name in available),
                  tkfont.nametofont("TkDefaultFont").actual("family"))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(family=family)
    return family


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, service, on_saved=lambda result: None):
        super().__init__(parent)
        family = native_font(self)
        self.service, self.on_saved = service, on_saved
        self.results = queue.Queue()
        self.working = False
        self.last_error = None
        self.title("知据 · 模型设置")
        self.geometry("760x730")
        self.minsize(650, 580)
        self.configure(bg="#f5f5ef")
        self.transient(parent)
        self.grab_set()
        settings, _ = runtime_paths()
        self.store = SettingsStore(Config.from_env(validate=False), settings / ".env")
        current = self.store.public()
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="把模型接进来，在这里就能完成。", font=(family, 15, "bold")).pack(anchor="w")
        ttk.Label(outer, text="没有 API Key？直接关闭窗口即可体验离线演示。\n只连接对话模型也能使用；向量检索和重排可以以后再配。",
                  wraplength=660, justify="left").pack(anchor="w", pady=(10, 16))
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.fields, self.action_buttons = {}, []
        def help_button(parent, topic):
            ttk.Button(parent, text="？", width=3, command=lambda: show_guide(self, topic)).pack(side="left", padx=(4, 10))
        self.pages = []
        descriptions = {
            "llm": ("1  对话模型", "写答案：读问题与证据，组织回答。真实 Agent 还需要模型支持工具调用。"),
            "embedding": ("2  向量模型（可选）", "找资料：按文字的意思查找段落。可以不配；更换后需要重建资料索引。"),
            "rerank": ("3  重排模型（可选）", "挑证据：把更有用的候选段落排前面。可以不配，未启用保留原排序。"),
        }
        for kind, (title, explanation) in descriptions.items():
            page = ttk.Frame(self.notebook)
            self.notebook.add(page, text=title)
            canvas = tk.Canvas(page, highlightthickness=0, bg="#f5f5ef", height=400)
            scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            tab = ttk.Frame(canvas, padding=16)
            item = canvas.create_window((0, 0), window=tab, anchor="nw")
            tab.bind("<Configure>", lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")))
            def resize(event, c=canvas, i=item, content=tab):
                c.itemconfigure(i, width=event.width)
                for widget in content.winfo_children():
                    if isinstance(widget, ttk.Label) and widget.cget("wraplength"):
                        widget.configure(wraplength=max(250, event.width - 40))
            canvas.bind("<Configure>", resize)
            self.pages.append(canvas)
            intro_row = ttk.Frame(tab)
            intro_row.pack(fill="x")
            ttk.Button(intro_row, text="这一类模型是什么、怎么获取？", command=lambda k=kind: show_guide(self, k)).pack(side="left")
            ttk.Label(tab, text=explanation, wraplength=620, justify="left").pack(anchor="w", pady=(0, 10))
            fields = {"enabled": tk.BooleanVar(value=current[kind]["enabled"]),
                      "url": tk.StringVar(value=current[kind]["url"]),
                      "model": tk.StringVar(value=current[kind]["model"]),
                      "key": tk.StringVar(), "clear_key": tk.BooleanVar()}
            self.fields[kind] = fields
            if kind == "llm":
                preset_row = ttk.Frame(tab)
                preset_row.pack(fill="x", pady=(0, 7))
                ttk.Button(preset_row, text="填入 DeepSeek 官方地址示例", command=self.deepseek_preset).pack(side="left")
                help_button(preset_row, "deepseek")
            enabled_row = ttk.Frame(tab)
            enabled_row.pack(fill="x", pady=(0, 10))
            ttk.Checkbutton(enabled_row, text="启用这个接口", variable=fields["enabled"]).pack(side="left")
            help_button(enabled_row, kind + "_enabled")
            for field, label, hint in (
                ("url", "服务地址（API Base URL）", "从服务商控制台复制，一般以 /v1 结尾；不要填写完整 /chat/completions 地址。"),
                ("key", "API Key（调用密钥）", "已保存密钥；留空保留原值。" if current[kind]["has_key"] else "从当前服务平台 API 控制台创建并复制；不是登录密码。本机模型可留空。"),
                ("model", "模型名", "填写服务商的模型 ID，或获取列表后选择符合本页用途的模型。"),
            ):
                label_row = ttk.Frame(tab)
                label_row.pack(fill="x")
                ttk.Label(label_row, text=label).pack(side="left")
                help_button(label_row, kind + "_" + field)
                if field == "model":
                    entry = ttk.Combobox(tab, textvariable=fields[field])
                    fields["model_widget"] = entry
                else:
                    entry = ttk.Entry(tab, textvariable=fields[field], show="•" if field == "key" else "")
                entry.pack(fill="x", pady=(3, 2))
                ttk.Label(tab, text=hint, wraplength=620, foreground="#65746d", font=(family, 9)).pack(anchor="w", pady=(0, 8))
                if field == "key":
                    fields["key_widget"] = entry
            key_row = ttk.Frame(tab)
            key_row.pack(fill="x", pady=3)
            show = tk.BooleanVar()
            ttk.Checkbutton(key_row, text="显示本次输入", variable=show,
                            command=lambda f=fields, v=show: f["key_widget"].configure(show="" if v.get() else "•")).pack(side="left")
            help_button(key_row, kind + "_show")
            ttk.Checkbutton(key_row, text="清除已保存 Key", variable=fields["clear_key"]).pack(side="left", padx=8)
            help_button(key_row, kind + "_clear")
            row = ttk.Frame(tab)
            row.pack(fill="x", pady=(8, 0))
            for label, callback in (("获取可用模型", lambda k=kind: self.get_models(k)), ("测试这个连接", lambda k=kind: self.test(k))):
                button = ttk.Button(row, text=label, command=callback)
                button.pack(side="left", padx=(0, 10))
                self.action_buttons.append(button)
                help_button(row, kind + ("_models" if label == "获取可用模型" else "_test"))
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x")
        advanced = ttk.Frame(footer)
        advanced.pack(fill="x", pady=10)
        self.json_mode = tk.BooleanVar(value=current["json_mode"])
        self.timeout = tk.StringVar(value=str(current["timeout"]))
        ttk.Checkbutton(advanced, text="JSON 输出约束（推荐）", variable=self.json_mode).pack(side="left")
        help_button(advanced, "json_mode")
        ttk.Label(advanced, text="超时 / 秒").pack(side="left", padx=(24, 8))
        ttk.Spinbox(advanced, from_=5, to=300, width=6, textvariable=self.timeout).pack(side="left")
        help_button(advanced, "timeout")
        ttk.Label(footer, text="这个是个人的，我不提供。密钥仅保存在这台电脑；测试只发送固定短文本。接入模型后，问题和必要资料片段会发送给你填写的服务，费用由服务商计收。",
                  wraplength=670, justify="left", foreground="#65746d", font=(family, 9)).pack(anchor="w")
        self.status = tk.StringVar(value="建议先测试连接，再保存。测试不会改变当前设置。")
        ttk.Label(footer, textvariable=self.status, wraplength=670, justify="left").pack(anchor="w", pady=10)
        self.failure_row = ttk.Frame(footer)
        self.failure_label = tk.StringVar()
        ttk.Label(self.failure_row, textvariable=self.failure_label, wraplength=440).pack(anchor="w")
        ttk.Button(self.failure_row, text="查看原因与解决方法", command=lambda: show_failure(self, self.last_error)).pack(side="left")
        def dismiss_error():
            self.failure_row.pack_forget()
            self.last_error = None
        ttk.Button(self.failure_row, text="×", width=3, command=dismiss_error).pack(side="right")
        actions = ttk.Frame(footer)
        actions.pack(fill="x")
        save_group = ttk.Frame(actions)
        save_group.pack(side="right")
        save = ttk.Button(save_group, text="保存并应用", command=self.save)
        save.pack(side="left")
        self.action_buttons.append(save)
        help_button(save_group, "save")
        ttk.Button(actions, text="关闭设置", command=self.destroy).pack(side="right", padx=10)
        offline = ttk.Button(actions, text="切回离线演示", command=self.use_offline)
        offline.pack(side="left")
        self.action_buttons.append(offline)
        help_button(actions, "offline")
        # Pack the flexible content last so the save action remains visible on
        # smaller displays; only the long tab contents need scrolling.
        self.notebook.pack_forget()
        self.notebook.pack(fill="both", expand=True)
        def resize_labels(event):
            for widget in outer.winfo_children() + footer.winfo_children():
                if isinstance(widget, ttk.Label) and widget.cget("wraplength"):
                    widget.configure(wraplength=max(250, event.width - 5))
        outer.bind("<Configure>", resize_labels)
        self._poll_id = self.after(80, self.poll)

    def use_offline(self):
        if self.working:
            return
        for fields in self.fields.values():
            fields["enabled"].set(False)
        self.save()

    def payload(self, kind=None):
        result = {k: {f: values[f].get() for f in ("enabled", "url", "model", "key", "clear_key")} for k, values in self.fields.items()}
        try:
            result.update(timeout=int(self.timeout.get()), json_mode=self.json_mode.get())
            if kind:
                result["kind"] = kind
            return (ProbeInput if kind else SettingsInput).model_validate(result).model_dump()
        except Exception:
            raise UserError("请检查超时是否为 5–300 的整数，以及地址、模型名和密钥长度。") from None

    def deepseek_preset(self):
        self.fields["llm"]["url"].set("https://api.deepseek.com")
        self.fields["llm"]["model"].set("deepseek-flash")
        self.fields["llm"]["enabled"].set(True)
        self.status.set("已填入 DeepSeek 官方地址和模型示例。请填写你自己账户的 API Key；该 API 按调用计费。模型名以服务商当前列表为准。")

    def request(self, route, payload):
        if self.service.ready:
            with httpx.Client(base_url=self.service.url, timeout=50, trust_env=False) as client:
                response = client.post("/api/settings" + route, json=payload)
            if response.status_code >= 300:
                body = response.json()
                error = UserError(body.get("detail", "操作未完成"))
                error.diagnostic = body.get("error")
                raise error
            return response.json()
        req = (ProbeInput if route else SettingsInput).model_validate(payload)
        if not route:
            self.store.save(req)
            return {"message": "设置已保存，正在重新启动服务。"}
        candidate = self.store.candidate(req, validate=False)
        return (available_models if route == "/models" else connection_test)(candidate, req.kind)

    def perform(self, route, kind, callback):
        if self.working:
            return
        try:
            payload = self.payload(kind)
        except UserError as error:
            self.present_error(diagnose(error, stage="检查模型设置输入", operation="配置模型"))
            return
        self.working = True
        for button in self.action_buttons:
            button.configure(state="disabled")
        self.status.set("正在处理，请稍候…")
        def worker():
            try:
                self.results.put((callback, self.request(route, payload), None))
            except Exception as error:
                stage = {"/test": "测试模型连接", "/models": "获取模型列表", "": "保存模型设置"}[route]
                secrets = [payload[k]["key"] for k in self.fields]
                diagnostic = getattr(error, "diagnostic", None) or diagnose(error, stage=stage, operation=stage, secrets=secrets)
                self.results.put((None, None, diagnostic))
        threading.Thread(target=worker, name="model-settings", daemon=True).start()

    def test(self, kind):
        self.perform("/test", kind, lambda result: self.status.set("；".join(result["checks"] + result["warnings"]) + f" · {result['duration_ms']} ms"))

    def get_models(self, kind):
        def complete(result):
            self.fields[kind]["model_widget"].configure(values=result["models"])
            self.status.set(f"已获取 {len(result['models'])} 个模型。点击模型名右侧下拉箭头选择；请区分对话 / 向量 / 重排模型。")
        self.perform("/models", kind, complete)

    def save(self):
        def complete(result):
            # Refresh the local recovery path after an HTTP save as well.
            settings, _ = runtime_paths()
            self.store = SettingsStore(Config.from_env(validate=False), settings / ".env")
            for fields in self.fields.values():
                fields["key"].set("")
                fields["clear_key"].set(False)
            self.status.set(result["message"])
            self.on_saved(result)
        self.perform("", None, complete)

    def present_error(self, error):
        self.status.set(error["message"])
        self.last_error = error
        self.failure_label.set(error["title"] + " · " + error["stage"])
        self.failure_row.pack(fill="x", pady=(0, 8))

    def poll(self):
        self._poll_id = None
        if not self.results.empty():
            callback, result, error = self.results.get()
            self.working = False
            for button in self.action_buttons:
                button.configure(state="normal")
            if error:
                self.present_error(error)
            else:
                callback(result)
        self._poll_id = self.after(80, self.poll)

    def destroy(self):
        poll_id = getattr(self, "_poll_id", None)
        if poll_id is not None:
            self.after_cancel(poll_id)
            self._poll_id = None
        super().destroy()
