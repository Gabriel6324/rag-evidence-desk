"""Native counterparts of the same field and failure guides used by the workbench."""
import tkinter as tk
from tkinter import ttk
import webbrowser
from rag.guides import load_guides


def _window(parent, title, text, copy_text=None, links=()):
    window = tk.Toplevel(parent)
    window.title("知据 · " + title)
    window.geometry("700x620")
    window.minsize(440, 360)
    if parent.state() != "withdrawn":
        window.transient(parent)
    window.grab_set()
    frame = ttk.Frame(window, padding=20)
    frame.pack(fill="both", expand=True)
    heading = ttk.Frame(frame)
    heading.pack(fill="x", pady=(0, 12))
    def close():
        window.destroy()
        if parent.winfo_exists() and isinstance(parent, tk.Toplevel):
            parent.grab_set()
    ttk.Button(heading, text="×", width=3, command=close).pack(side="right")
    title_label = ttk.Label(heading, text=title, font=("Microsoft YaHei UI", 13, "bold"), wraplength=560)
    title_label.pack(side="left", fill="x", expand=True)
    heading.bind("<Configure>", lambda event: title_label.configure(wraplength=max(180, event.width - 64)))
    window.protocol("WM_DELETE_WINDOW", close)
    footer = ttk.Frame(frame)
    footer.pack(side="bottom", fill="x", pady=(12, 0))
    ttk.Button(footer, text="返回继续操作", command=close).pack(side="right")
    if copy_text:
        def copy():
            window.clipboard_clear()
            window.clipboard_append(copy_text)
            copy_button.configure(text="已复制")
        copy_button = ttk.Button(footer, text="复制错误详情", command=copy)
        copy_button.pack(side="left")
    if links:
        link_row = ttk.Frame(frame)
        link_row.pack(side="bottom", fill="x", pady=(8, 0))
        for item in links:
            ttk.Button(link_row, text=item["label"] + " ↗", command=lambda u=item["url"]: webbrowser.open(u)).pack(anchor="w", pady=2)
    content = ttk.Frame(frame)
    content.pack(fill="both", expand=True)
    scrollbar = ttk.Scrollbar(content)
    scrollbar.pack(side="right", fill="y")
    body = tk.Text(content, wrap="word", font=("Microsoft YaHei UI", 10), padx=12, pady=12,
                   relief="flat", bg="#f6f7f1", yscrollcommand=scrollbar.set)
    body.pack(fill="both", expand=True)
    scrollbar.configure(command=body.yview)
    body.insert("1.0", text)
    body.configure(state="disabled")
    return window


def show_guide(parent, topic):
    guide = load_guides()[topic]
    text = guide["intro"] + "\n\n用途与操作步骤\n\n"
    text += "\n\n".join(f"{i}. {s}" for i, s in enumerate(guide["steps"], 1))
    if guide.get("issues"):
        text += "\n\n容易填错的地方\n\n" + "\n\n".join("• " + s for s in guide["issues"])
    if guide.get("note"):
        text += "\n\n" + guide["note"]
    return _window(parent, guide["title"], text, links=guide.get("links", ()))


def show_failure(parent, error):
    text = f"执行操作：{error['operation']}\n失败步骤：{error['stage']}\n涉及资源：{error['resource'] or '错误未提供具体路径'}\n\n"
    text += "已知原因\n\n" + error["cause"] + "\n\n处理步骤\n\n"
    text += "\n\n".join(f"{i}. {s}" for i, s in enumerate(error["steps"], 1))
    text += "\n\n处理后回到原操作重试。模型请求重试可能产生新费用。\n\n技术详情\n\n" + error["details"]
    return _window(parent, error["title"], text, copy_text=error["details"])

