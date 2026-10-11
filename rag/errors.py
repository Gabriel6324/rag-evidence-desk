"""Structured, redacted failure diagnostics with one guide per known failure type."""
import errno
import json
from pathlib import Path
import re
import sqlite3
import traceback
import uuid

from .config import ROOT, UserError

RULES = json.loads((ROOT / "rag/error-guides.json").read_text(encoding="utf-8"))


def redact(text, secrets=()):
    text = str(text)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "[密钥已遮挡]")
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[密钥已遮挡]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}", "[密钥已遮挡]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|password)=)[^&\s]+", r"\1[已遮挡]", text)
    return text[:6000]


def _guide(code, title, cause, steps, page="settings", topic="test"):
    return dict(code=code, title=title, cause=cause, steps=steps, page=page, topic=topic)


def diagnose(error, stage="处理请求", operation="操作", secrets=()):
    chain, current = [], error
    while current is not None and len(chain) < 8 and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    safe_message = redact(str(error), secrets) if isinstance(error, UserError) else ""
    os_error = next((x for x in chain if isinstance(x, OSError)), None)
    winerror = getattr(os_error, "winerror", None)
    number = getattr(os_error, "errno", None)
    resource = redact(getattr(os_error, "filename", "") or "", secrets)
    type_names = [type(x).__name__ for x in chain]
    guide = None
    status = None
    match = re.search(r"HTTP\s+(\d{3})", safe_message)
    if match:
        status = int(match[1])
    for item in chain:
        response = getattr(item, "response", None)
        if response is not None and getattr(response, "status_code", None):
            status = response.status_code
            break
    if status:
        guides = {
            400: ("API_BAD_REQUEST", "模型接口不接受当前请求", "平台拒绝了模型参数或请求格式。", ["确认选择的模型适合当前接口。", "若平台不支持 JSON 输出参数，在高级选项关闭 JSON 输出约束后测试。", "查看当前平台的兼容接口说明，修正基础地址与模型 ID。"]),
            401: ("API_KEY_INVALID", "API Key 未通过验证", "平台没有接受这次提供的调用密钥。", ["在对应平台 API 控制台确认 Key 仍有效。", "重新复制完整 Key，不加 Bearer、引号或换行；确认 Key 与地址来自同一平台。", "保存前重新测试。"]),
            402: ("API_BALANCE", "账户余额或付费权限不足", "平台要求可用余额或付费权限才能完成调用。", ["在该服务平台查看余额、套餐和当前模型的使用权限。", "处理账户额度后测试；也可先切回离线演示。"]),
            403: ("API_FORBIDDEN", "当前账户没有接口权限", "平台拒绝当前账户、模型或访问来源的权限。", ["在服务平台检查账户状态、模型授权和访问限制。", "确认没有把其他平台的 Key 填在这里。", "选择已授权模型后重新测试。"]),
            404: ("API_NOT_FOUND", "接口地址或模型名不存在", "平台找不到请求的路径或模型。", ["服务地址只填兼容接口基础地址，不加完整请求路径。", "从当前平台复制完整模型 ID；确认该模型仍可用且类型正确。", "重新测试本卡片。"]),
            408: ("API_TIMEOUT", "平台等待超时", "平台没有在允许时间内处理请求。", ["确认网络正常，稍后重试。", "在高级选项适当提高超时；反复超时可换更快的模型或减少输入。"]),
            413: ("API_INPUT_TOO_LARGE", "发给模型的内容太多", "平台拒绝超过输入大小限制的请求。", ["缩短问题；在检索参数中减少证据预算或 top-k。", "检查当前模型的输入限制，或换支持更多输入的模型。"]),
            415: ("API_FORMAT_UNSUPPORTED", "平台不支持当前请求类型", "接口没有接受当前 JSON 请求方式。", ["选择兼容本软件的聊天、Embedding 或 Rerank API。", "复制正确基础地址并按当前模型用途重新测试。"]),
            422: ("API_PARAMETERS", "平台未接受模型参数", "模型类型或参数结构与平台要求不一致。", ["确认模型用途与卡片一致。", "平台不支持 JSON 输出参数时，可关闭该选项后重新测试。"]),
            429: ("API_RATE_LIMIT", "额度不足或请求太频繁", "平台限制了这次调用。仅凭状态码无法区分余额与频率。", ["在平台控制台检查可用额度和模型限额。", "频率限制时等待一段时间后重试，避免连续点击。", "可先使用离线模式；重新运行真实评测会产生新调用。"]),
        }
        if status in guides:
            values = guides[status]
            guide = _guide(values[0], values[1], values[2], values[3])
        elif status >= 500:
            guide = _guide("API_SERVICE_UNAVAILABLE", "模型服务暂时异常", f"服务平台返回 HTTP {status}。", ["查看服务平台的状态或公告，稍后测试。", "反复失败时改用另一可用模型服务，或先切回离线演示。"])
        elif 300 <= status < 400:
            guide = _guide("API_REDIRECT", "接口要求跳转到其他地址", "为避免把调用密钥发到未知地址，软件没有自动跟随跳转。", ["从官方 API 接入文档确认新的直接基础地址。", "核对平台后更新地址并重新填写对应 Key；不要填跳转网址。"])
        else:
            guide = _guide("API_HTTP_ERROR", "模型接口拒绝请求", f"接口返回 HTTP {status}，具体原因尚未确定。", ["按当前卡片检查基础地址和模型 ID。", "复制错误详情，并在服务平台查看这次调用记录。"])
    if guide is None and (winerror in (32, 33) or (isinstance(error, sqlite3.OperationalError) and "locked" in str(error).lower())):
        guide = _guide("FILE_IN_USE", "文件或数据库正在被占用", "操作系统或数据库报告资源已被其他进程占用。", ["关闭多余的知据启动窗口，只保留一个实例。", "如果正在编辑或同步这个文件，等待处理结束；不要删除资料库来解决占用。", "重新打开知据后回到原页面重试。"], "library", "database")
    if guide is None and (isinstance(os_error, PermissionError) or number in (errno.EACCES, errno.EPERM)):
        guide = _guide("FILESYSTEM_PERMISSION", "文件或目录访问被拒绝", "系统没有允许这次读取、写入或清理操作。错误详情显示当前步骤与资源。", ["先查看失败步骤和目标路径；若是上传文件，用原软件确认该文件能打开。", "若是数据或报告目录，确认当前账户能在该目录新建文件；不要把软件解压在受保护的系统目录。", "关闭多余实例，检查目标文件是否只读或正被其他程序使用。", "处理对应路径后重试；仍失败时复制详情。无需先修改模型 Key。"], "library", "database")
    if guide is None and number == errno.ENOSPC:
        guide = _guide("DISK_FULL", "存储空间不足", "系统报告磁盘没有足够可用空间。", ["检查目标数据或报告目录所在磁盘的剩余空间。", "释放空间后重试；保留资料库文件，不要随意删除数据。"], "library", "database")
    if guide is None and any(isinstance(x, FileNotFoundError) for x in chain):
        guide = _guide("FILE_NOT_FOUND", "找不到需要的文件", "当前步骤所需的文件不存在，或已被移动。", ["完整重新解压运行包，保留 runtime、rag、web、samples 和 eval 目录。", "若是自己选择的资料文件，确认文件仍在原位置，再重新选择。", "不要只移动 EXE；重新打开后再操作。"], "library", "app")
    if guide is None and any(isinstance(x, (ImportError, ModuleNotFoundError)) for x in chain):
        guide = _guide("DEPENDENCY_MISSING", "运行组件缺失", "启动或任务需要的 Python 模块没有成功加载。", ["Windows 运行包完整重新解压，保留 runtime 目录。", "源码版在项目目录用相同 Python 安装 requirements.txt。", "确认杀毒或同步工具没有移动／隔离组件；仍失败时复制详情。"], "settings", "app")
    if guide is None and any("Timeout" in name for name in type_names):
        guide = _guide("NETWORK_TIMEOUT", "等待模型服务超时", "网络或服务没有在等待时间内返回。", ["确认电脑能够访问所填服务平台。", "高级选项适当调大超时后保存；减少证据预算或换更快模型。", "连接测试有独立的短等待时间，反复超时先检查地址与网络。"])
    if guide is None and any("JSONDecode" in name for name in type_names):
        code = "EVAL_DATASET" if "评测集" in safe_message or "评测题" in safe_message else "MODEL_JSON"
        guide = next(r.copy() for r in RULES if r["code"] == code)
    if guide is None and any(name in ("ConnectError", "ProxyError", "NetworkError", "ReadError", "RemoteProtocolError") for name in type_names):
        guide = _guide("NETWORK_CONNECTION", "无法连接模型服务", "连接未建立或在传输中断开；可能涉及网络、代理或服务地址。", ["从官方文档核对基础地址，确认该平台网页能访问。", "检查电脑当前的网络与代理设置。", "网络恢复后先测试接口，再重新运行任务。"])
    if guide is None and (safe_message or ""):
        guide = next((r.copy() for r in RULES if any(s in safe_message for s in r["match"])), None)
    if guide is None and any(isinstance(x, sqlite3.DatabaseError) for x in chain):
        guide = _guide("DATABASE_ERROR", "本机数据库操作失败", "数据库未完成当前读写，具体原因需查看诊断信息。", ["关闭重复实例，确认数据目录可写且磁盘有空间。", "保留数据库文件，复制详情用于排查；不要删除原数据。"], "library", "database")
    if guide is None and "already accessed" in str(error).lower():
        guide = _guide("DATABASE_IN_USE", "资料索引已被另一个实例打开", "同一个本机索引不能同时由多个软件实例使用。", ["关闭多余的知据启动窗口。", "只打开一个实例后重试；保留原数据。"], "library", "database")
    if guide is None:
        guide = _guide("UNCLASSIFIED", "这一步未完成", "暂时无法确定具体原因，不能仅据此判断模型配置或数据库有问题。", ["先查看失败步骤，回到对应页面检查当前操作。", "确认启动窗口仍运行，并完整保留运行包。", "复制错误详情用于排查；真实模型任务重试前检查平台调用记录。"], topic="app")
    if not safe_message:
        safe_message = guide["title"] + "。"
    frames = []
    for item in chain:
        frames += [f"{Path(f.filename).name}:{f.lineno} · {f.name}" for f in traceback.extract_tb(item.__traceback__)[-5:]]
    details = "\n".join([
        f"错误编号：{uuid.uuid4().hex}", f"操作：{operation}", f"失败步骤：{stage}",
        f"错误类型：{' → '.join(type_names)}", f"分类：{guide['code']}",
        f"HTTP：{status if status is not None else '无'}", f"系统错误码：{number if number is not None else '无'}",
        f"Windows 错误码：{winerror if winerror is not None else '无'}",
        f"涉及资源：{resource or '错误未提供具体文件／目录'}",
        f"提示：{safe_message}", "代码位置：", *frames,
    ])
    error_id = details.splitlines()[0].split("：", 1)[1]
    return {
        "id": error_id, "code": guide["code"], "title": guide["title"], "message": safe_message,
        "stage": redact(stage, secrets), "operation": operation, "cause": guide["cause"],
        "steps": guide["steps"], "target_page": guide["page"], "guide_topic": guide["topic"],
        "error_type": type_names[0], "resource": resource, "http_status": status,
        "details": redact(details, secrets),
    }


def save_diagnostic(diagnostic, data_dir):
    """A failed log write must not replace the original error."""
    try:
        folder = Path(data_dir) / "diagnostics"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("error_" + diagnostic["id"] + ".json")
        path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        return path.name
    except OSError:
        return None


def request_stage(path):
    operations = {
        "/api/settings/test": "测试模型连接", "/api/settings/models": "获取可用模型列表",
        "/api/settings": "保存并应用模型设置", "/api/documents": "读取并导入资料",
        "/api/build": "建立资料索引", "/api/ask": "资料检索与回答",
        "/api/evaluate": "评测设置检查", "/api/samples": "载入演示资料",
    }
    if path.startswith("/api/evaluations"):
        return "读取评测记录"
    return operations.get(path, "处理 " + path.split("?")[0])

