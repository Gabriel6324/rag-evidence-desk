"""Document extraction, with source locations retained before chunking."""
from io import BytesIO
from pathlib import Path
import re
import unicodedata
import zipfile

from .config import UserError

MAX_FILE = 10 * 1024 * 1024
MAX_CHARS = 300_000
ALLOWED = {".txt", ".md", ".pdf", ".docx"}


def clean(text):
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def safe_name(name):
    name = str(name).replace("\\", "/").split("/")[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    if not name or len(name) > 160:
        raise UserError("文件名不能为空，且须短于 160 字符。")
    return name


def extract(name: str, content: bytes):
    name = safe_name(name)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise UserError("支持 TXT、Markdown、含文本的 PDF 和 DOCX。")
    if not content or len(content) > MAX_FILE:
        raise UserError("文件为空或超过 10 MB。")
    units = []
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            pdf = PdfReader(BytesIO(content))
            if pdf.is_encrypted and not pdf.decrypt(""):
                raise UserError("请先移除 PDF 密码，再导入。")
            if len(pdf.pages) > 300:
                raise UserError("单份 PDF 最多 300 页，请拆分后导入。")
            for n, page in enumerate(pdf.pages, 1):
                text = clean(page.extract_text() or "")
                if text:
                    units.append({"text": text, "location": f"第 {n} 页", "page": n})
        elif ext == ".docx":
            from docx import Document
            with zipfile.ZipFile(BytesIO(content)) as z:
                if sum(i.file_size for i in z.infolist()) > 30 * 1024 * 1024:
                    raise UserError("DOCX 解压内容过大，请拆分后导入。")
            d = Document(BytesIO(content))
            # Paragraph numbers refer to extracted DOCX paragraphs, not layout pages.
            for n, p in enumerate(d.paragraphs, 1):
                if clean(p.text):
                    units.append({"text": clean(p.text), "location": f"段落 {n}", "page": None})
            for t, table in enumerate(d.tables, 1):
                for r, row in enumerate(table.rows, 1):
                    text = clean(" | ".join(c.text for c in row.cells))
                    if text:
                        units.append({"text": text, "location": f"表 {t} / 行 {r}", "page": None})
        else:
            try:
                text = clean(content.decode("utf-8-sig"))
            except UnicodeDecodeError:
                try:
                    text = clean(content.decode("gb18030"))
                except UnicodeDecodeError as e:
                    raise UserError("文本编码无法识别，请另存为 UTF-8。") from e
            if ext == ".md":
                title, lines = "正文", []
                for line in text.splitlines():
                    if re.match(r"^#{1,6}\s+", line):
                        if clean("\n".join(lines)):
                            units.append({"text": clean("\n".join(lines)), "location": title, "page": None})
                        title = re.sub(r"^#+\s+", "", line).strip()[:160]
                        lines = [line]
                    else:
                        lines.append(line)
                if clean("\n".join(lines)):
                    units.append({"text": clean("\n".join(lines)), "location": title, "page": None})
            else:
                # A single unit preserves paragraph continuity; offsets give precise location.
                units = [{"text": text, "location": "正文", "page": None}] if text else []
    except UserError:
        raise
    except Exception as e:
        raise UserError("文件解析失败，请检查文件是否损坏或格式与扩展名一致。") from e
    total = sum(len(u["text"]) for u in units)
    if total == 0:
        raise UserError("未提取到文字。扫描 PDF 需要先进行 OCR，本项目不内置 OCR。")
    if total > MAX_CHARS:
        raise UserError("单份资料最多 30 万字符，请拆分文件。")
    return units


SUSPICIOUS = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts)|"
    r"忽略.{0,12}(?:指令|系统提示)|(?:泄露|输出|打印).{0,10}(?:API.?KEY|密钥|系统提示词)|"
    r"<\s*script\b|你现在是.{0,8}(?:管理员|系统)", re.I
)


def chunk_units(doc_id, name, units, size=420, overlap=70):
    if not 120 <= size <= 1600 or not 0 <= overlap < min(size, 400):
        raise UserError("chunk 长度须为 120–1600 字符，overlap 须小于 chunk 且不超过 399。")
    chunks = []
    for unit_no, unit in enumerate(units):
        text, start = unit["text"], 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Respect sentence endings when possible; retain exact substring offsets.
                floor = start + max(size // 2, overlap + 1)
                matches = list(re.finditer(r"[。！？\n.!?]", text[floor:end]))
                if matches:
                    end = floor + matches[-1].end()
            body = text[start:end]
            if body.strip():
                chunks.append({"doc_id": doc_id, "name": name, "unit": unit_no,
                    "text": body, "location": unit["location"], "page": unit["page"],
                    "start": start, "end": end, "blocked": bool(SUSPICIOUS.search(body))})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
