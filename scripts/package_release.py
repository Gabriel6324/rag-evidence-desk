"""Build clean public releases using explicit source and application allowlists."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag import __version__

SOURCE_FILES = ('README.md', 'LICENSE', '.gitignore', '.env.example', '.dockerignore', 'Dockerfile',
                'requirements.txt', 'requirements-lock.txt', 'bootstrap.py', 'run.py', 'desktop.py',
                'desktop_settings.py', 'desktop_help.py', '启动.bat', '启动.command', 'start.sh')
SOURCE_DIRS = ('rag', 'web', 'samples', 'eval', 'scripts', 'tests', '.github')
PUBLIC_DOCS = ('docs/技术与评测说明.md', 'docs/构建与验证.md', 'docs/使用说明.txt',
               'reports/evaluation.json', 'reports/evaluation.csv', 'reports/evaluation.md',
               'reports/ablation.json', 'reports/ablation.csv', 'reports/ablation.md')
APP_FILES = ('run.py', 'desktop.py', 'desktop_settings.py', 'desktop_help.py', '.env.example', 'LICENSE')
APP_DIRS = ('rag', 'web', 'samples', 'eval')
BANNED_PARTS = {'__pycache__', '.git', '.venv', 'venv', 'data', 'diagnostics', 'logs', 'node_modules', 'screenshots'}
PRIVATE_NAMES = {'项目演示与岗位对照.md', '升级测试步骤.md', '升级验证.json', '知据_暂停记录.md'}


def safe_file(path):
    return (not path.is_symlink() and path.is_file() and not set(path.parts) & BANNED_PARTS
            and path.name not in PRIVATE_NAMES and path.suffix.lower() not in {'.pyc', '.pyo', '.log', '.sqlite', '.sqlite3', '.db'}
            and (not path.name.startswith('.env') or path.name == '.env.example'))


def copy_file(source, relative, destination):
    path = source / relative
    if not safe_file(path):
        raise ValueError('Missing or disallowed release file: ' + str(relative))
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def copy_tree(source, relative, destination):
    if not (source / relative).is_dir():
        raise ValueError('Missing release directory: ' + str(relative))
    for path in sorted((source / relative).rglob('*')):
        if safe_file(path):
            copy_file(source, path.relative_to(source), destination)


def archive_tree(root, destination):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent))
    with zipfile.ZipFile(destination) as archive:
        failure = archive.testzip()
        if failure:
            raise ValueError('Archive CRC failed: ' + failure)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f'{destination.name}: {destination.stat().st_size} bytes, SHA256 {digest}')


def main():
    parser = argparse.ArgumentParser(description='Build clean source and optional Windows packages')
    parser.add_argument('--windows-base', type=Path, help='Extracted verified Windows runtime and launcher directory')
    parser.add_argument('--out', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='zhiju_release_') as temp:
        source = Path(temp) / 'rag-evidence-desk'
        source.mkdir()
        for name in SOURCE_FILES + PUBLIC_DOCS:
            copy_file(ROOT, name, source)
        for name in SOURCE_DIRS:
            copy_tree(ROOT, name, source)
        archive_tree(source, out / f'RAG-Evidence-Desk-v{__version__}-Source.zip')
        if args.windows_base:
            base = args.windows_base.resolve()
            for name in ('ZhiJuEvidenceDesk.exe', 'runtime/pythonw.exe', 'runtime/python.exe', 'runtime/DLLs/_tkinter.pyd'):
                if not (base / name).is_file():
                    raise ValueError('Windows base is incomplete: ' + name)
            windows = Path(temp) / 'ZhiJuEvidenceDesk'
            windows.mkdir()
            copy_file(base, 'ZhiJuEvidenceDesk.exe', windows)
            copy_tree(base, 'runtime', windows)
            for name in APP_FILES:
                copy_file(ROOT, name, windows)
            for name in APP_DIRS:
                copy_tree(ROOT, name, windows)
            shutil.copy2(ROOT / 'docs/使用说明.txt', windows / '使用说明.txt')
            for name in PUBLIC_DOCS:
                if name.startswith('reports/'):
                    target = windows / '实验记录' / Path(name).name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(ROOT / name, target)
            archive_tree(windows, out / f'ZhiJuEvidenceDesk-v{__version__}-Windows-x64.zip')


if __name__ == '__main__':
    main()
