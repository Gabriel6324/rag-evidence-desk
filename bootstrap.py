"""Cross-platform first-run setup. Dependencies are installed only when needed."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import venv


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required. Python 3.12 is recommended.")
    root = Path(__file__).resolve().parent
    os.chdir(root)
    env = root / ".venv"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        print("[1/3] Creating a project Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    stamp = env / ".installed-requirements"
    expected = hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()
    if not stamp.exists() or stamp.read_text().strip() != expected:
        print("[2/3] Installing dependencies. First launch needs internet access...", flush=True)
        subprocess.run([str(executable), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(root / "requirements.txt")], check=True)
        stamp.write_text(expected)
    print("[3/3] Starting RAG Evidence Desk (Ctrl+C to stop)", flush=True)
    return subprocess.call([str(executable), str(root / "run.py"), *sys.argv[1:]])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError:
        raise SystemExit("Dependency installation failed. Check internet access, then run this launcher again.")
