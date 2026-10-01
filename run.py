"""Start locally; select a free port and open the browser only after startup."""
import argparse
import socket
import sys
import threading
import webbrowser


def bind_local_socket(preferred=8765, strict=False, notify=print):
    """Hold the selected port until Uvicorn takes ownership, avoiding a rebind race."""
    candidates = [preferred] if strict or preferred == 0 else [preferred, 18080, 18081, 18888, 19090, 8501, 0]
    attempts = []
    for port in dict.fromkeys(candidates):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                # Windows: don't share an address with another process.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(128)
            sock.setblocking(False)
            return sock
        except OSError as error:
            if sock is not None:
                sock.close()
            code = getattr(error, "winerror", None) or error.errno
            attempts.append(f"{port}:{code}")
            if not strict and port != 0:
                notify(f"Port {port} is unavailable (error {code}); trying another local port.")
    raise OSError(
        "No accessible local port was found. Tried port:error " + ", ".join(attempts) +
        ". Local socket access may be restricted by this computer's policy."
    )


def announce_when_ready(server, url, stopped, open_browser=True, opener=None, notify=print):
    """Never announce or open a URL for a server that failed during startup."""
    while not stopped.wait(.05):
        if server.started:
            notify(f"\n  RAG Evidence Desk - READY\n  Open: {url}\n  Stop: Ctrl+C\n")
            if open_browser:
                try:
                    (opener or webbrowser.open)(url)
                except Exception:
                    notify("Browser could not be opened automatically. Open the URL above manually.")
            return
        if server.should_exit:
            return


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Please install Python 3.10–3.13.")
    parser = argparse.ArgumentParser(description="RAG 资料问答系统")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of choosing a different port")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.port != 0 and not 1024 <= args.port <= 65535:
        parser.error("port must be 0 (automatic) or between 1024 and 65535")
    try:
        import uvicorn
        from rag.app import create_app
        app = create_app()
    except ImportError as e:
        raise SystemExit("Missing dependency. Run: python -m pip install -r requirements.txt") from e
    log = lambda text: print(text, flush=True)
    try:
        listener = bind_local_socket(args.port, args.strict_port, log)
    except OSError as error:
        raise SystemExit(str(error)) from error
    actual_port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{actual_port}"
    stopped = threading.Event()
    try:
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=actual_port,
                                             log_level="warning", access_log=False))
        watcher = threading.Thread(target=announce_when_ready,
            args=(server, url, stopped, not args.no_browser), kwargs={"notify": log}, daemon=True)
        watcher.start()
        log("Starting local service and preparing the document index...")
        # Pass the SAME socket; checking then closing and rebinding would race.
        server.run(sockets=[listener])
    finally:
        stopped.set()
        listener.close()


if __name__ == "__main__":
    main()
