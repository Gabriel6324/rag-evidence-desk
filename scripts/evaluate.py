import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import Config, ROOT
from rag.evaluation import run_evaluation, save_report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RAG 参数对照评测，默认完全离线")
    p.add_argument("--sizes", nargs="+", type=int, default=[240, 420, 700])
    p.add_argument("--top-k", nargs="+", type=int, default=[3, 5])
    p.add_argument("--strategies", nargs="+", choices=["vector", "hybrid"], default=["vector", "hybrid"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[.08])
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--no-expand", action="store_true")
    p.add_argument("--live", action="store_true", help="Use configured models; sends sample documents to API and may incur charges")
    p.add_argument("--dataset", type=Path, default=ROOT / "eval/questions.json")
    p.add_argument("--samples", type=Path, default=ROOT / "samples")
    p.add_argument("--out", type=Path, default=ROOT / "reports")
    a = p.parse_args()
    cfg = Config.from_env() if a.live else None
    result = run_evaluation(cfg, a.sizes, a.top_k, a.strategies, a.thresholds, a.overlap,
                            not a.no_expand, a.live, print, a.dataset, a.samples)
    files = save_report(result, a.out)
    print("Saved:", ", ".join(str(a.out / f) for f in files.values()))
