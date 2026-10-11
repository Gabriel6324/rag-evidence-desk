"""Optional /rerank adapter; provider failures visibly fall back to the original ranking."""
import math
import time

from .config import UserError
from .models import post_json


def rerank(config, question, candidates, top_k):
    if config.rerank_mode != "api":
        raise UserError("请先在接口设置启用并测试重排模型，再启用重排。")
    if not candidates:
        return [], {"applied": False, "candidate_count": 0, "duration_ms": 0}
    started = time.perf_counter()
    try:
        response = post_json(config.rerank_url, "/rerank", config.rerank_key,
                             {"model": config.rerank_model, "query": question,
                              "documents": [e["text"] for e in candidates],
                              "top_n": len(candidates), "return_documents": False}, config.timeout)
        rows = response["results"]
        if not isinstance(rows, list) or len(rows) != len(candidates):
            raise ValueError("incomplete ranking")
        if any(not isinstance(r, dict) or type(r.get("index")) is not int
               or type(r.get("relevance_score")) not in {int, float}
               or not math.isfinite(r["relevance_score"]) for r in rows):
            raise ValueError("invalid scores")
        if sorted(r["index"] for r in rows) != list(range(len(candidates))):
            raise ValueError("invalid indexes")
        rows = sorted(rows, key=lambda r: (-r["relevance_score"], r["index"]))
        ordered = [{**candidates[r["index"]], "rerank_score": float(r["relevance_score"])} for r in rows[:top_k]]
        return ordered, {"applied": True, "candidate_count": len(candidates),
                         "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
    except (UserError, KeyError, TypeError, ValueError, OverflowError) as error:
        reason = str(error) if isinstance(error, UserError) else "重排接口返回结构无效。"
        return candidates[:top_k], {"applied": False, "candidate_count": len(candidates),
                                  "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                                  "warning": reason + " 已回退到原检索排序。"}
