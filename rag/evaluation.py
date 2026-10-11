"""Reproducible small-sample evaluation. No model-as-judge or invented accuracy."""
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
from importlib.metadata import version

from .config import Config, ROOT, UserError
from .engine import Engine


def fraction(n, d):
    return round(n / d, 4) if d else None


def run_evaluation(config=None, sizes=(240, 420, 700), ks=(3, 5), strategies=("vector", "hybrid"),
                   thresholds=(.08,), overlap=60, expand=True, live=False, progress=lambda s: None,
                   dataset=None, samples=None, lexical_rescue=True, answer_guard=True):
    progress("读取评测问题与证据标注…")
    dataset = Path(dataset or ROOT / "eval/questions.json")
    samples = Path(samples or ROOT / "samples")
    try:
        questions = json.loads(dataset.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UserError("评测集必须是 UTF-8 编码的有效 JSON 问题列表。") from error
    if not isinstance(questions, list) or not questions:
        raise UserError("评测集必须为非空的问题列表。")
    ids = set()
    for q in questions:
        if (not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q["id"].strip()
                or q["id"] in ids or not isinstance(q.get("question"), str) or not q["question"].strip()
                or len(q["question"]) > 1500 or not isinstance(q.get("type"), str) or not q["type"].strip()):
            raise UserError("评测题须包含唯一的 id、有效的 question 和 type。")
        ids.add(q["id"])
        evidence, keywords = q.get("evidence", []), q.get("keywords", [])
        if (not isinstance(evidence, list) or any(not isinstance(atom, dict)
                or not isinstance(atom.get("source"), str) or not atom["source"].strip()
                or not isinstance(atom.get("quote"), str) or not atom["quote"].strip() for atom in evidence)):
            raise UserError(f"评测题 {q['id']} 的 evidence 须包含有效的 source 和非空 quote。")
        if not isinstance(keywords, list) or any(not isinstance(word, str) or not word.strip() for word in keywords):
            raise UserError(f"评测题 {q['id']} 的 keywords 须为非空字符串列表。")
    answerable = [q for q in questions if q.get("evidence")]
    if not answerable:
        raise UserError("评测集至少需要一道带证据标注的问题。")
    source_files = sorted(p for p in samples.iterdir() if p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"})
    if not source_files:
        raise UserError("没有找到评测资料。")
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "question_count": len(questions), "answerable_count": len(answerable),
              "python": platform.python_version(), "qdrant_client": version("qdrant-client"),
              "settings": {"overlap": overlap, "expand": expand, "live": live,
                           "lexical_rescue": lexical_rescue, "answer_guard": answer_guard},
              "limitations": ["小规模人工编写数据集，不能代表真实课程总体效果。", "引文匹配不等于语义支持或答案正确。",
                              "答案关键词覆盖仅是诊断项，不是推理正确率。", "演示模式只摘录原文，可能返回相关但无法回答问题的内容。"],
              "runs": []}
    with tempfile.TemporaryDirectory(prefix="rag_eval_") as work:
        base = config or Config(data_dir=Path(work))
        # UI/CLI default eval is isolated: no user data, API requests, or production Qdrant.
        cfg = replace(base, data_dir=Path(work), qdrant_url="", qdrant_key="", qdrant_memory=True)
        if not live:
            cfg = replace(cfg, embedding_mode="hash", answer_mode="extractive")
        result["mode"] = cfg.public()
        progress("初始化独立评测环境（内存向量库）…")
        engine = Engine(cfg)
        try:
            source_units = {}
            for p in source_files:
                progress("导入评测资料：" + p.name)
                doc = engine.ingest(p.name, p.read_bytes())
                source_units[p.name] = engine.source(doc["id"])["units"]
            for q in questions:
                progress("核对评测标注：" + q["id"])
                for atom in q.get("evidence", []):
                    units = source_units.get(atom["source"])
                    if units is None or not any(atom["quote"] in unit["text"] for unit in units):
                        raise UserError(f"评测题 {q['id']} 的证据不存在于来源 {atom['source']} 的提取文本中。")
            for size in sizes:
                progress(f"评测：建立 {size} 字符索引")
                engine.build(size, overlap)
                for k, strategy, threshold in product(ks, strategies, thresholds):
                    progress(f"评测：chunk={size} · top-k={k} · {strategy} · 阈值={threshold}")
                    details, recalls, any_hits, complete, mrr, multi = [], [], 0, 0, [], []
                    quotes, valid_ids, matched_quotes, expected_refusals, refusals = 0, 0, 0, 0, 0
                    latency = []
                    keyword_scores = []
                    for q in questions:
                        progress(f"回答评测题 {q['id']}：chunk={size}，top-k={k}，{strategy}")
                        start = time.perf_counter()
                        response = engine.ask(q["question"], top_k=k, strategy=strategy, threshold=threshold, expand=expand,
                                              lexical_rescue=lexical_rescue, answer_guard=answer_guard)
                        latency.append((time.perf_counter() - start) * 1000)
                        evidence = response["evidence"]
                        expected = q.get("evidence", [])
                        matches = [any(e["name"] == atom["source"] and atom["quote"] in e["text"] for e in evidence) for atom in expected]
                        if expected:
                            recall = sum(matches) / len(expected)
                            recalls.append(recall)
                            any_hits += any(matches)
                            complete += all(matches)
                            if q["type"] == "multi":
                                multi.append(float(all(matches)))
                            relevant_ranks = [rank for rank, e in enumerate(evidence, 1)
                                if any(e["name"] == a["source"] and a["quote"] in e["text"] for a in expected)]
                            mrr.append(1 / min(relevant_ranks) if relevant_ranks else 0)
                        else:
                            expected_refusals += 1
                            refusals += response["status"] == "insufficient"
                        mapping = {e["id"]: e for e in evidence}
                        for claim in response["claims"]:
                            for ref in claim["evidence"]:
                                quotes += 1
                                valid_ids += ref["id"] in mapping
                                matched_quotes += ref["id"] in mapping and ref["quote"] in mapping[ref["id"]]["text"]
                        text = "\n".join(c["text"] for c in response["claims"])
                        keywords = q.get("keywords", [])
                        coverage = fraction(sum(w in text for w in keywords), len(keywords))
                        if keywords:
                            keyword_scores.append(coverage)
                        details.append({"id": q["id"], "type": q["type"], "question": q["question"],
                                        "evidence_recall": fraction(sum(matches), len(matches)), "matches": matches,
                                        "answer_status": response["status"], "keyword_coverage": coverage,
                                        "expected_refusal": not bool(expected), "response": response})
                    n = len(recalls)
                    metrics = {"hit_at_k": fraction(any_hits, n), "evidence_recall": fraction(sum(recalls), n),
                               "all_evidence_at_k": fraction(complete, n), "mrr": fraction(sum(mrr), n),
                               "multi_all_evidence": fraction(sum(multi), len(multi)),
                               "citation_id_valid_rate": fraction(valid_ids, quotes), "quote_match_rate": fraction(matched_quotes, quotes),
                               "quote_count": quotes, "refusal_rate_on_unanswerable": fraction(refusals, expected_refusals),
                               "keyword_coverage": fraction(sum(keyword_scores), len(keyword_scores)),
                               "answered_on_answerable": fraction(sum(not d["expected_refusal"] and d["answer_status"] == "answered" for d in details), n),
                               "refused_answerable_count": sum(not d["expected_refusal"] and d["answer_status"] == "insufficient" for d in details),
                               "mean_latency_ms": round(sum(latency) / len(latency), 1)}
                    result["runs"].append({"size": size, "top_k": k, "strategy": strategy, "threshold": threshold,
                                           "chunks": len(engine.chunks), "metrics": metrics, "details": details})
        finally:
            if sys.exc_info()[0] is None:
                progress("关闭独立评测数据库与模型连接…")
            engine.close()
        progress("清理独立评测临时文件…")
    return result


def run_ablation(config=None, progress=lambda message: None, dataset=None, samples=None):
    """Fixed controls: one variable at a time, then both, on identical data."""
    result = None
    variants = [("原始策略", False, False), ("仅术语补召回", True, False),
                ("仅证据筛选", False, True), ("两项优化", True, True)]
    for label, rescue, guard in variants:
        part = run_evaluation(config=config, sizes=(240,), ks=(3,), strategies=("hybrid",),
                              lexical_rescue=rescue, answer_guard=guard, dataset=dataset, samples=samples,
                              progress=lambda message, name=label: progress(name + " · " + message))
        part["runs"][0].update(variant=label, lexical_rescue=rescue, answer_guard=guard)
        if result is None:
            result = part
            result["settings"] = {"live": False, "ablation": True, "overlap": 60, "expand": True}
        else:
            result["runs"].extend(part["runs"])
    return result


def save_report(result, out_dir, name="evaluation"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{**{k: v for k, v in r.items() if k not in {"metrics", "details"}}, **r["metrics"]} for r in result["runs"]]
    with (out_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# RAG 参数评测记录", "", f"运行时间：{result['created_at']}",
             f"模式：{result['mode']['embedding_model']} / {result['mode']['llm_model']}",
             f"题目：{result['question_count']}，其中有答案题 {result['answerable_count']}。", "",
             "这是小规模合成数据上的实测记录；不等同于真实课程问答准确率。", "",
             "| 策略组 | chunk | top-k | 检索 | 阈值 | 证据召回 | 全证据率 | 多证据完整率 | 无答案题拒答 | 有答案误拒 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        m = r["metrics"]
        fmt = lambda v: f"{v:.1%}" if v is not None else "N/A"
        lines.append(f"|{r.get('variant', '默认')}|{r['size']}|{r['top_k']}|{r['strategy']}|{r['threshold']}|{fmt(m['evidence_recall'])}|{fmt(m['all_evidence_at_k'])}|{fmt(m['multi_all_evidence'])}|{fmt(m['refusal_rate_on_unanswerable'])}|{m['refused_answerable_count']}|")
    lines += ["", "## 指标与边界", "", "证据召回按 source + 连续原文标注匹配；每道题等权。全证据率要求一道题所有必要片段被召回。", "",
              "引用编号有效率和引文匹配率的分母为实际展示的引用数；没有引用时为 null。它们不衡量语义支持。", "",
              "所有展示的回答先经过引用校验，所以引用格式指标可能达到 100%；这不是独立的答案质量评测。", "",
              "演示模式仅摘录，不判断多跳推理是否成立；相关资料可能被误当作有答案。无答案题拒答结果保留这类失败。", "",
              "逐题证据、回答、失败案例和数据哈希见同名 JSON。关键词覆盖率不是答案正确率。", ""]
    (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    return {ext: f"{name}.{ext}" for ext in ("json", "csv", "md")}
