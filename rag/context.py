"""A character budget is explicit: it is not a model tokenizer or a token billing claim."""
import re


def resolve_question(question, history):
    if history and len(question) <= 120 and re.search(r"它|他们|她|该|刚才|上面|上述|这个|那[个么]|这些|那[些人]|\bit\b|\bthat\b", question, re.I):
        return history[-1]["question"][:500] + "\n后续问题：" + question
    return question


def pack_evidence(evidence, budget):
    # Reserve a small, explicit allowance per citation for its id and source metadata.
    packed, used = [], 0
    for item in evidence:
        cost = len(item["text"]) + len(item["name"]) + len(item["location"]) + 80
        if used + cost > budget:
            continue  # Never cut a source sentence or silently invalidate its offsets.
        packed.append(item)
        used += cost
    return packed, {"budget_chars": budget, "used_chars": used, "candidate_count": len(evidence),
                    "packed_count": len(packed), "dropped_count": len(evidence) - len(packed)}
