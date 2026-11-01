"""PS3 bonus: compare two workers doing the same task, and check one clip against an expected step list.

A step is "verb object" (e.g. "fasten screw"). Steps are aligned with difflib's longest-matching-block algorithm,
so a missing, extra or swapped step shows up as one row instead of shifting everything after it.

    compare_clips(actions_a, actions_b)          -> (rows, summary)  who is faster, which steps differ
    check_sequence(actions, expected_steps)      -> report           missing / out-of-order / extra steps
"""
from __future__ import annotations

import difflib
from typing import Dict, List, Sequence, Tuple

from .merge import same_object
from .validate import normalise_verb

IGNORED_VERBS = {"hold", "carry"}          # supporting actions: excluded from step sequences by default


def _step(a: dict) -> str:
    return f"{a['verb']} {a['object']}".strip()


def step_list(actions: Sequence[dict], ignore=IGNORED_VERBS, collapse_repeats: bool = True) -> List[dict]:
    """Actions -> ordered steps (consecutive identical steps collapsed, durations summed)."""
    out: List[dict] = []
    for a in sorted(actions, key=lambda x: x["start_s"]):
        if a["verb"] in ignore:
            continue
        s = _step(a)
        if collapse_repeats and out and out[-1]["step"] == s:
            out[-1]["end_s"] = a["end_s"]
            out[-1]["duration_s"] = round(out[-1]["duration_s"] + a["end_s"] - a["start_s"], 2)
            out[-1]["count"] += 1
            continue
        out.append({"step": s, "verb": a["verb"], "object": a["object"], "start_s": a["start_s"],
                    "end_s": a["end_s"], "duration_s": round(a["end_s"] - a["start_s"], 2), "count": 1})
    return out


def _key(step: dict) -> str:
    """Comparison key: the verb plus the object's first noun-ish token, so 'small screw' ~ 'screw'."""
    words = [w for w in step["object"].split() if w not in {"the", "a", "an", "small", "large", "metal", "plastic"}]
    return f"{step['verb']} {words[-1] if words else 'unknown'}"


def compare_clips(actions_a: Sequence[dict], actions_b: Sequence[dict], name_a: str = "A",
                  name_b: str = "B") -> Tuple[List[dict], Dict]:
    sa, sb = step_list(actions_a), step_list(actions_b)
    ka, kb = [_key(s) for s in sa], [_key(s) for s in sb]
    sm = difflib.SequenceMatcher(a=ka, b=kb, autojunk=False)
    rows = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(max(i2 - i1, j2 - j1)):
            a = sa[i1 + k] if i1 + k < i2 else None
            b = sb[j1 + k] if j1 + k < j2 else None
            kind = {"equal": "same", "replace": "different", "delete": f"only {name_a}", "insert": f"only {name_b}"}[op]
            rows.append({"match": kind,
                         f"{name_a}_step": a["step"] if a else "", f"{name_a}_start_s": a["start_s"] if a else None,
                         f"{name_a}_duration_s": a["duration_s"] if a else None,
                         f"{name_b}_step": b["step"] if b else "", f"{name_b}_start_s": b["start_s"] if b else None,
                         f"{name_b}_duration_s": b["duration_s"] if b else None,
                         "duration_diff_s": round(b["duration_s"] - a["duration_s"], 2) if a and b else None})
    span = lambda acts: round(max((x["end_s"] for x in acts), default=0) - min((x["start_s"] for x in acts), default=0), 2)
    summary = {f"{name_a}_steps": len(sa), f"{name_b}_steps": len(sb),
               f"{name_a}_task_span_s": span(actions_a), f"{name_b}_task_span_s": span(actions_b),
               "matched_steps": sum(r["match"] == "same" for r in rows),
               "different_steps": sum(r["match"] == "different" for r in rows),
               f"only_{name_a}": sum(r["match"] == f"only {name_a}" for r in rows),
               f"only_{name_b}": sum(r["match"] == f"only {name_b}" for r in rows),
               "sequence_similarity": round(sm.ratio(), 3)}
    sa_t, sb_t = summary[f"{name_a}_task_span_s"], summary[f"{name_b}_task_span_s"]
    summary["faster"] = name_a if sa_t < sb_t else name_b if sb_t < sa_t else "equal"
    return rows, summary


def parse_expected(expected: Sequence[str]) -> List[dict]:
    """["pick screwdriver", "fasten screw", ...] -> step dicts with the verb mapped to the fixed vocabulary."""
    out = []
    for e in expected:
        parts = e.strip().lower().split(maxsplit=1)
        verb, _ = normalise_verb(parts[0] if parts else "other")
        out.append({"step": f"{verb} {parts[1] if len(parts) > 1 else 'unknown'}", "verb": verb,
                    "object": parts[1] if len(parts) > 1 else "unknown"})
    return out


def check_sequence(actions: Sequence[dict], expected_steps: Sequence[str]) -> Dict:
    """Which expected steps are missing, which happen out of order, and which observed steps were not expected."""
    exp = parse_expected(expected_steps)
    obs = step_list(actions)
    # greedy in-order matching: each expected step takes the earliest unused compatible observed step
    first_seen = {}
    for i, e in enumerate(exp):
        for j, o in enumerate(obs):
            if j in first_seen.values():
                continue
            if o["verb"] == e["verb"] and same_object(o["object"], e["object"]):
                first_seen[i] = j
                break
    missing = [exp[i]["step"] for i in range(len(exp)) if i not in first_seen]
    # out of order: matched expected steps whose observed positions are not increasing (outside the LIS)
    seq = [(i, first_seen[i]) for i in sorted(first_seen)]
    pos = [j for _, j in seq]
    lis_idx = _lis(pos)
    keep = {seq[k][0] for k in lis_idx}
    out_of_order = [{"expected_step": exp[i]["step"], "observed_at_s": obs[j]["start_s"]}
                    for i, j in seq if i not in keep]
    used = set(first_seen.values())
    extra = [{"step": o["step"], "start_s": o["start_s"]} for j, o in enumerate(obs) if j not in used]
    return {"expected": len(exp), "observed_steps": len(obs), "found": len(first_seen), "missing": missing,
            "out_of_order": out_of_order, "unexpected": extra,
            "complete_and_in_order": not missing and not out_of_order}


def _lis(xs: List[int]) -> List[int]:
    """Indices of one longest strictly increasing subsequence (O(n log n))."""
    import bisect
    tails, tails_idx, prev = [], [], [-1] * len(xs)
    for i, x in enumerate(xs):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x); tails_idx.append(i)
        else:
            tails[k] = x; tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k else -1
    out, i = [], tails_idx[-1] if tails_idx else -1
    while i != -1:
        out.append(i); i = prev[i]
    return out[::-1]
