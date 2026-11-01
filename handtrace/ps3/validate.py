"""Parse the model reply, recover from malformed JSON, and validate every action."""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from .config import HANDS, VERBS, VERB_SYNONYMS, PS3Config


# ------------------------------------------------------------------------------------------- JSON recovery
def parse_model_json(text: str) -> List[dict]:
    """Return the list of raw action dicts from a model reply. Tries strict JSON, then code-fence stripping,
    then salvages complete objects from a truncated or slightly broken reply. Raises ValueError if nothing usable."""
    if text is None or not text.strip():
        raise ValueError("empty reply")
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            data = data.get("actions", [])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    return _salvage_objects(s)


def _salvage_objects(s: str) -> List[dict]:
    """Decode the action objects one by one, keeping those that are complete (handles a cut-off tail)."""
    m = re.search(r'"actions"\s*:\s*\[', s)
    pos = m.end() if m else (s.find("[") + 1 if "[" in s else -1)
    if pos <= 0:
        raise ValueError("no JSON action list found")
    dec, out = json.JSONDecoder(), []
    while pos < len(s):
        while pos < len(s) and s[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(s) or s[pos] != "{":
            break
        try:
            obj, pos = dec.raw_decode(s, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
    if not out and not re.search(r'"actions"\s*:\s*\[\s*\]', s):
        raise ValueError("could not recover any action object")
    return out


# ------------------------------------------------------------------------------------------- field coercion
def to_seconds(value) -> Optional[float]:
    """12, 12.5, "12.5", "12.5s", "00:12", "1:02:03" -> seconds. None if not understood."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower().rstrip("s").strip()
        if re.fullmatch(r"\d+(\.\d+)?", v):
            return float(v)
        parts = v.split(":")
        if 2 <= len(parts) <= 3 and all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            return secs
    return None


def normalise_verb(v) -> Tuple[str, bool]:
    """(verb in the fixed vocabulary, was_unmapped)."""
    s = str(v or "").strip().lower().replace("_", " ").replace("-", " ")
    if s in VERBS:
        return s, False
    if s in VERB_SYNONYMS:
        return VERB_SYNONYMS[s], False
    first = s.split(" ")[0] if s else ""
    for cand in _verb_forms(first):
        if cand in VERBS:
            return cand, False
        if cand in VERB_SYNONYMS:
            return VERB_SYNONYMS[cand], False
    return "other", True


def _verb_forms(word: str) -> List[str]:
    """The word itself plus simple de-inflections: fastening -> fasten, picked -> pick, places -> place."""
    forms = [word]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[: -len(suffix)]
            forms += [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:        # putting -> putt -> put
                forms.append(stem[:-1])
    return forms


def normalise_hand(h) -> str:
    s = str(h or "").strip().lower()
    if s in HANDS:
        return s
    if "both" in s or "two" in s:
        return "both"
    if s.startswith("l"):
        return "left"
    if s.startswith("r"):
        return "right"
    return "unknown"


def normalise_object(o) -> str:
    s = re.sub(r"\s+", " ", str(o or "").strip().lower())
    return s if s and s not in {"none", "null", "n/a", "na", "?"} else "unknown"


# ------------------------------------------------------------------------------------------- time base
def resolve_time_base(spans: List[Tuple[float, float]], win_start: float, mode: str) -> Tuple[float, str]:
    """The model is asked for times from the start of the FULL video. If it answers relative to the window start
    instead, add win_start. 'auto' decides per window: if more than half of the actions begin before the window
    even starts, the reply must be window-relative (in absolute time that is impossible)."""
    if mode == "absolute" or win_start <= 0:
        return 0.0, "absolute"
    if mode == "relative":
        return win_start, "relative"
    if not spans:
        return 0.0, "absolute"
    before = sum(1 for s, _ in spans if s < win_start - 1.0)
    return (win_start, "relative(auto)") if before > len(spans) / 2 else (0.0, "absolute(auto)")


# ------------------------------------------------------------------------------------------- validation
def validate_actions(raw: List[dict], win_start: float, win_end: float, duration: float,
                     cfg: PS3Config, window_index: int = 0) -> Tuple[List[dict], List[dict], dict]:
    """-> (clean actions, dropped actions with a reason, info). Timestamps end up in ORIGINAL video seconds."""
    spans = []
    for d in raw:
        s, e = to_seconds(d.get("start_s")), to_seconds(d.get("end_s"))
        if s is not None and e is not None:
            spans.append((s, e))
    offset, base = resolve_time_base(spans, win_start, cfg.time_base)

    clean, dropped = [], []
    for d in raw:
        s, e = to_seconds(d.get("start_s")), to_seconds(d.get("end_s"))
        if s is None or e is None:
            dropped.append({"reason": "unreadable timestamps", "raw": d})
            continue
        s, e = s + offset, e + offset
        if e <= s:
            dropped.append({"reason": "end <= start", "raw": d})
            continue
        lo, hi = max(0.0, win_start - cfg.boundary_tol_s), min(duration, win_end + cfg.boundary_tol_s)
        if e <= lo or s >= hi:
            dropped.append({"reason": "outside window", "raw": d})
            continue
        s, e = max(s, 0.0), min(e, duration)                    # clamp to the video
        if e - s < cfg.min_action_s:
            dropped.append({"reason": "too short", "raw": d})
            continue
        verb, unmapped = normalise_verb(d.get("verb"))
        try:
            conf = float(d.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(max(conf, 0.0), 1.0)
        if conf < cfg.min_confidence:
            dropped.append({"reason": "low confidence", "raw": d})
            continue
        clean.append({
            "start_s": round(s, 2), "end_s": round(e, 2), "verb": verb,
            "object": normalise_object(d.get("object")), "hand": normalise_hand(d.get("hand")),
            "confidence": round(conf, 3), "windows": [window_index],
            **({"verb_unmapped": str(d.get("verb"))} if unmapped else {}),
        })
    clean.sort(key=lambda a: (a["start_s"], a["end_s"]))
    info = {"time_base": base, "offset_added_s": offset, "raw_count": len(raw), "kept": len(clean), "dropped": len(dropped)}
    return clean, dropped, info
