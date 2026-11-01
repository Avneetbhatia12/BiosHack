"""Plain-English search over the event index, e.g.
    "show every time someone uses a drill with the left hand"
    "right hand fasten screw longer than 5 s"      "no hands"      "both hands after 1:30"
    "touching the table"      "unsafe"      "pick before 0:45 in 000060-010"

The parser is deterministic and offline: it maps words to filters using the PS3 verb vocabulary, the object names
that actually occur in the index (plus close spellings / plurals), hand words and simple duration / time phrases.
The parsed filters are always returned, so a user can see exactly why something matched. `parse_with_gemini` is an
optional text-only alternative for free-form questions (same filter format; the rule parser stays the fallback).
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Dict, List, Optional, Sequence

from ..ps3.config import VERBS, VERB_SYNONYMS

STOP = {"show", "me", "every", "all", "time", "times", "when", "someone", "somebody", "worker", "the", "a", "an",
        "with", "using", "uses", "use", "used", "his", "her", "their", "hand", "hands", "is", "are", "was", "of",
        "in", "on", "at", "to", "and", "or", "for", "find", "where", "moments", "moment", "any", "does", "do",
        "clip", "clips", "s", "sec", "secs", "second", "seconds", "than", "longer", "shorter", "more", "less",
        "over", "under", "least", "most", "after", "before", "between", "by", "it", "something", "object",
        "objects", "thing", "things", "frames", "frame", "video", "videos", "that", "which", "what"}
CONTACT_WORDS = {"touch", "touches", "touching", "touched", "contact", "contacts", "grab", "grabs", "grabbing",
                 "grasp", "grasping", "holding", "holds", "held"}
SAFETY_WORDS = {"safety", "unsafe", "danger", "dangerous", "risky", "risk", "hazard", "hazardous"}
QUALITY_WORDS = {"blurry", "blur", "dark", "quality", "unusable"}
CONTACT_COMPATIBLE_VERBS = {"pick", "hold", "carry", "place"}
_NUM = r"(\d+:\d{2}(?::\d{2})?|\d+(?:\.\d+)?)"     # m:ss before plain numbers, or "1:30" parses as 1


def _secs(s: str) -> float:
    if ":" in s:
        v = 0.0
        for p in s.split(":"):
            v = v * 60 + float(p)
        return v
    return float(s)


def _verb_of(word: str) -> Optional[str]:
    w = word.lower()
    if w in VERBS and w != "other":
        return w
    if w in VERB_SYNONYMS:
        return VERB_SYNONYMS[w]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            stem = w[: -len(suf)]
            for cand in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem):
                if cand in VERBS and cand != "other":
                    return cand
                if cand in VERB_SYNONYMS:
                    return VERB_SYNONYMS[cand]
    return None


def _object_tokens(objects: Sequence[str]) -> Dict[str, set]:
    """token -> object names containing it (a query word matches an object by any of its words)."""
    tok: Dict[str, set] = {}
    for o in objects:
        for t in re.split(r"[\s_\-]+", o.lower()):
            if len(t) >= 3:
                tok.setdefault(t, set()).add(o)
    return tok


def parse_query(query: str, objects: Sequence[str] = (), clips: Sequence[str] = ()) -> Dict:
    q = query.lower().strip()
    f: Dict = {"query": query, "kinds": set(), "hands": set(), "verbs": set(), "objects": set(), "clips": set(),
               "min_dur": None, "max_dur": None, "after": None, "before": None, "unrecognised": []}
    # durations and times first (they contain numbers that must not become words)
    for pat, key in ((rf"(?:longer|more|over|at least)\s+(?:than\s+)?{_NUM}\s*(?:s|sec|secs|seconds?)?\b", "min_dur"),
                     (rf"(?:shorter|less|under|at most)\s+(?:than\s+)?{_NUM}\s*(?:s|sec|secs|seconds?)?\b", "max_dur")):
        m = re.search(pat, q)
        if m:
            f[key] = _secs(m.group(1)); q = q.replace(m.group(0), " ")
    m = re.search(rf"between\s+{_NUM}\s*(?:s\b)?\s*and\s+{_NUM}", q)
    if m:
        f["after"], f["before"] = _secs(m.group(1)), _secs(m.group(2)); q = q.replace(m.group(0), " ")
    for pat, key in ((rf"after\s+{_NUM}", "after"), (rf"before\s+{_NUM}", "before")):
        m = re.search(pat, q)
        if m:
            f[key] = _secs(m.group(1)); q = q.replace(m.group(0), " ")
    for c in clips:                                            # clip names (long, contain digits / dashes)
        if c.lower() in q:
            f["clips"].add(c); q = q.replace(c.lower(), " ")
    # phrases
    if re.search(r"\bno hands?\b|\bhands? (?:missing|absent|not visible|out of view)\b", q):
        f["kinds"].add("no_hands"); q = re.sub(r"\bno hands?\b|\bhands? (?:missing|absent|not visible|out of view)\b", " ", q)
    if re.search(r"\bboth hands?\b|\btwo hands?\b|\btwo[- ]handed\b", q):
        f["hands"].add("both"); q = re.sub(r"\bboth hands?\b|\btwo hands?\b|\btwo[- ]handed\b", " ", q)
    if re.search(r"\bvisible\b|\bin view\b", q):
        f["kinds"].add("hands_visible")
    tok = _object_tokens(objects)
    words = re.findall(r"[a-z][a-z0-9_\-]*", q)
    for w in words:
        if w in ("left", "right"):
            f["hands"].add(w); continue
        if w in SAFETY_WORDS:
            f["kinds"].add("safety"); continue
        if w in QUALITY_WORDS:
            f["kinds"].add("quality"); continue
        if w in CONTACT_WORDS:
            f["kinds"].add("contact")
            if w.startswith(("hold", "held")):
                f["verbs"].add("hold")
            continue
        if w in ("visible", "view", "no") or w in STOP:
            continue
        # objects before verbs, so "drill" / "press" as a known object is an object
        cands = [w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w]
        hit = next((c for c in cands if c in tok), None)
        if hit is None:
            close = difflib.get_close_matches(w, list(tok), n=1, cutoff=0.85)
            hit = close[0] if close else None
        if hit is not None:
            f["objects"] |= tok[hit]; continue
        v = _verb_of(w)
        if v:
            f["verbs"].add(v); continue
        f["unrecognised"].append(w)
    return f


def _match(e: dict, f: Dict) -> bool:
    kinds = f["kinds"]
    if f["clips"] and e["clip"] not in f["clips"]:
        return False
    if f["after"] is not None and e["t_end"] < f["after"]:
        return False
    if f["before"] is not None and e["t_start"] > f["before"]:
        return False
    if f["min_dur"] is not None and e["duration_s"] < f["min_dur"]:
        return False
    if f["max_dur"] is not None and e["duration_s"] > f["max_dur"]:
        return False
    k = e["kind"]
    if kinds:
        allowed = set(kinds)
        if "contact" in kinds:
            allowed.add("action")                              # "touching X": the PS3 actions on X are shown too
        if "hands_visible" in kinds:
            allowed.add("both_hands")
        if k not in allowed:
            return False
    elif k not in ("action", "contact"):                       # default: activity; PS1 / PS5 kinds only when asked
        if not (k == "both_hands" and f["hands"] == {"both"} and not f["verbs"] and not f["objects"]):
            return False
    if f["hands"]:
        h = e["hand"]
        if "both" in f["hands"]:
            if not (h == "both" or k == "both_hands"):
                return False
        elif h not in f["hands"] and not (h == "both" and k == "action"):
            return False
    if f["verbs"]:
        if k == "action" and e["verb"] not in f["verbs"]:
            return False
        if k == "contact" and not (f["verbs"] & CONTACT_COMPATIBLE_VERBS):
            return False
        if k not in ("action", "contact"):
            return False
    if f["objects"]:
        if not e["object"] or not any(o == e["object"] or o.replace("_", " ") == e["object"].replace("_", " ")
                                      for o in f["objects"]):
            return False
    return bool(kinds or f["hands"] or f["verbs"] or f["objects"] or f["clips"] or f["after"] is not None
                or f["before"] is not None or f["min_dur"] is not None)


def merge_moments(events: List[dict], gap_s: float = 0.5) -> List[dict]:
    """Overlapping matches in one clip (e.g. the PS2 contact and the PS3 action for the same grab) -> one moment."""
    out: List[dict] = []
    for e in sorted(events, key=lambda x: (x["clip"], x["t_start"])):
        if out and out[-1]["clip"] == e["clip"] and e["t_start"] <= out[-1]["t_end"] + gap_s:
            m = out[-1]
            m["t_end"] = max(m["t_end"], e["t_end"])
            m["sources"] = sorted(set(m["sources"]) | {e["source"]})
            m["what"] = m["what"] if describe(e) in m["what"] else m["what"] + " | " + describe(e)
        else:
            out.append({"clip": e["clip"], "t_start": e["t_start"], "t_end": e["t_end"], "sources": [e["source"]],
                        "what": describe(e)})
    for m in out:
        m["duration_s"] = round(m["t_end"] - m["t_start"], 2)
    return out


def describe(e: dict) -> str:
    k = e["kind"]
    if k == "action":
        return f"{e['verb']} {e['object']} ({e['hand']} {'hands' if e['hand'] == 'both' else 'hand'})"
    if k == "contact":
        return f"{e['hand']} hand touching {e['object']} [{e['object_type']}]"
    if k == "safety":
        return f"{e['hand']} hand near {e['object']} ({e['detail']})"
    if k == "quality":
        return f"poor video: {e['detail']}"
    if k == "no_hands":
        return "no hands visible"
    if k == "both_hands":
        return "both hands visible"
    return f"{e['hand']} hand visible"


def search(events: List[dict], query: str, clips: Sequence[str] = (), filters: Optional[Dict] = None) -> Dict:
    from .index import known_objects
    f = filters or parse_query(query, known_objects(events), clips or sorted({e["clip"] for e in events}))
    hits = [e for e in events if _match(e, f)]
    return {"filters": {k: (sorted(v) if isinstance(v, set) else v) for k, v in f.items()},
            "events": hits, "moments": merge_moments(hits)}


def parse_with_gemini(query: str, objects: Sequence[str], model: str = "gemini-3.6-flash",
                      api_key_env: str = "GEMINI_API_KEY") -> Dict:
    """Optional: a text-only Gemini call turns a free-form question into the same filters. No video is sent."""
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ[api_key_env])
    prompt = (
        "Turn this search request about factory hand-activity video into JSON filters.\n"
        f"Allowed verbs: {', '.join(v for v in VERBS if v != 'other')}.\n"
        f"Known objects (use only these, exactly as written): {', '.join(objects[:200])}.\n"
        "Allowed kinds: action, contact, hands_visible, both_hands, no_hands, safety, quality.\n"
        'Return {"kinds": [], "hands": [], "verbs": [], "objects": [], "min_dur": null, "max_dur": null, '
        '"after": null, "before": null}. Hands are left, right or both. Times are seconds.\n'
        f"Request: {query}")
    resp = client.models.generate_content(model=model, contents=prompt,
                                          config=types.GenerateContentConfig(response_mime_type="application/json"))
    d = json.loads(resp.text)
    f = parse_query("", objects)
    for k in ("kinds", "hands", "verbs", "objects"):
        f[k] = set(d.get(k) or [])
    for k in ("min_dur", "max_dur", "after", "before"):
        f[k] = d.get(k)
    f["query"] = query
    return f
