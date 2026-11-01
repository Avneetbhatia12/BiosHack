"""PS5 dashboard: time spent per task step, tool usage, hand utilisation, quality and safety per clip.

    rows, verb_time, tool_use = build_tables(OUT_ROOT, stems)
    write_html(OUT_ROOT, stems)  -> OUT_ROOT/ps5_dashboard.html   (numbers only, no frames: safe to share privately)
"""
from __future__ import annotations

import html
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from .. import results as R
from ..ps3.config import VERBS


def _union_len(intervals: List[Tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        total += cur[1] - cur[0]
    return total


def clip_stats(out_root: str, stem: str) -> Dict:
    p = R.paths(out_root, stem)
    hands = R.load_hands(out_root, stem)
    contacts = R.load_contacts(out_root, stem)
    actions = R.load_actions(out_root, stem)
    m1 = R.read_json(p["metrics"])
    info = R.video_info(out_root, stem)
    dur = info["duration_s"] or (hands[-1]["t"] if hands else max((a["end_s"] for a in actions), default=0.0))
    n = len(hands)
    side = lambda s: sum(any(h.get("side") == s for h in r["hands"]) for r in hands) / n if n else None  # noqa: E731
    row = {"clip": stem, "duration_s": round(dur, 1),
           "hand_coverage": round(sum(bool(r["hands"]) for r in hands) / n, 3) if n else None,
           "both_hands_share": round(sum({"left", "right"} <= {h.get("side") for h in r["hands"]} for r in hands) / n, 3) if n else None,
           "left_visible_share": round(side("left"), 3) if n else None,
           "right_visible_share": round(side("right"), 3) if n else None,
           "ps1_fps": m1.get("end_to_end_fps")}
    row["contacts"] = len(contacts)
    row["portable_contacts"] = sum(c.get("object_type") == "portable" for c in contacts)
    for s in ("left", "right"):
        busy = _union_len([(c["t_start"], c["t_end"]) for c in contacts if c.get("hand") == s])
        row[f"{s}_hand_busy_share"] = round(busy / dur, 3) if dur and contacts else None
    tools = Counter(c["object_name"] for c in contacts if c.get("object_type") == "portable")
    row["top_tools"] = ", ".join(f"{k} x{v}" for k, v in tools.most_common(4))
    verb_t = defaultdict(float)
    for a in actions:
        verb_t[a["verb"]] += a["end_s"] - a["start_s"]
    covered = _union_len([(a["start_s"], a["end_s"]) for a in actions])
    row.update({"actions": len(actions), "action_time_s": round(covered, 1),
                "idle_share": round(1 - covered / dur, 3) if dur and actions else None,
                "mean_action_s": round(sum(a["end_s"] - a["start_s"] for a in actions) / len(actions), 2) if actions else None,
                "ps2_refined_actions": sum(bool(a.get("refined")) for a in actions)})
    q = R.read_json(p["quality"])
    row["quality"] = q.get("verdict", "")
    row["quality_why"] = "; ".join(q.get("why", []))
    saf = R.read_csv(p["safety"])
    row["safety_high"] = sum(s.get("risk") == "HIGH" for s in saf)
    row["safety_medium"] = sum(s.get("risk") == "MEDIUM" for s in saf)
    row["safety_low"] = sum(s.get("risk") == "LOW" for s in saf)
    row["_verb_time"] = {k: round(v, 1) for k, v in verb_t.items()}
    row["_tools"] = dict(tools)
    row["_stages"] = R.stages_done(out_root, stem)
    return row


def build_tables(out_root: str, stems: Sequence[str]):
    rows = [clip_stats(out_root, s) for s in stems]
    verb_time = [{"clip": r["clip"], **{v: r["_verb_time"].get(v, 0.0) for v in VERBS}} for r in rows]
    all_tools = sorted({t for r in rows for t in r["_tools"]})
    tool_use = [{"clip": r["clip"], **{t: r["_tools"].get(t, 0) for t in all_tools}} for r in rows]
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    R.write_csv(os.path.join(out_root, "ps5_dashboard_clips.csv"), flat)
    R.write_csv(os.path.join(out_root, "ps5_time_per_step.csv"), verb_time)
    R.write_csv(os.path.join(out_root, "ps5_tool_usage.csv"), tool_use)
    return rows, verb_time, tool_use


PALETTE = ["#2a78d6", "#e3742f", "#3aa35c", "#9b59b6", "#c9a227", "#d1495b", "#1f9e9e", "#8c6d46", "#5f6b7a",
           "#e377c2", "#6a8d2f", "#b05d00", "#4c4cb3", "#007f5f", "#a23b72", "#7a7a7a", "#2f4b7c", "#999999"]


def write_html(out_root: str, stems: Sequence[str], title: str = "Factory video insights") -> str:
    rows, verb_time, tool_use = build_tables(out_root, stems)
    used_verbs = [v for v in VERBS if any(r[v] for r in verb_time)]
    colour = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(VERBS)}
    esc = html.escape
    pct = lambda x: "–" if x is None else f"{x:.0%}"  # noqa: E731

    def bar(r):
        tot = sum(r[v] for v in used_verbs) or 1
        segs = "".join(f'<span style="width:{100 * r[v] / tot:.2f}%;background:{colour[v]}" title="{esc(v)} {r[v]:.1f}s"></span>'
                       for v in used_verbs if r[v])
        return f'<div class="bar">{segs}</div>'

    legend = "".join(f'<span class="key"><i style="background:{colour[v]}"></i>{esc(v)}</span>' for v in used_verbs)
    trs = []
    for r, vt in zip(rows, verb_time):
        st = r["_stages"]
        stage_txt = " ".join(k.upper() for k in ("ps1", "ps2", "ps3") if st.get(k))
        saf = (f'<b class="hi">{r["safety_high"]} high</b> ' if r["safety_high"] else "") + \
              (f'{r["safety_medium"]} med ' if r["safety_medium"] else "") + (f'{r["safety_low"]} low' if r["safety_low"] else "")
        trs.append(
            f"<tr><td class='clip'>{esc(r['clip'])}<div class='sub'>{esc(stage_txt)}</div></td>"
            f"<td>{r['duration_s']:.0f}s</td><td>{pct(r['hand_coverage'])}</td><td>{pct(r['both_hands_share'])}</td>"
            f"<td>{r['actions']}</td><td>{pct(r['idle_share'])}</td><td class='wide'>{bar(vt) if r['actions'] else '–'}</td>"
            f"<td>{r['contacts']}</td><td>{pct(r['left_hand_busy_share'])} / {pct(r['right_hand_busy_share'])}</td>"
            f"<td class='tools'>{esc(r['top_tools']) or '–'}</td>"
            f"<td><span class='q {esc(r['quality'])}'>{esc(r['quality'] or '–')}</span></td><td>{saf or '–'}</td></tr>")
    tot_tools = Counter()
    for t in tool_use:
        for k, v in t.items():
            if k != "clip":
                tot_tools[k] += v
    mx = max(tot_tools.values(), default=1)
    tool_rows = "".join(f"<div class='trow'><span>{esc(k)}</span><div class='tbar'><i style='width:{100 * v / mx:.1f}%'></i></div>"
                        f"<b>{v}</b></div>" for k, v in tot_tools.most_common(15))
    tot_verb = {v: round(sum(r[v] for r in verb_time), 1) for v in used_verbs}
    mxv = max(tot_verb.values(), default=1) or 1
    verb_rows = "".join(f"<div class='trow'><span>{esc(v)}</span><div class='tbar'><i style='width:{100 * s / mxv:.1f}%;"
                        f"background:{colour[v]}'></i></div><b>{s:.1f}s</b></div>" if s < 10 else
                        f"<div class='trow'><span>{esc(v)}</span><div class='tbar'><i style='width:{100 * s / mxv:.1f}%;"
                        f"background:{colour[v]}'></i></div><b>{s:.0f}s</b></div>"
                        for v, s in sorted(tot_verb.items(), key=lambda x: -x[1]))
    n_act = sum(r["actions"] for r in rows)
    cov = [r["hand_coverage"] for r in rows if r["hand_coverage"] is not None]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#667085;--line:#e3e6ea;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--card:#1b2029;--fg:#e8ebf0;--mut:#98a2b3;--line:#2b3240;--acc:#5aa2f0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:24px 16px 48px}} h1{{font-size:22px;margin:0 0 4px}} .lead{{color:var(--mut);margin:0 0 20px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{display:block;font-size:22px}}
.kpi span{{color:var(--mut);font-size:12px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px;overflow-x:auto}}
h2{{font-size:15px;margin:0 0 10px}} table{{border-collapse:collapse;width:100%;min-width:980px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:middle}}
th{{font-size:12px;color:var(--mut);font-weight:600}} td.clip{{font-weight:600;white-space:nowrap}} .sub{{font-weight:400;color:var(--mut);font-size:11px}}
.bar{{display:flex;height:14px;border-radius:4px;overflow:hidden;background:var(--line);min-width:220px}} .bar span{{height:100%}}
.key{{display:inline-flex;align-items:center;gap:5px;margin:0 12px 6px 0;font-size:12px;color:var(--mut)}} .key i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
.tools{{font-size:12px;color:var(--mut);max-width:220px}} .q{{padding:2px 7px;border-radius:99px;font-size:12px;background:var(--line)}}
.q.keep{{background:#d8f3df;color:#185c2c}} .q.review{{background:#fdf0c8;color:#6b4e00}} .q.drop{{background:#fbd9d9;color:#7a1414}}
.hi{{color:#d1495b}} .grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.trow{{display:grid;grid-template-columns:150px 1fr 50px;gap:8px;align-items:center;margin:5px 0;font-size:13px}} .trow span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.tbar{{height:10px;background:var(--line);border-radius:3px;overflow:hidden}} .tbar i{{display:block;height:100%;background:var(--acc)}}
.note{{color:var(--mut);font-size:12px}}
</style></head><body><main>
<h1>{esc(title)}</h1><p class="lead">Built from PS1 hand tracking, PS2 hand–object contact and PS3 task-step timelines. Numbers only, no frames.</p>
<div class="kpis"><div class="kpi"><b>{len(rows)}</b><span>clips</span></div>
<div class="kpi"><b>{sum(r['duration_s'] for r in rows) / 60:.1f} min</b><span>video analysed</span></div>
<div class="kpi"><b>{(sum(cov) / len(cov)) if cov else 0:.0%}</b><span>frames with a hand (mean)</span></div>
<div class="kpi"><b>{n_act}</b><span>task steps (PS3)</span></div>
<div class="kpi"><b>{sum(r['contacts'] for r in rows)}</b><span>grab→release contacts (PS2)</span></div>
<div class="kpi"><b>{sum(r['safety_high'] for r in rows)}</b><span>high-risk safety stretches</span></div></div>
<div class="card"><h2>Per clip</h2><div>{legend}</div><table><thead><tr><th>clip</th><th>length</th><th>hand coverage</th><th>both hands</th>
<th>steps</th><th>idle</th><th>time per step type</th><th>contacts</th><th>L / R busy</th><th>main tools</th><th>quality</th><th>safety</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<div class="grid2"><div class="card"><h2>Time per step type, all clips</h2>{verb_rows or '<p class="note">PS3 has not run yet.</p>'}</div>
<div class="card"><h2>Tool use (portable objects grabbed), all clips</h2>{tool_rows or '<p class="note">PS2 has not run yet.</p>'}</div></div>
<p class="note">Hand coverage is the share of frames with at least one hand found, not accuracy; accuracy comes from the manual audit.
Busy = share of the clip a hand is in contact with an object. Idle = share of the clip not covered by any PS3 step. Safety stretches are 2D image proximity to hazard objects and need a human check.</p>
</main></body></html>"""
    path = os.path.join(out_root, "ps5_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    with open(os.path.join(out_root, "ps5_dashboard.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items()} for r in rows], f, indent=1, default=str)
    return path
