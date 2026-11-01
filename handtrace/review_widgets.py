"""In-notebook frame inspector (ipywidgets): scrub / play any clip with PS1 + PS2 + PS3 + PS5 drawn on the frame,
jump straight to hard moments, compare raw vs post-processed PS1, and record a manual audit for each frame.

    from handtrace.review_widgets import inspector
    inspector(ORDERED, OUT_ROOT)            # displays the widget

The picture is rendered by the kernel (OpenCV -> JPEG), so it works on Kaggle / Colab / Jupyter without extra
extensions. Shown frames are company footage: clear this cell's output before sharing the notebook.
"""
from __future__ import annotations

import html
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from .review import (DEFAULT_LAYERS, JUMP_KINDS, ClipView, audit_report,
                     sample_audit_frames)

LAYER_LABELS = [("ps1", "PS1 hands"), ("ps2", "PS2 contact"), ("objects_all", "all objects"), ("ps3", "PS3 step"),
                ("ps5", "PS5 safety/quality"), ("strip", "timeline"), ("hud", "frame info")]


def _jpeg(img: np.ndarray, q: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


def _parse_time(s: str) -> Optional[float]:
    s = s.strip()
    try:
        if ":" in s:
            v = 0.0
            for p in s.split(":"):
                v = v * 60 + float(p)
            return v
        return float(s)
    except ValueError:
        return None


def _info_html(view: ClipView, f: int) -> str:
    i = view.info_at(f)
    e = html.escape
    rows = []
    for h in i["hands"]:
        col = "#2a78d6" if h["side"] == "left" else "#e3742f"
        extra = []
        if h["interpolated"]:
            extra.append("interpolated")
        if h["side_corrected"]:
            extra.append(f"side fixed (was {h['side_raw']})")
        if h.get("ps2_state"):
            extra.append(f"PS2 {h['ps2_state']}" + (f": {', '.join(h['ps2_objects'])}" if h.get("ps2_objects") else ""))
        rows.append(f"<div><b style='color:{col}'>{e(str(h['side']).upper())} id{h['id']}</b> score {h['score']:.2f}"
                    f"<br><small>{e(' · '.join(extra))}</small></div>")
    body = "".join(rows) or "<div><i>no hand found</i></div>"
    for title, items, col in (("PS3 step", i["actions"], "#1f7a4d"), ("PS5 safety", i["safety"], "#c0392b"),
                              ("PS5 quality", i["quality"], "#9a6b00")):
        if items:
            body += f"<div style='margin-top:6px;color:{col}'><b>{title}</b><br>" + "<br>".join(e(x) for x in items) + "</div>"
    a = i["audited"]
    if a:
        body += (f"<div style='margin-top:6px;color:#555'><b>Audited</b>: missed {a.get('missed_hands')}, false "
                 f"{a.get('false_hands')}, swaps {a.get('side_swaps')}, bad kp {a.get('bad_keypoints')}</div>")
    return (f"<div style='font:13px system-ui;line-height:1.45'><b>{e(i['clip'])}</b> · frame {f} · {i['time']}"
            f"<hr style='margin:4px 0'>{body}</div>")


def inspector(clips: Sequence[str], out_root: str, width: int = 960, start_clip: Optional[str] = None,
              layers=DEFAULT_LAYERS, audit_samples: int = 100):
    import ipywidgets as W
    from IPython.display import display

    stems = [(os.path.splitext(os.path.basename(c))[0], c) for c in clips]
    st = {"view": None, "busy": False}
    clip_dd = W.Dropdown(options=stems, value=start_clip or stems[0][1], description="Clip", layout=W.Layout(width="420px"))
    slider = W.IntSlider(min=0, max=1, value=0, continuous_update=False, readout=True, description="Frame",
                         layout=W.Layout(width="100%"))
    play = W.Play(min=0, max=1, value=0, interval=120, step=1, description="play")
    W.jslink((play, "value"), (slider, "value"))
    play_step = W.Dropdown(options=[("1 frame", 1), ("2 frames", 2), ("5 frames", 5), ("15 frames", 15)], value=1,
                           layout=W.Layout(width="110px"))
    steps = [W.Button(description=d, layout=W.Layout(width="58px")) for d in ("-1s", "-10", "-1", "+1", "+10", "+1s")]
    time_box = W.Text(placeholder="m:ss.s", layout=W.Layout(width="90px"))
    go_btn = W.Button(description="go", layout=W.Layout(width="44px"))
    jump_dd = W.Dropdown(options=[(v, k) for k, v in JUMP_KINDS.items()], value="ps1_low_score",
                         layout=W.Layout(width="380px"))
    prev_btn, next_btn = W.Button(description="◀ prev", layout=W.Layout(width="80px")), W.Button(
        description="next ▶", layout=W.Layout(width="80px"))
    checks = {k: W.Checkbox(value=k in layers, description=lab, indent=False, layout=W.Layout(width="auto"))
              for k, lab in LAYER_LABELS}
    raw_cb = W.Checkbox(value=False, description="raw PS1 side by side", indent=False, layout=W.Layout(width="auto"))
    img_w = W.Image(format="jpeg", layout=W.Layout(width=f"{width}px"))
    info = W.HTML(layout=W.Layout(width="320px"))
    msg = W.HTML()
    # audit
    num = lambda d: W.BoundedIntText(value=0, min=0, max=6, description=d, style={"description_width": "150px"},  # noqa: E731
                                     layout=W.Layout(width="230px"))
    a_pred = W.IntText(value=0, description="predicted hands", disabled=True, style={"description_width": "150px"},
                       layout=W.Layout(width="230px"))
    a_miss, a_false, a_swap, a_kp = num("missed (visible, not found)"), num("false (not a wearer hand)"), \
        num("left/right swapped"), num("keypoints clearly off")
    tog = lambda d: W.ToggleButtons(options=["ok", "wrong", "na"], value="na", description=d,  # noqa: E731
                                    style={"description_width": "80px", "button_width": "52px"})
    a_ps2, a_ps3, a_saf = tog("PS2 contact"), tog("PS3 step"), tog("safety")
    a_note = W.Text(placeholder="note", layout=W.Layout(width="300px"))
    a_who = W.Text(placeholder="reviewer", layout=W.Layout(width="120px"))
    n_samp = W.BoundedIntText(value=audit_samples, min=10, max=1000, description="sample", layout=W.Layout(width="150px"))
    b_sample = W.Button(description="Draw audit sample", button_style="info")
    b_ok = W.Button(description="All correct + next", button_style="success")
    b_save = W.Button(description="Save frame")
    b_next = W.Button(description="Next sample")
    b_del = W.Button(description="Delete row")
    b_rep = W.Button(description="Accuracy report", button_style="primary")
    a_out = W.Output()

    def view() -> ClipView:
        return st["view"]

    def cur_layers():
        return {k for k, c in checks.items() if c.value}

    def render(*_):
        v = view()
        if v is None or st["busy"]:
            return
        st["busy"] = True
        try:
            f = slider.value
            img = v.compose(f, cur_layers(), width=width if not raw_cb.value else width // 2)
            if img is None:
                msg.value = f"<span style='color:#c0392b'>could not read frame {f}</span>"
                return
            if raw_cb.value:
                left = v.compose(f, {"ps1", "hud"}, raw=True, width=width // 2)
                pad = np.zeros((img.shape[0] - left.shape[0], left.shape[1], 3), np.uint8) if img.shape[0] > left.shape[0] else None
                left = np.concatenate([left, pad], 0) if pad is not None else left[: img.shape[0]]
                img = np.concatenate([left, img], 1)
            img_w.value = _jpeg(img)
            info.value = _info_html(v, f)
            hands = v.hands.get(f, {"hands": []})["hands"]
            a = v.audit.get(f)
            a_pred.value = len(hands)
            for w_, k in ((a_miss, "missed_hands"), (a_false, "false_hands"), (a_swap, "side_swaps"), (a_kp, "bad_keypoints")):
                w_.value = int(float(a[k])) if a and str(a.get(k, "")).strip() else 0
            for w_, k in ((a_ps2, "ps2_contact"), (a_ps3, "ps3_action"), (a_saf, "safety_flag")):
                w_.value = a[k] if a and a.get(k) in ("ok", "wrong", "na") else "na"
            a_note.value = (a or {}).get("note", "") or ""
            left_n = len(v.targets("audit_sample"))
            msg.value = (f"<small>{len(v.audit.rows)} frame(s) audited for this clip"
                         + (f" · {left_n} sampled frame(s) left" if v.samples else "") + (" · <b>this frame saved</b>" if a else "")
                         + "</small>")
        finally:
            st["busy"] = False

    def load_clip(*_):
        if view() is not None:
            view().close()
        v = ClipView(clip_dd.value, out_root)
        st["view"] = v
        n = max(1, v.n_frames)
        slider.max = n - 1; play.max = n - 1
        slider.value = 0
        counts = v.target_counts()
        jump_dd.options = [(f"{JUMP_KINDS[k]} ({counts.get(k, 0)})", k) for k in JUMP_KINDS]
        render()

    def step(delta, seconds=False):
        def h(_):
            v = view()
            d = int(round(delta * v.fps)) if (v and seconds) else delta
            slider.value = int(min(max(0, slider.value + d), slider.max)) if v else 0
        return h

    def jump(direction):
        def h(_):
            v = view()
            f = v.next_target(jump_dd.value, slider.value, direction) if v else None
            if f is None:
                msg.value = "<small>no further frame of this kind</small>"
            else:
                slider.value = int(f)
        return h

    def go(_):
        t = _parse_time(time_box.value)
        if t is not None and view():
            slider.value = int(min(slider.max, max(0, view().f_of(t))))

    def save(_=None):
        v = view()
        f = slider.value
        k = len(v.audit.rows)
        v.audit.save({"frame": f, "t": round(v.t_of(f), 2), "split": "dev" if k % 2 else "test",
                      "pred_hands": a_pred.value, "missed_hands": a_miss.value, "false_hands": a_false.value,
                      "side_swaps": a_swap.value, "bad_keypoints": a_kp.value, "ps2_contact": a_ps2.value,
                      "ps3_action": a_ps3.value, "safety_flag": a_saf.value, "note": a_note.value, "reviewer": a_who.value})
        render()

    def next_sample(_=None):
        v = view()
        f = v.next_target("audit_sample", slider.value, 1)
        if f is None:
            ts = v.targets("audit_sample")
            f = ts[0] if ts else None
        if f is None:
            msg.value = "<small>no sampled frames left: draw a sample or open the report</small>"
        else:
            slider.value = int(f)

    def all_ok(_):
        for w_ in (a_miss, a_false, a_swap, a_kp):
            w_.value = 0
        save(); next_sample()

    def draw_sample(_):
        v = view()
        fr = sample_audit_frames(v, n_samp.value)
        msg.value = f"<small>{len(fr)} frames sampled (hard moments + random); press <b>Next sample</b></small>"
        next_sample()

    def delete(_):
        view().audit.delete(slider.value); render()

    def report(_):
        import pandas as pd
        with a_out:
            a_out.clear_output()
            rep = audit_report(out_root, [s for s, _ in stems])
            if not rep["total"]:
                print("no audited frames yet"); return
            display(pd.DataFrame([rep["total"]], index=["all clips"]).T.rename(columns={0: "value"}))
            if len(rep["clips"]) > 1:
                display(pd.DataFrame(rep["clips"]).set_index("clip"))

    slider.observe(render, names="value")
    clip_dd.observe(load_clip, names="value")
    play_step.observe(lambda ch: setattr(play, "step", ch["new"]), names="value")
    for c in list(checks.values()) + [raw_cb]:
        c.observe(render, names="value")
    for b, (d, sec) in zip(steps, ((-1, True), (-10, False), (-1, False), (1, False), (10, False), (1, True))):
        b.on_click(step(d, sec))
    go_btn.on_click(go); prev_btn.on_click(jump(-1)); next_btn.on_click(jump(1))
    b_save.on_click(save); b_next.on_click(next_sample); b_ok.on_click(all_ok); b_sample.on_click(draw_sample)
    b_del.on_click(delete); b_rep.on_click(report)

    controls = W.VBox([
        W.HBox([clip_dd, play, play_step, time_box, go_btn]),
        slider,
        W.HBox(steps + [W.HTML("&nbsp;"), jump_dd, prev_btn, next_btn]),
        W.HBox(list(checks.values()) + [raw_cb]),
    ])
    audit_box = W.VBox([W.HTML("<b>Manual audit</b> <small>(count only the worker's own visible hands)</small>"),
                        a_pred, a_miss, a_false, a_swap, a_kp, a_ps2, a_ps3, a_saf, W.HBox([a_note, a_who]),
                        W.HBox([b_ok, b_save, b_next, b_del]), W.HBox([n_samp, b_sample, b_rep]), a_out])
    ui = W.VBox([controls, W.HBox([img_w, W.VBox([info, msg])]), audit_box])
    load_clip()
    display(ui)
    return ui
