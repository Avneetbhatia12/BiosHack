"""Standalone HTML review player for one clip: PS1 hands + PS2 objects/contacts + PS3 steps + PS5 flags drawn live
over the ORIGINAL video, with a clickable timeline, jump-to-hard-moment buttons, frame stepping and a manual audit.

The HTML file holds numbers only (keypoints, boxes, intervals). The video is NOT embedded: the reviewer opens the
clip from their own disk with the file picker, so the file can be passed around the team without copying footage.
Audit rows are kept in the browser (localStorage) and exported as CSV with the same columns as review.AuditLog;
`AuditLog(OUT_ROOT, stem).merge_csv(path)` brings them back into the notebook.

    export_review_html(clip_path, OUT_ROOT)  ->  OUT_ROOT/<stem>/review/<stem>_review.html
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import results as R
from .review import AUDIT_FIELDS, ClipView, JUMP_KINDS

SIDE = {"left": 0, "right": 1}
OTYPE = {"portable": 0, "fixed": 1, "unknown": 2}


def _payload(view: ClipView, max_objects: int = 8, include_all_objects: bool = True) -> dict:
    names: dict = {}

    def nid(n):
        if n not in names:
            names[n] = len(names)
        return names[n]

    n = view.n_frames or (max(view.hands) + 1 if view.hands else 0)
    hands, objs = [None] * n, [None] * n
    for f, r in view.hands.items():
        if 0 <= f < n and r["hands"]:
            hands[f] = [[SIDE.get(h.get("side"), 2), int(h.get("id", -1) or -1), int(round(100 * float(h.get("score", 0)))),
                         int(bool(h.get("interpolated"))), int(bool(h.get("side_corrected"))),
                         *[int(round(v)) for v in h["bbox"]],
                         *[int(round(c)) for p in (h.get("kp2d") or [])[:21] for c in p]] for h in r["hands"]]
    for f, r in view.objects.items():
        if not (0 <= f < n):
            continue
        cf = view.contact_frames.get(f, {})
        active = {c["object_id"] for h in cf.get("hands", []) for c in h.get("contacts", [])}
        lst = [o for o in r["objects"] if o["object_id"] in active]
        if include_all_objects:
            rest = sorted((o for o in r["objects"] if o["object_id"] not in active), key=lambda o: -o.get("confidence", 0))
            lst += rest[: max(0, max_objects - len(lst))]
        if lst:
            objs[f] = [[str(o["object_id"]), nid(str(o.get("object_name", "?"))), OTYPE.get(o.get("object_type"), 2),
                        *[int(round(v)) for v in o["bbox"]], int(o["object_id"] in active)] for o in lst]
    lanes = {
        "ps1": [[e["t_start"], e["t_end"], SIDE.get(e["hand"], 2)] for e in view.ps1_segments if e["kind"] == "hands_visible"],
        "ps2": [[c["t_start"], c["t_end"], SIDE.get(c.get("hand"), 2), c["object_name"], c.get("object_type", "")]
                for c in view.contacts],
        "ps3": [[a["start_s"], a["end_s"], a["verb"], a["object"], a.get("hand", ""), round(float(a.get("confidence") or 0), 2),
                 "; ".join(a.get("refined", []) or [])] for a in view.actions],
        "safety": [[s["t_start"], s["t_end"], s.get("hand", ""), s.get("hazard", ""), s.get("level", ""), s.get("risk", "")]
                   for s in view.safety],
        "quality": [[s["t_start"], s["t_end"], s["reason"]] for s in view.quality.get("bad_segments", [])],
    }
    jumps = {k: view.targets(k) for k in JUMP_KINDS if k != "audit_sample"}
    return {"stem": view.stem, "fps": view.fps, "W": view.W, "H": view.H, "n": n, "hands": hands, "objects": objs,
            "names": [k for k, _ in sorted(names.items(), key=lambda x: x[1])], "lanes": lanes, "jumps": jumps,
            "jump_labels": JUMP_KINDS, "samples": sorted(view.samples), "audit": list(view.audit.rows.values()),
            "audit_fields": AUDIT_FIELDS}


def export_review_html(clip_path: str, out_root: str, out_html: Optional[str] = None, max_objects: int = 8,
                       view: Optional[ClipView] = None) -> str:
    view = view or ClipView(clip_path, out_root)
    data = json.dumps(_payload(view, max_objects), separators=(",", ":")).replace("</", "<\\/")
    out_html = out_html or os.path.join(R.paths(out_root, view.stem)["review"], f"{view.stem}_review.html")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__TITLE__", f"Review {view.stem}").replace("__DATA__", data))
    return out_html


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#101318;--panel:#181c23;--line:#2a313c;--fg:#e7eaf0;--mut:#98a2b3;--acc:#4d9cf0;--ok:#3fb27f;--bad:#e5534b;--warn:#e3a33b}
:root[data-theme="light"]{--bg:#f4f5f7;--panel:#fff;--line:#dfe3e8;--fg:#1d2330;--mut:#667085;--acc:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0 12px 0 0}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px 16px}
@media (max-width:1000px){main{grid-template-columns:1fr}}
#stage{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:var(--ar,16/9)}
#stage video,#stage canvas{position:absolute;inset:0;width:100%;height:100%}
#drop{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;color:var(--mut);text-align:center;padding:20px}
.bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:8px 0}
button,select,input{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px}
button{cursor:pointer}button:hover{border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
input[type=number]{width:52px}label.ck{display:inline-flex;gap:4px;align-items:center;margin-right:8px;color:var(--mut)}
#tl{width:100%;height:150px;display:block;background:var(--panel);border:1px solid var(--line);border-radius:8px;cursor:pointer;margin-top:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.card h2{font-size:13px;margin:0 0 8px}.row{display:flex;justify-content:space-between;gap:8px;margin:3px 0}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;background:var(--line);margin:1px 2px;font-size:12px}
.L{color:#52a7ff}.R{color:#ff8a3d}.grid{display:grid;grid-template-columns:auto 60px;gap:4px 8px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}.seg button{border:0;border-radius:0;padding:3px 8px}
.seg button.on{background:var(--acc);color:#fff}kbd{background:var(--line);border-radius:4px;padding:0 5px;font-size:11px}
#time{font-variant-numeric:tabular-nums;min-width:150px}
</style></head><body>
<header><h1 id="title"></h1>
<label class="pri" style="cursor:pointer"><input id="file" type="file" accept="video/*" style="display:none"><span class="pill" style="background:var(--acc);color:#fff;padding:5px 10px">Open the clip…</span></label>
<span class="mut" id="filemsg">The video stays on your computer; this page only holds numbers.</span>
<span style="flex:1"></span><button id="theme">Light / dark</button></header>
<main><section>
<div id="stage"><video id="vid" playsinline preload="auto"></video><canvas id="ov"></canvas>
<div id="drop"><b>Open the original clip (same file the pipeline ran on)</b><span>Use “Open the clip…” above, or drop the file here.</span></div></div>
<div class="bar">
<button id="play" class="pri">Play</button><button data-step="-30">-1s</button><button data-step="-1">-1f</button><button data-step="1">+1f</button><button data-step="30">+1s</button>
<select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
<span id="time" class="mut"></span>
<span style="flex:1"></span>
<select id="jump"></select><button id="jprev">◀ prev</button><button id="jnext">next ▶</button></div>
<div class="bar"><label class="ck"><input type="checkbox" id="lh" checked>PS1 hands</label><label class="ck"><input type="checkbox" id="lk" checked>keypoints</label>
<label class="ck"><input type="checkbox" id="lo" checked>PS2 contact objects</label><label class="ck"><input type="checkbox" id="la">all objects</label>
<label class="ck"><input type="checkbox" id="lc" checked>PS3 step</label><label class="ck"><input type="checkbox" id="ls" checked>PS5 safety / quality</label>
<span class="mut">Keys: <kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> frame · <kbd>shift</kbd>+arrows 1 s · <kbd>c</kbd> all correct &amp; next · <kbd>s</kbd> save · <kbd>n</kbd> next sample</span></div>
<canvas id="tl"></canvas>
</section><aside>
<div class="card"><h2>At this frame</h2><div id="info" class="mut">Open the clip to start.</div></div>
<div class="card"><h2>Manual audit <span class="mut" id="auditstat"></span></h2>
<div class="mut" style="margin-bottom:6px">Count only the worker's own hands that are visible in this frame.</div>
<div class="grid"><span>Predicted hands</span><input id="a_pred" type="number" min="0" readonly>
<span>Missed (visible, not found)</span><input id="a_miss" type="number" min="0" value="0">
<span>False (not a wearer hand)</span><input id="a_false" type="number" min="0" value="0">
<span>Left/right swapped</span><input id="a_swap" type="number" min="0" value="0">
<span>Keypoints clearly off</span><input id="a_kp" type="number" min="0" value="0"></div>
<div class="row" style="margin-top:8px"><span>PS2 contact</span><span class="seg" data-k="ps2_contact"><button>ok</button><button>wrong</button><button class="on">na</button></span></div>
<div class="row"><span>PS3 step</span><span class="seg" data-k="ps3_action"><button>ok</button><button>wrong</button><button class="on">na</button></span></div>
<div class="row"><span>Safety flag</span><span class="seg" data-k="safety_flag"><button>ok</button><button>wrong</button><button class="on">na</button></span></div>
<div class="bar"><input id="a_note" placeholder="note" style="flex:1"><input id="a_who" placeholder="reviewer" style="width:90px"></div>
<div class="bar"><button id="a_ok" class="pri">All correct + next (c)</button><button id="a_save">Save (s)</button><button id="a_next">Next sample (n)</button></div>
<div class="bar"><button id="a_export">Export audit CSV</button><button id="a_clear">Delete this frame's row</button></div>
<div id="summary" class="mut"></div></div>
</aside></main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);const vid=$("#vid"),ov=$("#ov"),ctx=ov.getContext("2d"),tl=$("#tl"),tctx=tl.getContext("2d");
const EDGES=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const SIDEC=["#52a7ff","#ff8a3d","#cccccc"],OTC=["#ff9a2e","#3aa0e6","#aaaaaa"];
const VC={pick:"#50c850",place:"#e6a03c",carry:"#3ca0c8",hold:"#969696",insert:"#c85ac8",align:"#50c8e6",fasten:"#e65a28",tighten:"#c82828",loosen:"#c83c78",press:"#c8c800",push:"#dcdc64",pull:"#3c78b4",turn:"#a078dc",cut:"#a01e1e",wipe:"#78c8c8",inspect:"#2882fa",adjust:"#a0c88c",other:"#787878"};
$("#title").textContent=D.stem+" · "+D.n+" frames @ "+D.fps.toFixed(2)+" fps";
document.documentElement.style.setProperty("--ar",(D.W||16)+"/"+(D.H||9));
const KEY="htreview:"+D.stem;let audit={};
for(const r of D.audit||[])audit[+r.frame]=r;
try{const s=JSON.parse(localStorage.getItem(KEY)||"{}");for(const k in s)audit[+k]=s[k]}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(audit))}catch(e){}}
const fmt=t=>{t=Math.max(0,t);const m=Math.floor(t/60);return m+":"+(t-60*m).toFixed(2).padStart(5,"0")};
let curF=0;const fOf=t=>Math.min(D.n-1,Math.max(0,Math.floor(t*D.fps+1e-3)));
function seekF(f){f=Math.min(D.n-1,Math.max(0,f));vid.currentTime=(f+0.5)/D.fps;curF=f;draw()}
// jumps
const js=$("#jump");for(const k in D.jump_labels){const n=k==="audit_sample"?D.samples.length:(D.jumps[k]||[]).length;const o=document.createElement("option");o.value=k;o.textContent=D.jump_labels[k]+" ("+n+")";js.appendChild(o)}
function targets(k){if(k==="audit_sample")return D.samples.filter(f=>!(f in audit));return D.jumps[k]||[]}
function jump(dir){const ts=targets(js.value);if(!ts.length)return;let f=null;if(dir>0){f=ts.find(x=>x>curF)}else{for(let i=ts.length-1;i>=0;i--)if(ts[i]<curF){f=ts[i];break}}if(f!=null){vid.pause();seekF(f)}}
$("#jprev").onclick=()=>jump(-1);$("#jnext").onclick=()=>jump(1);
// file
function openFile(f){if(!f)return;vid.src=URL.createObjectURL(f);$("#drop").style.display="none";$("#filemsg").textContent=f.name+(f.name.indexOf(D.stem)<0?"  ⚠ name differs from "+D.stem:"")}
$("#file").onchange=e=>openFile(e.target.files[0]);
const st=$("#stage");st.ondragover=e=>e.preventDefault();st.ondrop=e=>{e.preventDefault();openFile(e.dataTransfer.files[0])};
$("#play").onclick=()=>vid.paused?vid.play():vid.pause();vid.onplay=()=>$("#play").textContent="Pause";vid.onpause=()=>{$("#play").textContent="Play";curF=fOf(vid.currentTime);draw()};
document.querySelectorAll("[data-step]").forEach(b=>b.onclick=()=>{vid.pause();seekF(curF+(+b.dataset.step))});
$("#speed").onchange=e=>vid.playbackRate=+e.target.value;
$("#theme").onclick=()=>{const r=document.documentElement;r.dataset.theme=r.dataset.theme==="light"?"dark":"light";drawTL()};
// overlay
function within(l,t){return l.filter(x=>x[0]<=t&&t<=x[1])}
function draw(){const r=st.getBoundingClientRect();const dpr=window.devicePixelRatio||1;
 if(ov.width!==Math.round(r.width*dpr)){ov.width=Math.round(r.width*dpr);ov.height=Math.round(r.height*dpr)}
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,ov.width,ov.height);const sx=ov.width/D.W,sy=ov.height/D.H;ctx.setTransform(sx,0,0,sy,0,0);
 const f=curF,t=f/D.fps,s=D.H/720;ctx.lineJoin="round";
 const show=id=>$(id).checked;
 const objs=D.objects[f]||[];
 if(show("#lo")||show("#la"))for(const o of objs){if(!o[7]&&!show("#la"))continue;ctx.strokeStyle=o[7]?"#3fdc5a":OTC[o[2]];ctx.lineWidth=(o[7]?5:2)*s;ctx.strokeRect(o[3],o[4],o[5]-o[3],o[6]-o[4]);
  const lab=o[0]+" "+D.names[o[1]];ctx.font=(14*s)+"px system-ui";ctx.fillStyle=o[7]?"#3fdc5a":OTC[o[2]];ctx.fillRect(o[3],o[4]-18*s,ctx.measureText(lab).width+8*s,18*s);ctx.fillStyle="#000";ctx.fillText(lab,o[3]+4*s,o[4]-5*s)}
 const hs=D.hands[f]||[];
 if(show("#lh"))for(const h of hs){const c=SIDEC[h[0]];ctx.strokeStyle=c;ctx.lineWidth=2*s;ctx.setLineDash(h[3]?[6*s,4*s]:[]);ctx.strokeRect(h[5],h[6],h[7]-h[5],h[8]-h[6]);ctx.setLineDash([]);
  if(show("#lk")&&h.length>=51){const k=i=>[h[9+2*i],h[10+2*i]];ctx.lineWidth=(h[3]?1.5:3)*s;ctx.beginPath();for(const [a,b] of EDGES){const p=k(a),q=k(b);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1])}ctx.stroke();ctx.fillStyle="#fff";for(let i=0;i<21;i++){const p=k(i);ctx.beginPath();ctx.arc(p[0],p[1],3*s,0,7);ctx.fill()}}
  const lab=(h[0]===0?"L":h[0]===1?"R":"?")+" id"+h[1]+" "+(h[2]/100).toFixed(2)+(h[3]?" interp":"")+(h[4]?" side-fixed":"");ctx.font="bold "+(15*s)+"px system-ui";
  ctx.fillStyle=c;ctx.fillRect(h[5],h[6]-20*s,ctx.measureText(lab).width+8*s,20*s);ctx.fillStyle="#fff";ctx.fillText(lab,h[5]+4*s,h[6]-5*s)}
 if(show("#lc")){const acts=within(D.lanes.ps3,t);ctx.font="bold "+(20*s)+"px system-ui";let x=10*s;for(const a of acts.slice(0,3)){const lab=a[2].toUpperCase()+" "+a[3]+" ["+a[4]+"]";const w=ctx.measureText(lab).width+16*s;ctx.fillStyle=VC[a[2]]||"#888";ctx.fillRect(x,8*s,w,32*s);ctx.fillStyle="#fff";ctx.fillText(lab,x+8*s,31*s);x+=w+10*s}}
 if(show("#ls")){const sf=within(D.lanes.safety,t);if(sf.length){const hi=sf.some(x=>x[5]==="HIGH");ctx.strokeStyle=hi?"#ff2a2a":"#ff9a2e";ctx.lineWidth=10*s;ctx.strokeRect(5*s,5*s,D.W-10*s,D.H-10*s);
  const lab="SAFETY "+sf[0][5]+": "+sf[0][2]+" hand "+sf[0][4]+" "+sf[0][3];ctx.font="bold "+(18*s)+"px system-ui";ctx.fillStyle=hi?"#ff2a2a":"#ff9a2e";ctx.fillRect(14*s,D.H-70*s,ctx.measureText(lab).width+16*s,30*s);ctx.fillStyle="#fff";ctx.fillText(lab,22*s,D.H-49*s)}
  const q=within(D.lanes.quality,t);if(q.length){const lab="QUALITY: "+q[0][2];ctx.font=(16*s)+"px system-ui";ctx.fillStyle="#ffd23c";ctx.fillRect(14*s,D.H-108*s,ctx.measureText(lab).width+16*s,28*s);ctx.fillStyle="#000";ctx.fillText(lab,22*s,D.H-88*s)}}
 $("#time").textContent="frame "+f+" · "+fmt(t);info(f,t,hs);drawTL();loadAudit(f,hs)}
function info(f,t,hs){const sid=["L","R","?"];let h="";
 h+=hs.length?hs.map(x=>`<div class="row"><span class="${sid[x[0]]}">${sid[x[0]]} id${x[1]}</span><span>score ${(x[2]/100).toFixed(2)}${x[3]?" · interp":""}${x[4]?" · side fixed":""}</span></div>`).join(""):"<div>no hand found</div>";
 const c=within(D.lanes.ps2,t);if(c.length)h+="<div style='margin-top:6px'>Contacts: "+c.map(x=>`<span class="pill">${sid[x[2]]} · ${x[3]}</span>`).join("")+"</div>";
 const a=within(D.lanes.ps3,t);if(a.length)h+="<div style='margin-top:6px'>Step: "+a.map(x=>`<span class="pill">${x[2]} ${x[3]} · ${x[4]} · ${fmt(x[0])}–${fmt(x[1])}</span>`+(x[6]?`<div class="mut">refined: ${x[6]}</div>`:"")).join("")+"</div>";
 const s=within(D.lanes.safety,t);if(s.length)h+="<div style='margin-top:6px;color:var(--bad)'>Safety: "+s.map(x=>`${x[2]} near ${x[3]} (${x[4]}, ${x[5]})`).join("; ")+"</div>";
 $("#info").innerHTML=h}
// timeline
const LANES=[["L hand","ps1",0],["R hand","ps1",1],["L touch","ps2",0],["R touch","ps2",1],["step","ps3"],["safety","safety"],["quality","quality"]];
let tlView=null;
function drawTL(){const r=tl.getBoundingClientRect(),dpr=window.devicePixelRatio||1;tl.width=Math.round(r.width*dpr);tl.height=Math.round(r.height*dpr);tctx.setTransform(dpr,0,0,dpr,0,0);
 const W=r.width,H=r.height,lab=64,lh=(H-22)/LANES.length,dur=D.n/D.fps;const cs=getComputedStyle(document.documentElement);
 tctx.clearRect(0,0,W,H);tctx.font="11px system-ui";const X=t=>lab+(W-lab-6)*t/dur;tlView={X,lab,W,dur};
 LANES.forEach((L,i)=>{const y=4+i*lh;tctx.fillStyle=cs.getPropertyValue("--mut");tctx.fillText(L[0],4,y+lh*0.7);tctx.fillStyle=cs.getPropertyValue("--line");tctx.fillRect(lab,y+1,W-lab-6,lh-3);
  let items=D.lanes[L[1]];if(L[2]!==undefined)items=items.filter(x=>x[2]===L[2]);
  for(const it of items){const x0=X(it[0]),w=Math.max(1.5,X(it[1])-x0);tctx.fillStyle=L[1]==="ps1"?SIDEC[L[2]]:L[1]==="ps2"?(it[4]==="portable"?"#ff9a2e":"#3aa0e6"):L[1]==="ps3"?(VC[it[2]]||"#888"):L[1]==="safety"?(it[5]==="HIGH"?"#ff2a2a":it[5]==="MEDIUM"?"#ff9a2e":"#c8b464"):"#ffd23c";tctx.fillRect(x0,y+2,w,lh-5)}});
 for(const f in audit){tctx.fillStyle=cs.getPropertyValue("--ok");tctx.fillRect(X(f/D.fps)-0.5,H-16,1.5,6)}
 for(const f of D.samples){if(!(f in audit)){tctx.fillStyle=cs.getPropertyValue("--warn");tctx.fillRect(X(f/D.fps)-0.5,H-10,1.5,6)}}
 tctx.fillStyle=cs.getPropertyValue("--fg");const xp=X(curF/D.fps);tctx.fillRect(xp-1,0,2,H)}
tl.onclick=e=>{if(!tlView)return;const r=tl.getBoundingClientRect();const t=(e.clientX-r.left-tlView.lab)/(tlView.W-tlView.lab-6)*tlView.dur;vid.pause();seekF(Math.round(t*D.fps))};
// audit
document.querySelectorAll(".seg").forEach(g=>g.querySelectorAll("button").forEach(b=>b.onclick=()=>{g.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on")}));
const segv=k=>document.querySelector(`.seg[data-k="${k}"] .on`).textContent;
const setseg=(k,v)=>document.querySelectorAll(`.seg[data-k="${k}"] button`).forEach(b=>b.classList.toggle("on",b.textContent===(v||"na")));
function loadAudit(f,hs){const r=audit[f];$("#a_pred").value=hs.length;$("#a_miss").value=r?r.missed_hands:0;$("#a_false").value=r?r.false_hands:0;$("#a_swap").value=r?r.side_swaps:0;$("#a_kp").value=r?r.bad_keypoints:0;
 setseg("ps2_contact",r&&r.ps2_contact);setseg("ps3_action",r&&r.ps3_action);setseg("safety_flag",r&&r.safety_flag);$("#a_note").value=r?r.note||"":"";
 const done=Object.keys(audit).length,left=D.samples.filter(x=>!(x in audit)).length;$("#auditstat").textContent=`· ${done} frames reviewed${D.samples.length?" · "+left+" samples left":""}${r?" · this frame saved":""}`;summary()}
function save(){const f=curF;audit[f]={clip:D.stem,frame:f,t:(f/D.fps).toFixed(2),split:(Object.keys(audit).length%2?"dev":"test"),pred_hands:+$("#a_pred").value,missed_hands:+$("#a_miss").value,false_hands:+$("#a_false").value,side_swaps:+$("#a_swap").value,bad_keypoints:+$("#a_kp").value,
 ps2_contact:segv("ps2_contact"),ps3_action:segv("ps3_action"),safety_flag:segv("safety_flag"),note:$("#a_note").value,reviewer:$("#a_who").value,saved_at:new Date().toISOString()};persist();draw()}
function nextSample(){const ts=D.samples.filter(x=>!(x in audit));if(!ts.length){$("#summary").textContent="No samples left: export the CSV.";return}const f=ts.find(x=>x>curF)??ts[0];vid.pause();seekF(f)}
$("#a_save").onclick=save;$("#a_next").onclick=nextSample;
$("#a_ok").onclick=()=>{["#a_miss","#a_false","#a_swap","#a_kp"].forEach(s=>$(s).value=0);save();nextSample()};
$("#a_clear").onclick=()=>{delete audit[curF];persist();draw()};
$("#a_export").onclick=()=>{const F=D.audit_fields;const esc=v=>{v=v==null?"":String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
 const rows=[F.join(",")].concat(Object.keys(audit).map(Number).sort((a,b)=>a-b).map(f=>F.map(k=>esc(audit[f][k])).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}));a.download=D.stem+"_manual_audit.csv";a.click()};
function summary(){let fr=0,pred=0,miss=0,fal=0,sw=0,kp=0;for(const k in audit){const r=audit[k];fr++;pred+=+r.pred_hands;miss+=+r.missed_hands;fal+=+r.false_hands;sw+=+r.side_swaps;kp+=+r.bad_keypoints}
 if(!fr){$("#summary").textContent="";return}const found=Math.max(0,pred-fal),pc=x=>isFinite(x)?(100*x).toFixed(1)+"%":"–";
 $("#summary").innerHTML=`${fr} frames · recall ${pc(found/(found+miss))} · precision ${pc(found/pred)} · side accuracy ${pc((found-sw)/found)} · keypoints ok ${pc((found-kp)/found)}`}
// keys + sync
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"&&e.target.type!=="checkbox")return;const big=e.shiftKey?Math.round(D.fps):1;
 if(e.key===" "){e.preventDefault();vid.paused?vid.play():vid.pause()}else if(e.key==="ArrowRight"){e.preventDefault();vid.pause();seekF(curF+big)}else if(e.key==="ArrowLeft"){e.preventDefault();vid.pause();seekF(curF-big)}
 else if(e.key==="c")$("#a_ok").click();else if(e.key==="s")save();else if(e.key==="n")nextSample()});
document.querySelectorAll("input[type=checkbox]").forEach(c=>c.onchange=draw);
function tick(now,meta){curF=fOf(meta?meta.mediaTime:vid.currentTime);draw();if(vid.requestVideoFrameCallback)vid.requestVideoFrameCallback(tick)}
if(vid.requestVideoFrameCallback)vid.requestVideoFrameCallback(tick);else(function loop(){if(!vid.paused){curF=fOf(vid.currentTime);draw()}requestAnimationFrame(loop)})();
vid.onseeked=()=>{curF=fOf(vid.currentTime);draw()};window.onresize=draw;
vid.onloadedmetadata=()=>{if(Math.abs(vid.duration-D.n/D.fps)>1)$("#filemsg").textContent+="  ⚠ length "+vid.duration.toFixed(1)+" s ≠ "+(D.n/D.fps).toFixed(1)+" s: wrong clip?";seekF(D.samples.find(x=>!(x in audit))??0)};
draw();
</script></body></html>"""
