"""Structured video-analysis prompt."""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from .config import HANDS, VERBS

VERB_GUIDE = {
    "pick": "take an object or tool into the hand from a surface, bin or fixture",
    "place": "put down or release a held object onto a surface, fixture or into a container",
    "carry": "move a held object from one place to another (use when the movement itself is the step)",
    "hold": "hold an object steady as a deliberate step (e.g. one hand supports a part while the other works)",
    "insert": "push or slide a part into another part, a slot or a hole",
    "align": "position a part so it lines up with another part",
    "fasten": "join parts with a screw, bolt, clip or similar (running a screwdriver, driving a screw)",
    "tighten": "final tightening or torquing of an already fitted fastener",
    "loosen": "unscrew, remove or detach a fastener or part",
    "press": "push down on something with force (button, press-fit, pressing a part into place)",
    "push": "push or slide an object or lever",
    "pull": "pull or draw an object, cable or lever toward the worker",
    "turn": "rotate a knob, handle, part or tool",
    "cut": "cut, trim or snip",
    "wipe": "clean, wipe, brush or polish",
    "inspect": "look closely at or measure a part, handled or held up for checking",
    "adjust": "fine-tune the position or setting of something",
    "other": "a clear hand-driven action that fits none of the verbs above",
}


def build_prompt(win_start: float, win_end: float, video_duration: float,
                 object_hints: Optional[Sequence[str]] = None) -> str:
    verbs = "\n".join(f"  - {v}: {VERB_GUIDE[v]}" for v in VERBS)
    hints = ""
    if object_hints:
        hints = (
            "\nOBJECT HINTS (from an automatic detector, may be wrong or absent from this segment): "
            + ", ".join(object_hints[:25])
            + ".\nUse a hint only if you can actually see that object; otherwise name what you see, or use \"unknown\".\n"
        )
    return f"""You are annotating first-person video from a camera worn on a factory worker's head.
The camera is NOT mirrored: "left" and "right" mean the worker's own left and right hand.

Analyse ONLY the part of the video from {win_start:.1f} s to {win_end:.1f} s (the full video is {video_duration:.1f} s long).
Report timestamps in seconds measured from the START OF THE FULL VIDEO, not from the start of this segment.
Every action must lie between {win_start:.1f} s and {win_end:.1f} s.

TASK: list the meaningful hand-driven task steps the worker performs, in chronological order.

For each action give:
  start_s, end_s : when the action begins and ends, in seconds, one decimal
  verb           : exactly one of the allowed verbs below
  object         : the tool, component or object being handled (short lowercase noun phrase), or "unknown"
  hand           : one of {", ".join(HANDS)}
  confidence     : 0.0 to 1.0, how sure you are that the action, object and hand are right

ALLOWED VERBS (use only these):
{verbs}

RULES
1. Describe what the worker's hands do, not what the machine does or what other people do.
2. One continuous activity is ONE action. Do not split a long fastening, inspection or carrying into repeated entries.
3. Start when the hand begins the action (reaching for it counts for pick), end when the action is finished.
4. Do not invent objects or actions. If the object is unclear use "unknown"; if the hand is unclear use "unknown".
5. If both hands take part in the same action, use "both".
6. Ignore moments with no purposeful hand action (walking, looking around, hands idle).
7. Actions may follow each other directly; short overlaps are fine if the hands do different things at once.
8. If you see no clear actions in this segment, return {{"actions": []}}.
{hints}
Return JSON only, in this exact structure:
{{"actions": [{{"start_s": 12.0, "end_s": 20.0, "verb": "pick", "object": "screwdriver", "hand": "right", "confidence": 0.9}}]}}
"""


def prompt_key(model: str, fps: float, prompt: str, win_start: float, win_end: float) -> str:
    """Cache key: a stored response is reused only if model, fps, prompt and window are all unchanged."""
    raw = f"{model}|{fps}|{win_start:.3f}|{win_end:.3f}|{prompt}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
