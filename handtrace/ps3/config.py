"""PS3 settings and the fixed action vocabulary (one vocabulary for every clip)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional

VERBS: List[str] = [
    "pick", "place", "carry", "hold", "insert", "align", "fasten", "tighten", "loosen",
    "press", "push", "pull", "turn", "cut", "wipe", "inspect", "adjust", "other",
]
HANDS: List[str] = ["left", "right", "both", "unknown"]

# Words the model might use anyway -> the fixed vocabulary.
VERB_SYNONYMS = {
    "grab": "pick", "take": "pick", "pick up": "pick", "pickup": "pick", "grasp": "pick", "lift": "pick",
    "put": "place", "put down": "place", "set": "place", "set down": "place", "drop": "place", "release": "place",
    "move": "carry", "transfer": "carry", "bring": "carry",
    "grip": "hold", "support": "hold", "steady": "hold",
    "screw": "fasten", "bolt": "fasten", "clip": "fasten", "attach": "fasten", "assemble": "fasten",
    "connect": "fasten", "mount": "fasten", "fix": "fasten", "secure": "fasten",
    "unscrew": "loosen", "undo": "loosen", "remove": "loosen", "detach": "loosen",
    "tighten up": "tighten", "torque": "tighten",
    "push in": "insert", "slide in": "insert", "plug": "insert", "fit": "insert",
    "position": "align", "line up": "align",
    "press down": "press", "click": "press", "push button": "press",
    "rotate": "turn", "twist": "turn", "spin": "turn",
    "trim": "cut", "snip": "cut", "saw": "cut",
    "clean": "wipe", "polish": "wipe", "brush": "wipe",
    "check": "inspect", "look": "inspect", "examine": "inspect", "measure": "inspect", "test": "inspect",
    "tweak": "adjust", "set up": "adjust", "reposition": "adjust",
}


@dataclass
class PS3Config:
    # --- model / API ---------------------------------------------------------------------------------
    # "auto" = the newest plain "gemini-<version>-flash" in the live model list (Oct 2026: gemini-3.5-flash is
    # deprecated and routed to gemini-3.6-flash). An explicit id is used as given (with a warning if it is not listed).
    model: str = "auto"
    model_preference: List[str] = field(default_factory=lambda: ["gemini-3.8-flash", "gemini-3.6-flash",
                                                                 "gemini-3.5-flash", "gemini-3-flash"])
    api_key_env: str = "GEMINI_API_KEY"    # the key is read from this environment variable, never printed or saved
    fps: float = 2.0                       # frames per second the model sees (API default is 1; range (0, 24])
    temperature: Optional[float] = None    # None = model default (Google recommends the default for Gemini 3.x)
    thinking_level: Optional[str] = "medium"   # minimal | low | medium | high | None (server default)
    media_resolution: Optional[str] = None     # None | "low" (66 tokens/frame) | "medium" | "high": token cost vs detail
    max_output_tokens: int = 8192
    use_schema: bool = True                # constrain the reply with a response schema; falls back if rejected
    # --- what is uploaded ------------------------------------------------------------------------------
    upload_small_copy: bool = True         # upload a 480p copy (same timeline): faster, fits the free-tier 2 GB limit
    small_copy_roots: List[str] = field(default_factory=list)   # where a ready-made clips_480p/ copy may be found
    transcode_height: int = 480            # no ready copy -> ffmpeg makes one (deleted again when the clip is done)
    # --- windows -------------------------------------------------------------------------------------
    window_s: float = 60.0
    overlap_s: float = 10.0
    time_base: str = "auto"                # "auto" | "absolute" | "relative": see validate.resolve_time_base
    only_windows: Optional[List[int]] = None   # e.g. [0] or [1]: smoke test; finals go to ps3/partial/
    # --- reliability ---------------------------------------------------------------------------------
    max_retries: int = 6
    retry_base_s: float = 4.0
    retry_cap_s: float = 90.0
    poll_s: float = 5.0
    poll_timeout_s: float = 900.0
    parse_retries: int = 1                 # extra API calls if a reply is not valid JSON
    delete_remote: bool = True             # delete the uploaded file after the clip succeeded
    force: bool = False                    # True: reprocess a finished clip (otherwise it is skipped)
    # --- validation / merge ---------------------------------------------------------------------------
    min_action_s: float = 0.4
    boundary_tol_s: float = 1.5            # allowed overshoot past a window edge before an action is dropped
    min_confidence: float = 0.0            # drop actions below this (0 keeps everything; confidence is stored)
    merge_gap_s: float = 1.0               # same action separated by less than this is one action
    # --- PS2 links -------------------------------------------------------------------------------------
    use_ps2_hints: bool = True             # object names from ps2_final/contacts.csv go into the prompt as hints
    refine_with_ps2: bool = True           # snap action edges to PS2 grab/release moments -> actions_final.csv
    snap_tol_s: float = 1.5                # an edge moves at most this far to meet a grab / release

    def to_dict(self) -> dict:
        return asdict(self)
