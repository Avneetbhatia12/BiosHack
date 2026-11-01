"""Response schema sent to Gemini (verb and hand are enums, so the vocabulary is enforced)."""
from enum import Enum

from pydantic import BaseModel

from .config import HANDS, VERBS

Verb = Enum("Verb", {v: v for v in VERBS}, type=str)
Hand = Enum("Hand", {h: h for h in HANDS}, type=str)


class ActionItem(BaseModel):
    start_s: float
    end_s: float
    verb: Verb
    object: str
    hand: Hand
    confidence: float


class ActionList(BaseModel):
    actions: list[ActionItem]
