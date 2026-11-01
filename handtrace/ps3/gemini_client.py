"""Thin wrapper around the official Google GenAI SDK: upload, poll, generate (with retries), delete.
The API key is read from the environment variable named in the config and is never printed or stored."""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import PS3Config

RETRY_CODES = {408, 429, 500, 502, 503, 504}


@dataclass
class RemoteFile:
    name: str
    uri: str
    mime_type: str


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b([45]\d\d)\b", str(exc)[:200])
    return int(m.group(1)) if m else None


def is_retryable(exc: Exception) -> bool:
    if _status_code(exc) in RETRY_CODES:
        return True
    text = str(exc).upper()
    return any(k in text for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "OVERLOADED",
                                   "TIMED OUT", "TIMEOUT", "CONNECTION", "TEMPORARILY"))


def _server_delay(exc: Exception) -> float:
    m = re.search(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", str(exc), flags=re.I)
    return float(m.group(1)) if m else 0.0


class GeminiVideoClient:
    def __init__(self, cfg: PS3Config, log: Callable = print):
        key = os.environ.get(cfg.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"{cfg.api_key_env} is not set. In Kaggle: Add-ons -> Secrets, then copy it into "
                               f"os.environ in a cell WITHOUT printing it.")
        from google import genai                                   # imported late: the module loads without the SDK
        from google.genai import types
        self.cfg, self.log, self._key = cfg, log, key
        self._types = types
        self._client = genai.Client(api_key=key)
        self._schema_ok = cfg.use_schema
        self._extras_ok = True                                     # thinking_level / media_resolution accepted
        self.model = self.resolve_model()

    # ----------------------------------------------------------------------------------------- safety
    def _clean(self, text: str) -> str:
        return text.replace(self._key, "***")

    def _with_retries(self, fn: Callable, what: str):
        cfg = self.cfg
        for attempt in range(cfg.max_retries + 1):
            try:
                return fn()
            except Exception as exc:                                # noqa: BLE001: SDK raises many types
                if attempt >= cfg.max_retries or not is_retryable(exc):
                    raise type(exc)(self._clean(str(exc))) from None
                delay = max(min(cfg.retry_cap_s, cfg.retry_base_s * 2 ** attempt) * (0.5 + random.random()),
                            _server_delay(exc))
                self.log(f"  [{what}] temporary error ({self._clean(str(exc))[:120]}); retry {attempt + 1}/"
                         f"{cfg.max_retries} in {delay:.0f}s")
                time.sleep(delay)

    # ----------------------------------------------------------------------------------------- model check
    def live_models(self) -> Optional[list]:
        """Model ids that support generateContent, or None if the list could not be read."""
        try:
            out = []
            for m in self._client.models.list():
                acts = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
                if not acts or any("generate" in str(a).lower() for a in acts):
                    out.append(m.name.split("/")[-1])
            return out
        except Exception:                                           # noqa: BLE001
            return None

    def model_available(self) -> Optional[bool]:
        """True/False if the configured model id is in the live model list, None if the list could not be read."""
        names = self.live_models()
        return None if names is None else self.model.split("/")[-1] in names

    def resolve_model(self) -> str:
        """cfg.model as given, or for "auto": the first of cfg.model_preference that is live, else the newest
        plain gemini-<version>-flash (no lite / preview / image / tts / live variants)."""
        want = self.cfg.model.strip()
        names = self.live_models()
        if want and want != "auto":
            if names is not None and want.split("/")[-1] not in names:
                self.log(f"  WARNING: model '{want}' is not in the live model list; it may be deprecated or renamed")
            return want
        if not names:
            fallback = self.cfg.model_preference[1] if len(self.cfg.model_preference) > 1 else "gemini-3.6-flash"
            self.log(f"  model list unavailable; using {fallback}")
            return fallback
        for cand in self.cfg.model_preference:
            if cand in names:
                return cand
        plain = []
        for n in names:
            m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n)
            if m:
                plain.append((float(m.group(1)), n))
        if plain:
            return max(plain)[1]
        flash = sorted(n for n in names if "flash" in n and "lite" not in n and "image" not in n)
        if flash:
            return flash[-1]
        raise RuntimeError("no Gemini flash model is available to this API key")

    # ----------------------------------------------------------------------------------------- files
    @staticmethod
    def _state(f) -> str:
        return getattr(f.state, "name", str(f.state)).upper() if getattr(f, "state", None) is not None else "ACTIVE"

    def upload(self, path: str) -> RemoteFile:
        f = self._with_retries(lambda: self._client.files.upload(file=path), "upload")
        return self.wait_active(RemoteFile(f.name, f.uri, f.mime_type or "video/mp4"))

    def wait_active(self, remote: RemoteFile) -> RemoteFile:
        t0 = time.time()
        while True:
            f = self._with_retries(lambda: self._client.files.get(name=remote.name), "poll")
            state = self._state(f)
            if state == "ACTIVE":
                return RemoteFile(f.name, f.uri, f.mime_type or remote.mime_type)
            if state == "FAILED":
                raise RuntimeError(f"Gemini failed to process the uploaded video ({remote.name})")
            if time.time() - t0 > self.cfg.poll_timeout_s:
                raise TimeoutError(f"upload still {state} after {self.cfg.poll_timeout_s:.0f}s")
            time.sleep(self.cfg.poll_s)

    def get_active(self, name: str) -> Optional[RemoteFile]:
        """A previously uploaded file if it still exists and is ACTIVE, else None (it expires after about 48 h)."""
        try:
            f = self._client.files.get(name=name)
        except Exception:                                           # noqa: BLE001
            return None
        return RemoteFile(f.name, f.uri, f.mime_type or "video/mp4") if self._state(f) == "ACTIVE" else None

    def delete(self, name: str) -> bool:
        try:
            self._with_retries(lambda: self._client.files.delete(name=name), "delete")
            return True
        except Exception as exc:                                    # noqa: BLE001
            self.log(f"  could not delete {name}: {self._clean(str(exc))[:120]}")
            return False

    def delete_all_remote(self) -> int:
        """Event-end cleanup: remove every file in this key's Gemini file storage."""
        n = 0
        for f in self._client.files.list():
            n += int(self.delete(f.name))
        return n

    # ----------------------------------------------------------------------------------------- generation
    def generate(self, remote: RemoteFile, start_s: float, end_s: float, prompt: str) -> str:
        t, cfg = self._types, self.cfg
        part = t.Part(
            file_data=t.FileData(file_uri=remote.uri, mime_type=remote.mime_type),
            video_metadata=t.VideoMetadata(start_offset=f"{start_s:.1f}s", end_offset=f"{end_s:.1f}s", fps=cfg.fps),
        )
        contents = [t.Content(role="user", parts=[part, t.Part(text=prompt)])]

        def call(with_schema: bool) -> str:
            kwargs = dict(response_mime_type="application/json", max_output_tokens=cfg.max_output_tokens)
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            if self._extras_ok and cfg.thinking_level:
                kwargs["thinking_config"] = t.ThinkingConfig(thinking_level=cfg.thinking_level.upper())
            if self._extras_ok and cfg.media_resolution:
                kwargs["media_resolution"] = f"MEDIA_RESOLUTION_{cfg.media_resolution.upper()}"
            if with_schema:
                from .schema import ActionList
                kwargs["response_schema"] = ActionList
            resp = self._client.models.generate_content(model=self.model, contents=contents,
                                                        config=t.GenerateContentConfig(**kwargs))
            if not resp.text:
                reason = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
                raise ValueError(f"empty reply (finish_reason={reason})")
            return resp.text

        def attempt(with_schema: bool) -> str:
            try:
                return self._with_retries(lambda: call(with_schema), "generate")
            except Exception as exc:                                # noqa: BLE001
                low = str(exc).lower()
                if _status_code(exc) == 400 and self._extras_ok and ("thinking" in low or "media_resolution" in low
                                                                     or "resolution" in low):
                    self.log("  thinking_level / media_resolution rejected by this model; retrying without them")
                    self._extras_ok = False
                    return self._with_retries(lambda: call(with_schema), "generate")
                raise

        if self._schema_ok:
            try:
                return attempt(True)
            except Exception as exc:                                # noqa: BLE001
                if _status_code(exc) == 400 and "schema" in str(exc).lower():
                    self.log("  response schema rejected by this model; continuing with plain JSON mode")
                    self._schema_ok = False
                else:
                    raise
        return attempt(False)
