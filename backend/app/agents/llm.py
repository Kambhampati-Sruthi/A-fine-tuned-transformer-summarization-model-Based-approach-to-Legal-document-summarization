"""Provider-agnostic LLM adapter (OpenAI-compatible chat-completions). Disabled by default;
the system is fully functional (extractive) without it."""
from __future__ import annotations
import json, logging, re
import httpx
from ..config import Settings

log = logging.getLogger(__name__)


class LLM:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.enabled = cfg.llm_provider == "openai_compatible" and bool(cfg.llm_base_url and cfg.llm_model)

    def complete_json(self, system: str, user: str, timeout: float = 60.0) -> dict | None:
        if not self.enabled:
            return None
        try:
            r = httpx.post(
                self.cfg.llm_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.llm_api_key}"} if self.cfg.llm_api_key else {},
                json={"model": self.cfg.llm_model, "temperature": 0,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=timeout)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception as exc:  # network/provider/JSON errors → caller falls back to extractive
            log.warning("LLM call failed: %s", type(exc).__name__)
            return None
