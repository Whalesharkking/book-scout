"""Lean Ollama client for the wish re-ranker."""

import json
import logging
import time

import requests

from . import prompts
from .profile import norm_title

log = logging.getLogger("agent")

BATCH = 20  # books per re-ranker call

RATINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "wish_fit": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["title", "wish_fit", "reason"],
            },
        },
    },
    "required": ["ratings"],
}


class Ollama:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def wait_ready(self, timeout: int = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                requests.get(f"{self.host}/api/version", timeout=5)
                return
            except requests.RequestException:
                time.sleep(3)
        raise RuntimeError(f"Ollama not reachable at {self.host}")

    def ensure_model(self) -> None:
        tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        if any(n == self.model or n.startswith(self.model + ":") for n in names):
            return
        log.info("Downloading model %s (one time, may take a while) ...", self.model)
        last_pct = -10
        with requests.post(
            f"{self.host}/api/pull",
            json={"model": self.model},
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                status = json.loads(line)
                total, done = status.get("total"), status.get("completed")
                if total and done:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 10:
                        log.info("Model download: %d%%", pct)
                        last_pct = pct
        log.info("Model %s ready", self.model)

    def _chat(self, system: str, user: str, schema: dict, temperature: float) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        # Only thinking models know this parameter, others reject it.
        if self.model.startswith(("qwen3", "deepseek")):
            payload["think"] = False
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def score_wish(self, wish_block: str, books: list[dict], passes: int) -> dict[str, tuple]:
        """Rates the wish fit of every book, averaged over several passes.

        Returns norm_title -> (wish_fit, reason). Books the LLM failed to
        rate are simply missing and fall back to the no-wish weighting.
        """
        # norm_title -> list of (fit, reason) from the individual passes
        collected: dict[str, list] = {}
        for start in range(0, len(books), BATCH):
            batch = books[start : start + BATCH]
            names = "\n".join(f'- "{b["title"]}" by {b["author"]}' for b in batch)
            user = prompts.RERANK_USER.format(wish_block=wish_block, names=names)
            for _ in range(max(1, passes)):
                try:
                    data = self._chat(prompts.RERANK_SYSTEM, user, RATINGS_SCHEMA, 0.2)
                except (requests.RequestException, ValueError, KeyError) as exc:
                    log.warning("Re-ranker call failed: %s", exc)
                    continue
                for item in data.get("ratings", []):
                    try:
                        fit = max(0, min(100, int(item.get("wish_fit", 0))))
                    except (TypeError, ValueError):
                        continue
                    title = norm_title(str(item.get("title", "")))
                    collected.setdefault(title, []).append(
                        (fit, str(item.get("reason", ""))[:140])
                    )
            done = min(start + BATCH, len(books))
            log.info("Re-ranker: %d/%d books rated", done, len(books))

        results = {}
        for book in books:
            runs = collected.get(norm_title(book["title"]))
            if runs:
                fit = round(sum(r[0] for r in runs) / len(runs))
                results[norm_title(book["title"])] = (fit, runs[0][1])
        return results
