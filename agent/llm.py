"""Schlanker Ollama-Client für Generator- und Scoring-Aufrufe."""

import json
import logging
import time

import requests

from . import prompts, scoring
from .profile import norm_title

log = logging.getLogger("agent")

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "language": {"type": "string", "enum": ["de", "en"]},
                },
                "required": ["title", "author", "language"],
            },
        },
    },
    "required": ["candidates"],
}

SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "wish_fit": {"type": "integer"},
                    "taste_fit": {"type": "integer"},
                    "quality": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["title", "wish_fit", "taste_fit", "quality", "reason"],
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
        raise RuntimeError(f"Ollama unter {self.host} nicht erreichbar")

    def ensure_model(self) -> None:
        tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        if any(n == self.model or n.startswith(self.model + ":") for n in names):
            return
        log.info("Lade Modell %s herunter (einmalig, kann dauern) ...", self.model)
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
                        log.info("Modell-Download: %d%%", pct)
                        last_pct = pct
        log.info("Modell %s bereit", self.model)

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
        # Nur Thinking-Modelle kennen den Parameter; andere lehnen ihn ab.
        if self.model.startswith(("qwen3", "deepseek")):
            payload["think"] = False
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])

    def generate_candidates(
        self, category: str, profile_block: str, ranking: list, avoid: list, count: int = 10
    ) -> list[dict]:
        if ranking:
            lines = [prompts.RANKING_HEADER.format(worst=ranking[-1]["score"])]
            lines += [
                f"- \"{e['title']}\" by {e['author']} (score {e['score']})" for e in ranking
            ]
            ranking_block = "\n".join(lines)
        else:
            ranking_block = "The leaderboard is still empty."
        avoid_block = "\n".join(f"- {a}" for a in avoid) if avoid else "(none yet)"
        user = prompts.GENERATOR_USER.format(
            brief=prompts.CATEGORY_BRIEFS[category],
            profile_block=profile_block,
            ranking_block=ranking_block,
            avoid_block=avoid_block,
            count=count,
        )
        try:
            data = self._chat(prompts.GENERATOR_SYSTEM, user, CANDIDATES_SCHEMA, 0.9)
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Generator-Aufruf fehlgeschlagen: %s", exc)
            return []
        return [
            c
            for c in data.get("candidates", [])
            if isinstance(c, dict) and c.get("title") and c.get("author")
        ]

    def score(
        self, category: str, profile_block: str, books: list[dict], passes: int = 2
    ) -> list[dict]:
        """Bewertet Bücher in drei Dimensionen, über mehrere Durchläufe gemittelt,
        und kombiniert das Ergebnis mit dem Open-Library-Lesersignal.

        Zuordnung der LLM-Antwort über den normalisierten Titel.
        """
        if not books:
            return []
        names = "\n".join(f'- "{b["title"]}" by {b["author"]}' for b in books)
        user = prompts.SCORER_USER.format(
            brief=prompts.CATEGORY_BRIEFS[category],
            profile_block=profile_block,
            names=names,
        )
        # norm_title -> Liste von (dims, reason) aus den einzelnen Durchläufen
        collected: dict[str, list] = {}
        for _ in range(max(1, passes)):
            try:
                data = self._chat(prompts.SCORER_SYSTEM, user, SCORES_SCHEMA, 0.2)
            except (requests.RequestException, ValueError, KeyError) as exc:
                log.warning("Scoring-Aufruf fehlgeschlagen: %s", exc)
                continue
            for item in data.get("ratings", []):
                title = norm_title(str(item.get("title", "")))
                try:
                    dims = {
                        d: max(0, min(100, int(item.get(d, 0)))) for d in scoring.DIMENSIONS
                    }
                except (TypeError, ValueError):
                    continue
                collected.setdefault(title, []).append(
                    (dims, str(item.get("reason", ""))[:140])
                )

        results = []
        for book in books:
            runs = collected.get(norm_title(book["title"]))
            if not runs:
                continue
            dims = {
                d: round(sum(r[0][d] for r in runs) / len(runs)) for d in scoring.DIMENSIONS
            }
            final, ol = scoring.combine(
                dims, book.get("ratings_average"), book.get("ratings_count")
            )
            results.append(
                {**book, "score": final, "dims": dims, "ol_score": ol, "reason": runs[0][1]}
            )
        return results
