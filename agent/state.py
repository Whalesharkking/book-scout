"""Persistent state: leaderboards plus every proposal ever checked."""

import json
import os

from .profile import _norm

TOP_N = 20
MAX_PER_AUTHOR = 2  # diversity, at most 2 books by the same author per list


def rank(entries: list[dict]) -> list[dict]:
    """Sorts by descending score, limits per author and cuts to TOP_N."""
    ranked = sorted(entries, key=lambda e: (-e["score"], e["title"]))
    per_author: dict[str, int] = {}
    result = []
    for entry in ranked:
        author = _norm(entry.get("author", ""))
        if per_author.get(author, 0) >= MAX_PER_AUTHOR:
            continue
        per_author[author] = per_author.get(author, 0) + 1
        result.append(entry)
        if len(result) == TOP_N:
            break
    return result


class State:
    def __init__(self, path: str):
        self.path = path
        self.iteration = 0
        # book_key -> "found" | "notfound" | "read" (insertion order = check order)
        self.seen: dict[str, str] = {}
        # category -> [{key, title, author, year, language, score, reason}], descending
        self.lists: dict[str, list] = {"nonfiction": [], "other": []}
        # hash of the reading profile at the last scoring, a change triggers a rescore
        self.profile_hash = ""
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.iteration = data.get("iteration", 0)
        self.seen = data.get("seen", {})
        self.lists = data.get("lists", self.lists)
        self.profile_hash = data.get("profile_hash", "")

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "iteration": self.iteration,
                    "seen": self.seen,
                    "lists": self.lists,
                    "profile_hash": self.profile_hash,
                },
                fh,
                ensure_ascii=False,
                indent=1,
            )
        os.replace(tmp, self.path)

    def avoid_sample(self, limit: int = 100) -> list[str]:
        """Recently checked books so that the generator does not repeat them."""
        return list(self.seen)[-limit:]

    def merge(self, category: str, scored: list[dict]) -> int:
        """Adds scored candidates and cuts to TOP_N. Returns the number of new entries."""
        combined = {e["key"]: e for e in self.lists[category]}
        for entry in scored:
            old = combined.get(entry["key"])
            if old is None or entry["score"] > old["score"]:
                combined[entry["key"]] = entry
        new_list = rank(list(combined.values()))
        added = sum(
            1 for e in new_list if e["key"] not in {x["key"] for x in self.lists[category]}
        )
        self.lists[category] = new_list
        return added
