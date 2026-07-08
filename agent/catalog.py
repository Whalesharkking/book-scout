"""Book catalog plus ALS item vectors built by scripts/build_index.py.

Every book proposal originates from this catalog of real Goodreads books,
so invented titles are impossible by construction.
"""

import gzip
import json
import os
import re

import numpy as np
from rapidfuzz import fuzz, process

from .profile import _norm

MATCH_CUTOFF = 80  # fuzzy title score needed when the author last name matches
TITLE_ONLY_CUTOFF = 92  # stricter fallback when no author match exists
AUTHOR_CUTOFF = 60


def _clean_title(title: str) -> str:
    # drop series/edition suffixes such as "(The Expanse, #1)"
    return _norm(re.sub(r"\([^)]*\)", " ", title))


def _lastname(author: str) -> str:
    parts = _norm(author).split()
    return parts[-1] if parts else ""


class Catalog:
    def __init__(self, index_dir: str):
        self.factors = np.load(os.path.join(index_dir, "item_factors.npy"))
        with gzip.open(os.path.join(index_dir, "books.jsonl.gz"), "rt", encoding="utf-8") as fh:
            self.books = [json.loads(line) for line in fh]
        if len(self.books) != self.factors.shape[0]:
            raise ValueError("catalog and factor matrix are out of sync, rebuild the index")
        self.titles = [_clean_title(b["title"]) for b in self.books]
        self._by_lastname: dict[str, list[int]] = {}
        for row, book in enumerate(self.books):
            self._by_lastname.setdefault(_lastname(book["author"]), []).append(row)

    def __len__(self) -> int:
        return len(self.books)

    def match(self, title: str, author: str) -> int | None:
        """Maps a profile book to a catalog row, tolerating spelling variants.

        Catalog rows are ordered by popularity, so among equally good title
        matches the best-known edition wins.
        """
        wanted = _clean_title(title)
        if not wanted:
            return None
        rows = self._by_lastname.get(_lastname(author), [])
        if rows:
            best = process.extractOne(
                wanted,
                {row: self.titles[row] for row in rows},
                scorer=fuzz.token_set_ratio,
                score_cutoff=MATCH_CUTOFF,
            )
            if best:
                return best[2]
        # author unknown or spelled too differently: exact-ish title, then verify author
        best = process.extractOne(
            wanted, self.titles, scorer=fuzz.token_set_ratio, score_cutoff=TITLE_ONLY_CUTOFF
        )
        if best and fuzz.partial_ratio(_norm(author), _norm(self.books[best[2]]["author"])) >= AUTHOR_CUTOFF:
            return best[2]
        return None
