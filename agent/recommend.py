"""Candidate retrieval: nearest neighbours of the favourites in ALS space."""

import numpy as np

from .catalog import Catalog
from .profile import _norm, book_key

TOP_N = 20
MAX_PER_AUTHOR = 2  # diversity, at most 2 books by the same author


def retrieve(catalog: Catalog, fav_rows: list[int], blocked: set[str], pool: int) -> list[dict]:
    """Top candidate pool, most similar to the favourites first.

    The taste score is the rank within the pool mapped to 100..0: it expresses
    how close a candidate is to the reading history relative to the other
    candidates. Each entry also names the favourites that pulled it in.
    """
    favs = catalog.factors[fav_rows]  # (k, factors), rows are L2-normalized
    sims = catalog.factors @ favs.T  # (books, k) cosine similarities
    agg = sims.sum(axis=1)
    agg[fav_rows] = -np.inf

    blocked = set(blocked)  # local copy, also blocks duplicate titles within the pool
    entries: list[dict] = []
    for row in np.argsort(-agg):
        if len(entries) >= pool:
            break
        book = catalog.books[row]
        key = book_key(book["title"], book["author"])
        if not key or key in blocked:
            continue
        blocked.add(key)
        anchors = np.argsort(-sims[row])[:2]
        entries.append(
            {
                "row": int(row),
                "title": book["title"],
                "author": book["author"],
                "year": book.get("year"),
                "avg": book.get("avg"),
                "count": book.get("count", 0),
                "because": [fav_rows[a] for a in anchors],
            }
        )

    span = max(len(entries) - 1, 1)
    for i, entry in enumerate(entries):
        entry["taste"] = round(100 * (1 - i / span))
    return entries


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
