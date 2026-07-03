"""Existence check for book proposals via the Open Library search API.

Prevents hallucinated titles from reaching the lists and provides the
canonical spelling, the publication year and reader ratings.
"""

import logging

import requests

from .profile import _norm

log = logging.getLogger("agent")

API = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "book-scout/1.0 (private hobby project)"}


def _title_matches(proposed: str, found: str) -> bool:
    a, b = _norm(proposed), _norm(found)
    return bool(a and b) and (a in b or b in a)


def _author_matches(proposed: str, names: list[str]) -> bool:
    lastname = _norm(proposed).split()[-1] if _norm(proposed) else ""
    return any(lastname and lastname in _norm(n) for n in names)


def lookup(title: str, author: str) -> tuple[str, dict | None]:
    """Returns ('found', info), ('notfound', None) or ('error', None).

    'found' carries the canonical spelling from Open Library.
    On 'error' (network or rate limit) the candidate is not marked as
    checked and can be proposed again later.
    """
    fields = "title,author_name,first_publish_year,ratings_average,ratings_count"
    try:
        resp = requests.get(
            API,
            params={"title": title, "author": author, "limit": 5, "fields": fields},
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs") or []

        for doc in docs:
            names = doc.get("author_name") or []
            if _title_matches(title, doc.get("title", "")) and _author_matches(author, names):
                return "found", {
                    "title": doc.get("title") or title,
                    "author": names[0] if names else author,
                    "year": doc.get("first_publish_year"),
                    "ratings_average": doc.get("ratings_average"),
                    "ratings_count": doc.get("ratings_count"),
                }

        # Translated editions (for example German titles) are indexed under
        # the canonical work title. Fall back to a free text search where
        # only the author has to match.
        resp = requests.get(
            API,
            params={"q": f"{title} {author}", "limit": 3, "fields": fields},
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("Open Library request failed (%s): %s", title, exc)
        return "error", None

    for doc in docs:
        names = doc.get("author_name") or []
        if _author_matches(author, names):
            return "found", {
                "title": title,  # keep the proposed (for example German) title
                "author": names[0] if names else author,
                "year": doc.get("first_publish_year"),
                "ratings_average": doc.get("ratings_average"),
                "ratings_count": doc.get("ratings_count"),
            }
    return "notfound", None
