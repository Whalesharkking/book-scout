"""Existenz-Check für Buchvorschläge über die Open-Library-Suche.

Verhindert, dass halluzinierte Titel in die Listen gelangen, und liefert
kanonische Schreibweise plus Erscheinungsjahr.
"""

import logging

import requests

from .profile import _norm

log = logging.getLogger("agent")

API = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "book-scanner/1.0 (privates Hobby-Projekt)"}


def _title_matches(proposed: str, found: str) -> bool:
    a, b = _norm(proposed), _norm(found)
    return bool(a and b) and (a in b or b in a)


def _author_matches(proposed: str, names: list[str]) -> bool:
    lastname = _norm(proposed).split()[-1] if _norm(proposed) else ""
    return any(lastname and lastname in _norm(n) for n in names)


def lookup(title: str, author: str) -> tuple[str, dict | None]:
    """Gibt ('found', {title, author, year}), ('notfound', None) oder ('error', None) zurück.

    'found' liefert die kanonische Schreibweise aus Open Library.
    Bei 'error' (Netzwerk/Rate-Limit) wird der Kandidat nicht als geprüft
    markiert und kann später erneut vorgeschlagen werden.
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

        # Übersetzte Ausgaben (z.B. deutsche Titel) stehen bei Open Library unter
        # dem kanonischen Werktitel -> Freitext-Suche, nur der Autor muss stimmen.
        resp = requests.get(
            API,
            params={"q": f"{title} {author}", "limit": 3, "fields": fields},
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("Open-Library-Abfrage fehlgeschlagen (%s): %s", title, exc)
        return "error", None

    for doc in docs:
        names = doc.get("author_name") or []
        if _author_matches(author, names):
            return "found", {
                "title": title,  # vorgeschlagenen (z.B. deutschen) Titel behalten
                "author": names[0] if names else author,
                "year": doc.get("first_publish_year"),
                "ratings_average": doc.get("ratings_average"),
                "ratings_count": doc.get("ratings_count"),
            }
    return "notfound", None
