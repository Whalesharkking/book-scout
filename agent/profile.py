"""Leseprofil: vom Nutzer gepflegtes Markdown mit Lesewunsch und gelesenen Büchern."""

import hashlib
import json
import os
import re

TEMPLATE = """# Leseprofil

Dieses File gehört dir - der Agent liest es bei jeder Iteration neu ein.
Änderungen (neuer Lesewunsch, neue Bewertungen) führen automatisch dazu,
dass beide Bestenlisten neu bewertet werden.

## Aktueller Lesewunsch

Stichpunkte, welche Genres/Themen/Typen du gerade lesen möchtest.
Leer lassen = allgemeine Empfehlungen passend zu deinem Geschmack.

- (leer = allgemein)

## Gelesene Bücher

Bücher, die du gelesen und gut gefunden hast. Typ: `fach` oder `andere`.
Sie werden nie mehr vorgeschlagen und steuern, was dir künftig empfohlen
wird (mehr in diese Richtung).

| Titel | Autor | Typ |
|-------|-------|-----|
"""


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9äöüéèàêß ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def book_key(title: str, author: str) -> str:
    """Stabiler Schlüssel zum Deduplizieren, z.B. 'der marsianer | andy weir'."""
    t, a = _norm(title), _norm(author)
    return f"{t} | {a}" if t and a else ""


def norm_title(title: str) -> str:
    return _norm(title)


class Profile:
    def __init__(self, wish: list[str], read: list[dict]):
        self.wish = wish
        self.read = read  # [{title, author, typ}] - gelesen und gut gefunden
        payload = json.dumps({"wish": wish, "read": read}, ensure_ascii=False, sort_keys=True)
        self.hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def read_keys(self) -> set[str]:
        return {k for b in self.read if (k := book_key(b["title"], b["author"]))}

    def as_prompt_block(self) -> str:
        lines = ["Current reading mood (what the user wants to read right now):"]
        if self.wish:
            lines += [f"- {w}" for w in self.wish]
        else:
            lines.append(
                "- no specific wish right now: recommend broadly and generally, "
                "guided only by the reading history below"
            )
        lines.append("")
        lines.append("Books the user has read AND enjoyed (their taste - more like these):")
        if self.read:
            lines += [f'- "{b["title"]}" by {b["author"]} ({b["typ"]})' for b in self.read]
        else:
            lines.append("- (none entered yet)")
        return "\n".join(lines)


def ensure_template(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE)


def load(path: str) -> Profile:
    ensure_template(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    wish: list[str] = []
    read: list[dict] = []
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].lower()
            section = "wish" if "lesewunsch" in heading else "read" if "gelesen" in heading else None
            continue
        if section == "wish" and stripped.startswith("- "):
            item = stripped[2:].strip()
            # eingeklammerte Zeilen sind Platzhalter/Hinweise, kein Lesewunsch
            if item and not item.startswith("("):
                wish.append(item)
        elif section == "read" and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) < 3 or cells[0].lower() == "titel" or set(cells[0]) <= {"-", ":", " "}:
                continue
            read.append(
                {
                    "title": cells[0],
                    "author": cells[1],
                    "typ": "fach" if cells[2].lower() == "fach" else "andere",
                }
            )
    return Profile(wish, read)
