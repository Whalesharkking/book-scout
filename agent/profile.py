"""Reading profile: user maintained markdown with reading wish and read books."""

import hashlib
import json
import os
import re

TEMPLATE = """# Reading Profile

This file belongs to you. Enter your data, then run the agent once to
(re)compute the recommendation list.

## Current reading wish

Bullet points describing what you want to read right now. This is the only
part that uses the LLM. Leave empty for pure taste-based recommendations.

- (empty = general)

## Books read

Books you have read and enjoyed. They are never proposed again and steer
what gets recommended next. English original titles match best.

| Title | Author |
|-------|--------|
"""


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9äöüéèàêß ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def book_key(title: str, author: str) -> str:
    """Stable key for deduplication, for example 'der marsianer | andy weir'."""
    t, a = _norm(title), _norm(author)
    return f"{t} | {a}" if t and a else ""


def norm_title(title: str) -> str:
    return _norm(title)


class Profile:
    def __init__(self, wish: list[str], read: list[dict]):
        self.wish = wish
        self.read = read  # [{title, author}], read and enjoyed
        payload = json.dumps({"wish": wish, "read": read}, ensure_ascii=False, sort_keys=True)
        self.hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def read_keys(self) -> set[str]:
        return {k for b in self.read if (k := book_key(b["title"], b["author"]))}


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
            # German and English headings are both accepted
            if "lesewunsch" in heading or "wish" in heading or "mood" in heading:
                section = "wish"
            elif "gelesen" in heading or "read" in heading:
                section = "read"
            else:
                section = None
            continue
        if section == "wish" and stripped.startswith("- "):
            item = stripped[2:].strip()
            # lines in parentheses are placeholders, not a reading wish
            if item and not item.startswith("("):
                wish.append(item)
        elif section == "read" and stripped.startswith("|"):
            # extra columns (for example the old Type column) are ignored
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if (
                len(cells) < 2
                or cells[0].lower() in ("titel", "title")
                or set(cells[0]) <= {"-", ":", " "}
            ):
                continue
            read.append({"title": cells[0], "author": cells[1]})
    return Profile(wish, read)
