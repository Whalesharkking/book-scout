"""Reading profile: user maintained markdown with reading wish and read books."""

import hashlib
import json
import os
import re

TEMPLATE = """# Reading Profile

This file belongs to you. The agent reads it again on every iteration.
Changes (a new reading wish or new books) automatically trigger a fresh
scoring of both leaderboards.

## Current reading wish

Bullet points describing what you want to read right now.
Leave empty for general recommendations matching your taste.

- (empty = general)

## Books read

Books you have read and enjoyed. Type: `fach` (non-fiction) or `andere`
(everything else). They are never proposed again and steer what gets
recommended next.

| Title | Author | Type |
|-------|--------|------|
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
        self.read = read  # [{title, author, typ}], read and enjoyed
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
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if (
                len(cells) < 3
                or cells[0].lower() in ("titel", "title")
                or set(cells[0]) <= {"-", ":", " "}
            ):
                continue
            read.append(
                {
                    "title": cells[0],
                    "author": cells[1],
                    "typ": "fach" if cells[2].lower() == "fach" else "andere",
                }
            )
    return Profile(wish, read)
