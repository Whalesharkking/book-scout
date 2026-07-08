"""Book scout: collaborative-filtering recommendations with an LLM wish re-ranker.

Computes both top 20 lists from the reading profile and exits - run it again
after editing the profile. It matches the read books against the Goodreads
catalog, pulls the nearest neighbours in ALS space as candidates and blends
the taste signal with real Goodreads reader ratings and - when a reading
wish is present - with an LLM rating for wish fit. Requires the index built
by scripts/build_index.py.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from . import profile, recommend, scoring
from .catalog import Catalog
from .llm import Ollama
from .profile import book_key, norm_title

DATA_DIR = os.environ.get("DATA_DIR", "/data")
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(DATA_DIR, "index"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "")  # empty = no wish re-ranking
MODEL = os.environ.get("MODEL", "gemma3:12b")
SCORER_PASSES = int(os.environ.get("SCORER_PASSES", "2"))  # re-ranker passes, averaged
POOL = int(os.environ.get("POOL", "200"))  # candidates per list before re-ranking

CATEGORY_TITLES = {"nonfiction": "Non-Fiction", "other": "Other Books"}

log = logging.getLogger("agent")


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(
        os.path.join(DATA_DIR, "agent.log"), maxBytes=2_000_000, backupCount=2
    )
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def write_text(path: str, text: str) -> None:
    """Atomic write so a crash never leaves a half-written file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def match_profile(catalog: Catalog, prof: profile.Profile):
    """Maps the read books to catalog rows; unmatched ones still get blocked."""
    fav_rows, matched, unmatched = [], [], []
    blocked = prof.read_keys()
    for book in prof.read:
        row = catalog.match(book["title"], book["author"])
        if row is None:
            unmatched.append(book)
            continue
        entry = catalog.books[row]
        fav_rows.append(row)
        matched.append({**book, "catalog_title": entry["title"], "catalog_author": entry["author"]})
        blocked.add(book_key(entry["title"], entry["author"]))
    return fav_rows, matched, unmatched, blocked


def write_match_report(matched: list[dict], unmatched: list[dict]) -> None:
    lines = [
        "# Profile Matching",
        "",
        "How your read books were matched against the Goodreads catalog.",
        "Unmatched books cannot steer the recommendations - usually the",
        "spelling differs; try the English original title.",
        "",
        f"## Matched ({len(matched)})",
        "",
        "| Your entry | Matched catalog book |",
        "|------------|----------------------|",
    ]
    lines += [
        f"| {b['title']} — {b['author']} | {b['catalog_title']} — {b['catalog_author']} |"
        for b in matched
    ]
    lines += ["", f"## Not matched ({len(unmatched)})", ""]
    lines += [f"- {b['title']} — {b['author']}" for b in unmatched] or ["- (none)"]
    lines.append("")
    write_text(os.path.join(DATA_DIR, "profile-match.md"), "\n".join(lines))


def write_markdown(lists: dict, catalog: Catalog, matched: int, total: int, note: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for category, entries in lists.items():
        lines = [
            f"# Top {len(entries)} Book Recommendations: {CATEGORY_TITLES[category]}",
            "",
            f"_Updated: {stamp} · catalog: {len(catalog)} books · "
            f"profile: {matched} of {total} read books matched (see profile-match.md)_",
            "",
            "Read one of these and enjoyed it? Add it to `reading-profile.md`.",
            "It then disappears from the list and sharpens future recommendations.",
            "",
            "Score detail: T = taste closeness to your read books, "
            "Q = Goodreads reader quality, W = wish fit.",
        ]
        if note:
            lines += ["", f"**Note:** {note}"]
        lines += [
            "",
            "| # | Title | Author | Year | Score | Detail | Because you liked | Wish |",
            "|--:|-------|--------|-----:|------:|--------|-------------------|------|",
        ]
        for i, e in enumerate(entries, 1):
            detail = f"T{e['taste']} Q{e['quality']}"
            if e.get("wish_fit") is not None:
                detail += f" W{e['wish_fit']}"
            lines.append(
                f"| {i} | {e['title']} | {e['author']} | {e.get('year') or '?'} "
                f"| {e['score']} | {detail} | {e['because']} | {e.get('reason', '')} |"
            )
        lines.append("")
        write_text(os.path.join(DATA_DIR, f"top_{category}.md"), "\n".join(lines))


def wish_scores_for(prof: profile.Profile, pools: dict) -> tuple[dict, str]:
    """LLM wish fit per candidate; degrades gracefully to pure CF ranking."""
    if not prof.wish:
        return {}, ""
    if not OLLAMA_HOST:
        return {}, "A reading wish is set but no LLM is configured (OLLAMA_HOST); ranked by taste and quality only."
    llm = Ollama(OLLAMA_HOST, MODEL)
    try:
        llm.wait_ready()
        llm.ensure_model()
    except Exception as exc:  # keep serving without the LLM
        log.warning("LLM unavailable, ranking without wish fit: %s", exc)
        return {}, "LLM unreachable; ranked by taste and quality only."
    wish_block = "\n".join(f"- {w}" for w in prof.wish)
    scores = {}
    for category, entries in pools.items():
        log.info("[%s] re-ranking %d candidates against the reading wish", category, len(entries))
        try:
            scores.update(llm.score_wish(wish_block, entries, SCORER_PASSES))
        except Exception as exc:  # degrade to CF ranking instead of failing the rebuild
            log.warning("Re-ranking failed for %s: %s", category, exc)
            return scores, "LLM re-ranking failed partway; ranked (partly) by taste and quality only."
    return scores, ""


def rebuild(catalog: Catalog, prof: profile.Profile) -> None:
    fav_rows, matched, unmatched, blocked = match_profile(catalog, prof)
    write_match_report(matched, unmatched)
    for book in unmatched:
        log.info("profile book not in catalog: '%s' by %s", book["title"], book["author"])
    if not fav_rows:
        note = (
            "No read book could be matched against the catalog yet, so there is "
            "nothing to learn your taste from. Fill the 'Books read' table in "
            "reading-profile.md (English original titles match best)."
        )
        write_markdown({c: [] for c in recommend.CATEGORIES}, catalog, 0, len(prof.read), note)
        return

    pools = recommend.retrieve(catalog, fav_rows, blocked, POOL)
    wish_scores, note = wish_scores_for(prof, pools)

    lists = {}
    for category, entries in pools.items():
        for e in entries:
            wish = wish_scores.get(norm_title(e["title"]))
            e["quality"] = scoring.quality_score(e["avg"], e["count"])
            e["wish_fit"] = wish[0] if wish else None
            e["reason"] = wish[1] if wish else ""
            e["score"] = scoring.combine(e["taste"], e["quality"], e["wish_fit"])
            e["because"] = "; ".join(catalog.books[r]["title"] for r in e["because"])
        lists[category] = recommend.rank(entries)
        top = lists[category][0] if lists[category] else None
        log.info(
            "[%s] list rebuilt, top1=%s",
            category,
            f"{top['title']}({top['score']})" if top else "-",
        )
    write_markdown(lists, catalog, len(matched), len(prof.read), note)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging()
    try:
        catalog = Catalog(INDEX_DIR)
    except (FileNotFoundError, ValueError) as exc:
        log.error(
            "No recommendation index at %s (%s). Build it once:\n"
            "  sh scripts/download_data.sh data/goodreads\n"
            "  python -m scripts.build_index --data-dir data",
            INDEX_DIR,
            exc,
        )
        sys.exit(1)
    log.info("Catalog loaded: %d books, %d factors", len(catalog), catalog.factors.shape[1])

    prof = profile.load(os.path.join(DATA_DIR, "reading-profile.md"))
    log.info("Profile: %d read books, %d wish lines", len(prof.read), len(prof.wish))
    rebuild(catalog, prof)


if __name__ == "__main__":
    main()
