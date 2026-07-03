"""Endlos laufender Buch-Scout: empfiehlt Bücher passend zum Leseprofil.

Wechselt pro Iteration zwischen den Kategorien 'fach' und 'andere' ab:
Profil einlesen -> Kandidaten generieren -> Open-Library-Check ->
kritisch scoren -> Top-20 pflegen. Ändert sich das Leseprofil
(neuer Lesewunsch oder neue Bewertung), werden beide Listen neu bewertet.
"""

import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

from . import openlibrary, profile, state as state_mod
from .llm import Ollama
from .profile import book_key
from .state import State

DATA_DIR = os.environ.get("DATA_DIR", "/data")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "gemma3:12b")
ITERATION_SLEEP = float(os.environ.get("ITERATION_SLEEP", "120"))
SCORER_PASSES = int(os.environ.get("SCORER_PASSES", "2"))  # Scoring-Durchläufe, gemittelt
LOOKUP_SLEEP = 1.0  # Pause zwischen Open-Library-Anfragen (Rate-Limit-Höflichkeit)
RESCORE_CYCLE = 24  # alle 24 Iterationen wird jede Liste einmal neu bewertet

CATEGORIES = ["fach", "andere"]
CATEGORY_TITLES = {"fach": "Fachbücher", "andere": "Andere Bücher"}

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


def write_markdown(state: State) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    checked = len(state.seen)
    for category in CATEGORIES:
        lines = [
            f"# Top {len(state.lists[category])} Buchempfehlungen – {CATEGORY_TITLES[category]}",
            "",
            f"_Stand: {stamp} · Iteration {state.iteration} · insgesamt geprüft: {checked} Bücher_",
            "",
            "Gelesen und gut gefunden? Trage das Buch in `leseprofil.md` ein –",
            "es verschwindet dann aus der Liste und schärft künftige Empfehlungen.",
            "",
            "Score-Detail: W = Passung Lesewunsch, G = Geschmacksnähe,",
            "Q = Qualität/Renommee, OL = Open-Library-Leserbewertung.",
            "",
            "| # | Titel | Autor | Jahr | Sprache | Score | Detail | Begründung |",
            "|--:|-------|-------|-----:|:-------:|------:|--------|------------|",
        ]
        for i, e in enumerate(state.lists[category], 1):
            year = e.get("year") or "?"
            dims = e.get("dims")
            if dims:
                detail = f"W{dims['wish_fit']} G{dims['taste_fit']} Q{dims['quality']}"
                if e.get("ol_score") is not None:
                    detail += f" OL{e['ol_score']}"
            else:
                detail = "–"
            lines.append(
                f"| {i} | {e['title']} | {e['author']} | {year} "
                f"| {e.get('language', '?')} | {e['score']} | {detail} | {e['reason']} |"
            )
        lines.append("")
        path = os.path.join(DATA_DIR, f"top_{category}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def apply_profile(state: State, prof: profile.Profile) -> None:
    """Gelesene Bücher als geprüft markieren und aus den Listen entfernen."""
    read_keys = prof.read_keys()
    for key in read_keys:
        state.seen[key] = "read"
    for category in CATEGORIES:
        kept = [e for e in state.lists[category] if e["key"] not in read_keys]
        for e in state.lists[category]:
            if e["key"] in read_keys:
                log.info("[%s] '%s' wurde gelesen – aus der Liste entfernt", category, e["title"])
        state.lists[category] = kept


def rescore_list(llm: Ollama, state: State, prof: profile.Profile, category: str) -> None:
    """Bestehende Liste mit dem aktuellen Profil frisch bewerten, damit die
    Kalibrierung stimmt und Profil-Änderungen sich sofort auswirken."""
    entries = state.lists[category]
    if not entries:
        return
    scored = llm.score(category, prof.as_prompt_block(), entries, passes=SCORER_PASSES)
    if scored:
        # nicht zugeordnete Einträge behalten ihren alten Score statt zu verschwinden
        scored_keys = {e["key"] for e in scored}
        kept = [e for e in entries if e["key"] not in scored_keys]
        state.lists[category] = state_mod.rank(scored + kept)
    log.info("[%s] Liste neu bewertet (%d Einträge)", category, len(state.lists[category]))


def run_iteration(llm: Ollama, state: State, prof: profile.Profile, category: str) -> None:
    ranking = state.lists[category]
    profile_block = prof.as_prompt_block()
    candidates = llm.generate_candidates(category, profile_block, ranking, state.avoid_sample())

    fresh = []
    for cand in candidates:
        key = book_key(cand["title"], cand["author"])
        if key and key not in state.seen and key not in {f["key"] for f in fresh}:
            fresh.append({**cand, "key": key})

    verified = []
    for cand in fresh:
        status, info = openlibrary.lookup(cand["title"], cand["author"])
        if status == "found":
            state.seen[cand["key"]] = "found"
            entry = {
                "key": cand["key"],
                "title": info["title"],
                "author": info["author"],
                "year": info["year"],
                "language": cand.get("language", "?"),
                "ratings_average": info.get("ratings_average"),
                "ratings_count": info.get("ratings_count"),
            }
            # kanonische Schreibweise kann vom Vorschlag abweichen -> beide sperren
            state.seen.setdefault(book_key(entry["title"], entry["author"]), "found")
            verified.append(entry)
        elif status == "notfound":
            state.seen[cand["key"]] = "notfound"
        # bei "error" nichts merken, damit der Kandidat später erneut geprüft wird
        time.sleep(LOOKUP_SLEEP)

    added = 0
    best_new = None
    if verified:
        scored = llm.score(category, profile_block, verified, passes=SCORER_PASSES)
        # Bücher, die keine Bewertung bekamen (Fehler/Titel nicht zugeordnet),
        # wieder freigeben, damit sie später erneut vorgeschlagen werden können
        scored_keys = {e["key"] for e in scored}
        for entry in verified:
            if entry["key"] not in scored_keys:
                state.seen.pop(entry["key"], None)
                state.seen.pop(book_key(entry["title"], entry["author"]), None)
        if scored:
            best_new = max(e["score"] for e in scored)
            added = state.merge(category, scored)

    top = state.lists[category][0] if state.lists[category] else None
    log.info(
        "[%s] it=%d vorgeschlagen=%d neu=%d existiert=%d aufgenommen=%d bester_neuer=%s top1=%s",
        category,
        state.iteration,
        len(candidates),
        len(fresh),
        len(verified),
        added,
        best_new if best_new is not None else "-",
        f"{top['title']}({top['score']})" if top else "-",
    )


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging()
    log.info("Starte Buch-Scout (Modell %s, Ollama %s)", MODEL, OLLAMA_HOST)

    llm = Ollama(OLLAMA_HOST, MODEL)
    llm.wait_ready()
    llm.ensure_model()

    profile_path = os.path.join(DATA_DIR, "leseprofil.md")
    state = State(os.path.join(DATA_DIR, "state.json"))
    log.info(
        "Zustand geladen: Iteration %d, %d geprüfte Bücher", state.iteration, len(state.seen)
    )

    while True:
        category = CATEGORIES[state.iteration % 2]
        try:
            prof = profile.load(profile_path)
            apply_profile(state, prof)
            if prof.hash != state.profile_hash:
                if state.profile_hash:
                    log.info("Leseprofil geändert – beide Listen werden neu bewertet")
                    for cat in CATEGORIES:
                        rescore_list(llm, state, prof, cat)
                state.profile_hash = prof.hash
            elif state.iteration % RESCORE_CYCLE >= RESCORE_CYCLE - 2 and state.iteration > 0:
                rescore_list(llm, state, prof, category)
            run_iteration(llm, state, prof, category)
        except Exception as exc:
            log.warning("Iteration %d übersprungen: %s", state.iteration, exc)
        state.iteration += 1
        state.save()
        write_markdown(state)
        time.sleep(ITERATION_SLEEP)


if __name__ == "__main__":
    main()
