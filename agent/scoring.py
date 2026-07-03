"""Score-Berechnung: gewichtete LLM-Dimensionen + Open-Library-Lesersignal."""

# Gewichtung der LLM-Dimensionen (Summe 1.0)
WISH_WEIGHT = 0.45  # Passung zum aktuellen Lesewunsch
TASTE_WEIGHT = 0.35  # Geschmacksnähe zu den gelesenen Büchern
QUALITY_WEIGHT = 0.20  # Qualität/Renommee laut LLM

OL_WEIGHT = 0.15  # Anteil des Open-Library-Lesersignals am Endscore
OL_PSEUDO_COUNT = 20  # Dämpfung: wirkt wie 20 neutrale Zusatz-Bewertungen
OL_NEUTRAL_AVG = 3.5  # neutraler Durchschnitt (Skala 1-5)
OL_MISSING_SCORE = 55  # Annahme ohne Leserbewertungen: leicht unter neutral (62)

DIMENSIONS = ("wish_fit", "taste_fit", "quality")


def llm_score(dims: dict) -> float:
    return (
        dims["wish_fit"] * WISH_WEIGHT
        + dims["taste_fit"] * TASTE_WEIGHT
        + dims["quality"] * QUALITY_WEIGHT
    )


def openlibrary_score(avg: float | None, count: int | None) -> int | None:
    """Leserbewertung (1-5) zu 0-100, mit Bayes-Dämpfung Richtung Neutral,
    damit wenige Einzelstimmen das Ergebnis nicht dominieren."""
    if not avg or not count:
        return None
    shrunk = (avg * count + OL_NEUTRAL_AVG * OL_PSEUDO_COUNT) / (count + OL_PSEUDO_COUNT)
    return round((shrunk - 1) / 4 * 100)


def combine(dims: dict, ol_avg: float | None, ol_count: int | None) -> tuple[int, int | None]:
    """Endscore 0-100 aus LLM-Dimensionen und Open-Library-Signal.

    Bücher ohne Leserbewertungen fliegen nicht raus, bekommen aber einen
    leichten Malus (OL_MISSING_SCORE), damit belegt gute Bücher vorne liegen.
    """
    base = llm_score(dims)
    ol = openlibrary_score(ol_avg, ol_count)
    blend = OL_MISSING_SCORE if ol is None else ol
    return round(base * (1 - OL_WEIGHT) + blend * OL_WEIGHT), ol
