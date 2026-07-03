"""Score computation: weighted LLM dimensions plus the Open Library reader signal."""

# Weights of the LLM dimensions (sum 1.0)
WISH_WEIGHT = 0.45  # fit to the current reading wish
TASTE_WEIGHT = 0.35  # taste kinship with the read books
QUALITY_WEIGHT = 0.20  # quality and reputation according to the LLM

OL_WEIGHT = 0.15  # share of the Open Library reader signal in the final score
OL_PSEUDO_COUNT = 20  # damping, acts like 20 additional neutral votes
OL_NEUTRAL_AVG = 3.5  # neutral average (scale 1 to 5)
OL_MISSING_SCORE = 55  # assumption without reader ratings, slightly below neutral (62)

DIMENSIONS = ("wish_fit", "taste_fit", "quality")


def llm_score(dims: dict) -> float:
    return (
        dims["wish_fit"] * WISH_WEIGHT
        + dims["taste_fit"] * TASTE_WEIGHT
        + dims["quality"] * QUALITY_WEIGHT
    )


def openlibrary_score(avg: float | None, count: int | None) -> int | None:
    """Reader rating (1 to 5) mapped to 0 to 100, with Bayes damping toward
    neutral so that a few single votes cannot dominate the result."""
    if not avg or not count:
        return None
    shrunk = (avg * count + OL_NEUTRAL_AVG * OL_PSEUDO_COUNT) / (count + OL_PSEUDO_COUNT)
    return round((shrunk - 1) / 4 * 100)


def combine(dims: dict, ol_avg: float | None, ol_count: int | None) -> tuple[int, int | None]:
    """Final score 0 to 100 from LLM dimensions and the Open Library signal.

    Books without reader ratings are not excluded but receive a small
    penalty (OL_MISSING_SCORE) so that provably good books rank first.
    """
    base = llm_score(dims)
    ol = openlibrary_score(ol_avg, ol_count)
    blend = OL_MISSING_SCORE if ol is None else ol
    return round(base * (1 - OL_WEIGHT) + blend * OL_WEIGHT), ol
