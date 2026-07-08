"""Score computation: CF taste signal, Goodreads reader quality, LLM wish fit."""

# Weights when a reading wish is present (sum 1.0)
WISH_WEIGHT = 0.45  # LLM: fit to the current reading wish
TASTE_WEIGHT = 0.35  # CF: closeness to the read books in ALS space
QUALITY_WEIGHT = 0.20  # Goodreads reader ratings

# Weights without a reading wish (the taste signal carries the ranking)
NOWISH_TASTE_WEIGHT = 0.70
NOWISH_QUALITY_WEIGHT = 0.30

GR_PSEUDO_COUNT = 200  # damping, acts like 200 additional neutral votes
GR_NEUTRAL_AVG = 3.8  # roughly the Goodreads global average (scale 1 to 5)
QUALITY_MISSING = 50  # assumption for the rare book without ratings


def quality_score(avg: float | None, count: int | None) -> int:
    """Goodreads rating (1 to 5) mapped to 0 to 100, Bayes damped toward the
    global average so that thin ratings cannot dominate."""
    if not avg or not count:
        return QUALITY_MISSING
    shrunk = (avg * count + GR_NEUTRAL_AVG * GR_PSEUDO_COUNT) / (count + GR_PSEUDO_COUNT)
    return round((shrunk - 1) / 4 * 100)


def combine(taste: int, quality: int, wish_fit: int | None) -> int:
    """Final score 0 to 100. Without a wish (or when the LLM is unavailable)
    the wish share is redistributed to taste and quality."""
    if wish_fit is None:
        return round(taste * NOWISH_TASTE_WEIGHT + quality * NOWISH_QUALITY_WEIGHT)
    return round(wish_fit * WISH_WEIGHT + taste * TASTE_WEIGHT + quality * QUALITY_WEIGHT)
