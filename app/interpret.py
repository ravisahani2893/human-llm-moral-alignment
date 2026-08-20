_EDGES = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
_LABELS = [
    "Extremely negative",
    "Strongly negative",
    "Moderately negative",
    "Slightly negative",
    "Slightly positive",
    "Moderately positive",
    "Strongly positive",
    "Extremely positive",
]


def valence_label(value: float | None) -> str | None:
    """
    Map a continuous -1..+1 valence score to a human-readable band label,
    using the same anchor scale described in the scoring prompt.
    """
    if value is None:
        return None
    if value == 0:
        return "Neutral"
    for i in range(len(_EDGES) - 1):
        low, high = _EDGES[i], _EDGES[i + 1]
        if low < value <= high:
            return _LABELS[i]
    return "Extremely negative" if value < -1 else "Extremely positive"


_AGREEMENT_BANDS = [
    (0.20, "Very weak"),
    (0.40, "Weak"),
    (0.60, "Moderate"),
    (0.70, "Moderately Strong"),
    (0.80, "Strong"),
]


def agreement_label(value: float | None) -> str | None:
    """
    Map a CCC (or Pearson/Spearman, shown alongside CCC in the same
    tables) value to a human-readable strength-of-agreement band:

        CCC < 0            No agreement
        0.00 <= CCC < 0.20  Very weak
        0.20 <= CCC < 0.40  Weak
        0.40 <= CCC < 0.60  Moderate
        0.60 <= CCC < 0.70  Moderately Strong
        0.70 <= CCC < 0.80  Strong
        CCC >= 0.80         Very Strong

    A negative value is labelled "No agreement" outright rather than
    banded by magnitude — for an agreement coefficient, the sign itself
    (systematic disagreement vs. no relationship) matters more than how
    negative it is.
    """
    if value is None:
        return None
    if value < 0:
        return "No agreement"
    for cutoff, label in _AGREEMENT_BANDS:
        if value < cutoff:
            return label
    return "Very Strong"


def error_label(value: float | None) -> str | None:
    """
    Map an MAE/RMSE value (error on the -1..+1 valence scale, so a
    maximum possible error of 2) to a human-readable band. Thresholds are
    a plain, round-number split of that range (roughly 15%/25% of it),
    not a validated statistical convention.
    """
    if value is None:
        return None
    if value < 0.3:
        return "Low error"
    if value < 0.5:
        return "Moderate error"
    return "High error"
