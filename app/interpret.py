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
    (0.80, "Strong"),
    (1.01, "Very strong"),
]


def agreement_label(value: float | None) -> str | None:
    """
    Map a correlation-type coefficient (CCC, Pearson, or Spearman — all
    range roughly -1..+1) to a human-readable strength-of-agreement band,
    using a standard general-purpose correlation-strength convention
    (commonly cited thresholds of 0.2/0.4/0.6/0.8 for very weak/weak/
    moderate/strong/very strong). This is a general interpretive
    convention, not a metric-specific or domain-validated threshold —
    label it as such wherever it's shown.
    """
    if value is None:
        return None
    magnitude = min(abs(value), 1.0)
    for cutoff, label in _AGREEMENT_BANDS:
        if magnitude < cutoff:
            return f"{label} (inverse)" if value < 0 else label
    return "Very strong (inverse)" if value < 0 else "Very strong"


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
