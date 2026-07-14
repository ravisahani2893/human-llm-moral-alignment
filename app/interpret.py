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
