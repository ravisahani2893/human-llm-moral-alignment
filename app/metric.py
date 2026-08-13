import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr


def calculate_ccc(human_scores, model_scores):
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC).

    Parameters
    ----------
    human_scores : array-like
        Ground truth human annotations.

    model_scores : array-like
        Predicted scores from the LLM.

    Returns
    -------
    float
        Lin's Concordance Correlation Coefficient.
    """

    human = np.asarray(human_scores, dtype=float)
    model = np.asarray(model_scores, dtype=float)

    if len(human) != len(model):
        raise ValueError(
            "Human and model arrays must have the same length."
        )

    if len(human) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    # Pearson correlation
    pearson_corr, _ = stats.pearsonr(human, model)

    # Means
    mean_human = np.mean(human)
    mean_model = np.mean(model)

    # Variances
    var_human = np.var(human, ddof=1)
    var_model = np.var(model, ddof=1)

    # Standard deviations
    std_human = np.sqrt(var_human)
    std_model = np.sqrt(var_model)

    # CCC
    numerator = 2 * pearson_corr * std_human * std_model

    denominator = (
        var_human
        + var_model
        + (mean_human - mean_model) ** 2
    )

    return numerator / denominator


def calculate_mae(human_scores, model_scores):
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    human_scores : array-like
        Human annotations.

    model_scores : array-like
        Model predictions.

    Returns
    -------
    float
        Mean Absolute Error.
    """

    human = np.asarray(human_scores, dtype=float)
    model = np.asarray(model_scores, dtype=float)

    if len(human) != len(model):
        raise ValueError(
            "Human and model arrays must have the same length."
        )

    return np.mean(np.abs(human - model))


def calculate_rmse(human_scores, model_scores):
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    human_scores : array-like
        Human annotations.

    model_scores : array-like
        Model predictions.

    Returns
    -------
    float
        Root Mean Squared Error.
    """

    human = np.asarray(human_scores, dtype=float)
    model = np.asarray(model_scores, dtype=float)

    if len(human) != len(model):
        raise ValueError(
            "Human and model arrays must have the same length."
        )

    mse = np.mean((human - model) ** 2)

    return np.sqrt(mse)

def calculate_pearson(human_scores, model_scores):
    """
    Calculate Pearson Correlation Coefficient.

    Parameters
    ----------
    human_scores : array-like
        Human annotations.

    model_scores : array-like
        Model predictions.

    Returns
    -------
    float
        Pearson Correlation Coefficient.
    """

    human = np.asarray(human_scores, dtype=float)
    model = np.asarray(model_scores, dtype=float)

    if len(human) != len(model):
        raise ValueError(
            "Human and model arrays must have the same length."
        )

    correlation, _ = pearsonr(human, model)

    return correlation


def calculate_spearman(human_scores, model_scores):
    """
    Calculate Spearman Rank Correlation Coefficient.

    Parameters
    ----------
    human_scores : array-like
        Human annotations.

    model_scores : array-like
        Model predictions.

    Returns
    -------
    float
        Spearman Rank Correlation Coefficient.
    """

    human = np.asarray(human_scores, dtype=float)
    model = np.asarray(model_scores, dtype=float)

    if len(human) != len(model):
        raise ValueError(
            "Human and model arrays must have the same length."
        )

    correlation, _ = spearmanr(human, model)

    return correlation


def calculate_wilcoxon(scores_a, scores_b):
    """
    Wilcoxon signed-rank test on paired differences (scores_b - scores_a).

    Used for the bias/variant perturbation study, not the human-model
    alignment metrics above: a paired, non-parametric test is appropriate
    there because sample sizes are small (a few dozen scenario pairs) and
    valence deltas aren't assumed to be normally distributed.

    Parameters
    ----------
    scores_a, scores_b : array-like
        Paired scores (e.g. model valence on variant A vs. variant B of the
        same scenario, same order/ID alignment already guaranteed by caller).

    Returns
    -------
    dict with "statistic", "p_value", "mean_delta", "median_delta", "n" —
    or all-None (except n) if every pair is tied (scipy.stats.wilcoxon
    raises on an all-zero difference vector, which is a legitimate result,
    not an error).
    """

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the same length.")
    if len(a) < 1:
        raise ValueError("At least one paired observation is required.")

    delta = b - a
    n = len(delta)

    if np.all(delta == 0):
        return {"statistic": None, "p_value": None, "mean_delta": 0.0, "median_delta": 0.0, "n": n}

    statistic, p_value = stats.wilcoxon(a, b)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "n": n,
    }
