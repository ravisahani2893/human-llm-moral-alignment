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
