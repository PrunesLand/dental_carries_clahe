"""
Statistical tests for repeated k-fold cross-validation results.

Uses the Nadeau-Bengio corrected paired t-test, which adjusts the variance
estimate to account for the non-independence of CV folds.  Holm correction
is applied across encoders by the caller (see report.py).
"""

import numpy as np
from scipy import stats

import config


def corrected_paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    n_train_folds: int | None = None,
    n_test_folds: int = 1,
) -> tuple[float, float, float]:
    """
    Nadeau-Bengio corrected resampled paired t-test.

    Args:
        scores_a: per-run scores for condition A (baseline).
        scores_b: per-run scores for condition B (CLAHE).
        n_train_folds: defaults to K_FOLDS - 1.
        n_test_folds: number of test folds (almost always 1).

    Returns:
        (t_statistic, two_sided_p_value, mean_difference)
    """
    if n_train_folds is None:
        n_train_folds = config.K_FOLDS - 1

    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = b - a
    n = len(d)
    mean_d, var_d = d.mean(), d.var(ddof=1)

    if var_d == 0:
        return 0.0, 1.0, float(mean_d)

    correction = (1.0 / n) + (n_test_folds / n_train_folds)
    t = mean_d / np.sqrt(correction * var_d)
    p = stats.t.sf(np.abs(t), n - 1) * 2
    return float(t), float(p), float(mean_d)


def corrected_ci(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
    n_train_folds: int | None = None,
    n_test_folds: int = 1,
) -> tuple[float, float]:
    """
    95% confidence interval for the mean difference using the same
    corrected standard error as corrected_paired_ttest.

    Returns:
        (lower_bound, upper_bound)
    """
    if n_train_folds is None:
        n_train_folds = config.K_FOLDS - 1

    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    d = b - a
    n = len(d)
    mean_d, var_d = d.mean(), d.var(ddof=1)
    se = np.sqrt(((1.0 / n) + (n_test_folds / n_train_folds)) * var_d)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(mean_d - tcrit * se), float(mean_d + tcrit * se)
