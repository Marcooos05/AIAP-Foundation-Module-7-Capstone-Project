"""
Advanced models for diabetes risk classification, compared against the
Logistic Regression baseline in `src.models`.

Two architectures, each distinct from the baseline and from each other:

- **Random Forest** (bagging): many deep trees, each fitted independently on a
  bootstrap sample with random feature subsets, then averaged. Averaging
  reduces *variance*.
- **Gradient Boosting** (`HistGradientBoostingClassifier`, boosting): shallow
  trees fitted one after another, each correcting the errors of the ones
  before. Adding trees in sequence reduces *bias*.

Unlike the linear baseline, both are non-linear: they find thresholds and
interactions on their own, so they use the raw `trees` feature variant
without engineered features. Both use `class_weight='balanced'` and the same
tuning, threshold and evaluation machinery as the baseline (`ModelTraining`),
so the comparison is like for like.
"""

import logging
from typing import Any, Callable, Dict

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.models import ModelTraining

logger = logging.getLogger(__name__)


class AdvancedModelTraining(ModelTraining):
    """
    Trains and tunes the advanced models (Random Forest, Gradient Boosting) on
    both glucose variants. Pipeline building, GridSearchCV tuning, threshold
    tuning and test evaluation are inherited from `ModelTraining`, so every
    model sees the same CV folds and is judged by the same rules.
    """

    MODEL_NAMES = ('random_forest', 'gradient_boosting')

    def _classifier_factories(self) -> Dict[str, Callable[[], Any]]:
        """Model name -> factory for an unfitted, class-balanced ensemble classifier."""
        random_state = self.config['random_state']
        return {
            'random_forest': lambda: RandomForestClassifier(class_weight='balanced', random_state=random_state),
            'gradient_boosting': lambda: HistGradientBoostingClassifier(
                class_weight='balanced', random_state=random_state,
            ),
        }


def compare_to_baseline(
    results: Dict[str, Dict[str, Any]],
    baseline_key: str,
    metric: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compares each model's per-fold CV scores with the baseline's on the same
    folds. Pairing removes the fold-to-fold noise that both models share, so
    it is a fairer test than comparing two means with overlapping SDs.

    The paired t-test p-value is indicative only: with k folds the training
    sets overlap, so the fold scores are not independent and the test is
    optimistic (Dietterich, 1998).

    Args:
        results (Dict[str, Dict[str, Any]]): CV results from `ModelTraining.tune_models`
            (each with `fold_scores`), including the baseline.
        baseline_key (str): Key of the baseline result.
        metric (str): CV metric to compare.
    Returns:
        Dict[str, Dict[str, Any]]: Model key -> mean and SD of the paired
            difference (model - baseline), folds improved, and p-value.
    Raises:
        ValueError: If the baseline is missing or the fold counts differ.
    """
    if baseline_key not in results:
        raise ValueError(f"Baseline '{baseline_key}' not found among {list(results)}.")
    baseline = np.asarray(results[baseline_key]['fold_scores'][metric])

    comparison = {}
    for key, result in results.items():
        if key == baseline_key:
            continue
        scores = np.asarray(result['fold_scores'][metric])
        if scores.shape != baseline.shape:
            raise ValueError(f"'{key}' has {len(scores)} folds but the baseline has {len(baseline)}.")
        diff = scores - baseline
        p_value = float(stats.ttest_rel(scores, baseline).pvalue) if np.any(diff != 0) else 1.0
        comparison[key] = {
            'mean_diff': float(diff.mean()),
            'std_diff': float(diff.std(ddof=1)),
            'n_folds_improved': int((diff > 0).sum()),
            'n_folds': int(len(diff)),
            'paired_t_p_value': p_value,
        }
        logger.info(
            "%s vs %s: %s diff %+.4f ± %.4f, better in %d/%d folds, paired t p = %.3f",
            key, baseline_key, metric, comparison[key]['mean_diff'], comparison[key]['std_diff'],
            comparison[key]['n_folds_improved'], len(diff), p_value,
        )
    return comparison
