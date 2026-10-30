"""Reusable feature engineering, feature selection and feature validation for diabetes risk detection."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    """
    One engineered feature.

    Attributes:
        sources (Tuple[str, ...]): Raw columns the feature is computed from.
        build (Callable[[pd.DataFrame], pd.Series]): Row-wise builder; uses fixed
            cut-offs only, so nothing is learned from the data.
        rationale (str): Why the feature is a candidate (see eda.ipynb).
        replaces (Optional[str]): Raw feature it stands in for when used (e.g. a
            monotonic transform); None if it is added alongside the raw features.
    """
    sources: Tuple[str, ...]
    build: Callable[[pd.DataFrame], pd.Series]
    rationale: str
    replaces: Optional[str] = None


def _sqrt_physical_activity(df: pd.DataFrame) -> pd.Series:
    n_negative = int((df['physical_activity_frequency'] < 0).sum())
    if n_negative:
        raise ValueError(f"{n_negative} row(s) have a negative physical_activity_frequency.")
    return np.sqrt(df['physical_activity_frequency'])


def _family_history_x_bmi(df: pd.DataFrame) -> pd.Series:
    return (df['family_history_of_diabetes'] == 'Present').astype(int) * df['bmi']


def _band(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    """Ordinal band 0..len(edges): band i covers [edges[i-1], edges[i])."""
    return pd.Series(np.digitize(values, edges), index=values.index)


# Every engineered feature, in the order it is added. Clinical cut-offs:
# WHO BMI classes, ADA screening ages and prediabetes glucose, ACC/AHA stage 1 hypertension.
FEATURE_SPECS: Dict[str, FeatureSpec] = {
    'sqrt_physical_activity': FeatureSpec(
        ('physical_activity_frequency',), _sqrt_physical_activity,
        "Removes the right skew of activity (0.57 -> -0.06, EDA §4.3, §17.2).",
        replaces='physical_activity_frequency',
    ),
    'family_history_x_bmi': FeatureSpec(
        ('family_history_of_diabetes', 'bmi'), _family_history_x_bmi,
        "The family-history effect varies across BMI bands (EDA §12.1).",
    ),
    'bmi_squared': FeatureSpec(
        ('bmi',), lambda df: df['bmi'] ** 2,
        "Non-linear BMI term: the positive rate drops steeply from underweight to normal BMI (EDA §12).",
    ),
    'bmi_band': FeatureSpec(
        ('bmi',), lambda df: _band(df['bmi'], [18.5, 25, 30]),
        "WHO BMI class (underweight / normal / overweight / obese) (EDA §12).",
    ),
    'age_band': FeatureSpec(
        ('age',), lambda df: _band(df['age'], [35, 45, 65]),
        "ADA screening age bands (<35 / 35-44 / 45-64 / 65+) (EDA §12).",
    ),
    'elevated_bp': FeatureSpec(
        ('blood_pressure',), lambda df: (df['blood_pressure'] >= 130).astype(int),
        "Stage 1 hypertension flag, BP >= 130 (EDA §12).",
    ),
    'prediabetic_glucose': FeatureSpec(
        ('fasting_glucose_level',), lambda df: (df['fasting_glucose_level'] >= 100).astype(int),
        "ADA prediabetes flag, fasting glucose >= 100 mg/dL (EDA §12-13).",
    ),
}


def features_derived_from(columns: Iterable[str], features: Optional[Iterable[str]] = None) -> List[str]:
    """
    Lists the engineered features computed from any of the given raw columns,
    e.g. to drop glucose-derived features from the "without glucose" variant.

    Args:
        columns (Iterable[str]): Raw column names.
        features (Optional[Iterable[str]]): Engineered features to search; all if None.
    Returns:
        List[str]: Matching engineered feature names.
    """
    columns = set(columns)
    names = FEATURE_SPECS if features is None else features
    return [name for name in names if columns & set(FEATURE_SPECS[name].sources)]


class ClinicalFeatureEngineer(TransformerMixin, BaseEstimator):
    """
    scikit-learn transformer that adds engineered features (`FEATURE_SPECS`) to
    a raw DataFrame. Every feature is row-wise with fixed cut-offs, so `fit`
    learns nothing beyond the input columns; it still follows fit/transform so
    it can sit at the start of a `Pipeline` and run inside each CV fold.

    Attributes:
        features (Optional[Sequence[str]]): Engineered features to add; all if None.
    """

    def __init__(self, features: Optional[Sequence[str]] = None):
        self.features = features

    def _selected_features(self) -> List[str]:
        return list(FEATURE_SPECS) if self.features is None else list(self.features)

    def _check_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"ClinicalFeatureEngineer expects a pandas DataFrame, got {type(X).__name__}.")

        unknown = sorted(set(self._selected_features()) - set(FEATURE_SPECS))
        if unknown:
            raise ValueError(f"Unknown engineered feature(s) {unknown}. Expected some of {list(FEATURE_SPECS)}.")

        sources = sorted({col for name in self._selected_features() for col in FEATURE_SPECS[name].sources})
        missing = [col for col in sources if col not in X.columns]
        if missing:
            raise ValueError(f"Input is missing column(s) needed for feature engineering: {missing}.")

        n_missing = {col: int(count) for col, count in X[sources].isna().sum().items() if count}
        if n_missing:
            raise ValueError(f"Missing value(s) in feature engineering input: {n_missing}.")

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'ClinicalFeatureEngineer':
        """
        Checks the input and records its columns.

        Args:
            X (pd.DataFrame): Raw features.
            y (Any): Ignored.
        Returns:
            ClinicalFeatureEngineer: The fitted transformer.
        """
        self._check_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the selected engineered features.

        Args:
            X (pd.DataFrame): Raw features with the same columns as in `fit`.
        Returns:
            pd.DataFrame: A copy of X with the engineered features appended.
        Raises:
            ValueError: If a source column is missing or has a missing value, or
                physical activity is negative.
        """
        check_is_fitted(self, 'feature_names_in_')
        self._check_input(X)
        out = X.copy()
        for name in self._selected_features():
            out[name] = FEATURE_SPECS[name].build(X)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, 'feature_names_in_')
        return np.asarray([*self.feature_names_in_, *self._selected_features()], dtype=object)


def l1_selection(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """
    Regularisation-based selection: fits a pipeline that ends in an L1
    `LogisticRegressionCV` on the training data and reads which (scaled)
    features keep a non-zero coefficient at the CV-chosen strength.

    Args:
        pipeline (Pipeline): Unfitted feature steps followed by an L1 `LogisticRegressionCV`.
        X (pd.DataFrame): Raw training features.
        y (pd.Series): Training target.
    Returns:
        Dict[str, Any]: The chosen C, coefficients by |size|, and the kept / dropped features.
    """
    fitted = clone(pipeline).fit(X, y)
    names = list(fitted[:-1].get_feature_names_out())
    coefs = pd.Series(fitted[-1].coef_.ravel(), index=names)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    return {
        'C': float(np.ravel(fitted[-1].C_)[0]),
        'coefficients': {name: round(float(value), 4) for name, value in coefs.items()},
        'kept': [name for name, value in coefs.items() if value != 0],
        'dropped': [name for name, value in coefs.items() if value == 0],
    }


def permutation_ranking(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    random_state: int,
    scoring: str = 'average_precision',
    n_repeats: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Model-agnostic ranking: for each stratified CV fold of the training data,
    fits the pipeline on the training fold and measures how much the score on
    the held-out fold drops when each model input column is shuffled.

    Args:
        pipeline (Pipeline): Unfitted feature steps followed by a classifier.
        X (pd.DataFrame): Raw training features.
        y (pd.Series): Training target.
        cv (int): Number of folds.
        random_state (int): Seed for the folds and the shuffles.
        scoring (str): scikit-learn scorer name.
        n_repeats (int): Shuffles per feature per fold.
    Returns:
        Dict[str, Dict[str, float]]: Feature -> mean / std score drop across folds, best first.
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    drops = []
    for train_idx, val_idx in folds.split(X, y):
        fitted = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        features, model = fitted[:-1], fitted[-1]
        result = permutation_importance(
            model, features.transform(X.iloc[val_idx]), y.iloc[val_idx],
            scoring=scoring, n_repeats=n_repeats, random_state=random_state,
        )
        drops.append(pd.Series(result.importances_mean, index=features.get_feature_names_out()))

    drops = pd.concat(drops, axis=1)
    summary = pd.DataFrame({'mean': drops.mean(axis=1), 'std': drops.std(axis=1)}).sort_values('mean', ascending=False)
    return {name: {'mean': round(float(row['mean']), 4), 'std': round(float(row['std']), 4)}
            for name, row in summary.iterrows()}


def compare_feature_sets(
    pipelines: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    random_state: int,
    baseline: str = 'baseline',
    primary_metric: str = 'average_precision',
) -> Dict[str, Dict[str, Any]]:
    """
    Cross-validates each feature set on the training data with the same
    stratified folds, so the per-fold gains over the baseline are paired.

    Args:
        pipelines (Dict[str, Pipeline]): Feature set name -> unfitted model pipeline;
            must include `baseline`.
        X (pd.DataFrame): Raw training features.
        y (pd.Series): Training target.
        cv (int): Number of folds.
        random_state (int): Seed for the folds.
        baseline (str): Name of the reference set.
        primary_metric (str): Metric used for the gain over the baseline.
    Returns:
        Dict[str, Dict[str, Any]]: Set name -> mean / std per metric, plus the
            mean gain over the baseline and the number of folds it improved.
    """
    if baseline not in pipelines:
        raise ValueError(f"Baseline feature set '{baseline}' is not among {list(pipelines)}.")

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    metrics = list(dict.fromkeys([primary_metric, 'average_precision', 'roc_auc', 'recall']))
    fold_scores = {
        name: cross_validate(pipeline, X, y, cv=folds, scoring=metrics)
        for name, pipeline in pipelines.items()
    }

    baseline_folds = fold_scores[baseline][f'test_{primary_metric}']
    results = {}
    for name, scores in fold_scores.items():
        gains = scores[f'test_{primary_metric}'] - baseline_folds
        results[name] = {
            **{f'{metric}_mean': round(float(scores[f'test_{metric}'].mean()), 4) for metric in metrics},
            **{f'{metric}_std': round(float(scores[f'test_{metric}'].std()), 4) for metric in metrics},
            'gain_vs_baseline': round(float(gains.mean()), 4),
            'n_folds_improved': int((gains > 0).sum()),
        }
    return results


def passes_retention_rule(result: Dict[str, Any], n_folds: int, min_gain: float) -> bool:
    """
    A feature set beats the baseline if its mean gain is at least `min_gain`
    and it improves the score in most folds.

    Args:
        result (Dict[str, Any]): One entry from `compare_feature_sets`.
        n_folds (int): Number of CV folds.
        min_gain (float): Minimum mean gain in the primary metric.
    Returns:
        bool: Whether the gain is large and consistent enough.
    """
    return result['gain_vs_baseline'] >= min_gain and result['n_folds_improved'] > n_folds / 2
