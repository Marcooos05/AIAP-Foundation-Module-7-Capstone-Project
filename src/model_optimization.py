"""
Cross-validation strategies, hyperparameter tuning and final model selection
for diabetes risk classification, across the baseline and advanced models.

The fixed-grid tuning in `src.models` uses one 5-fold split. This module
addresses what that left open (README, Model optimization):

- **Noise:** `RepeatedStratifiedKFold` scores every candidate on several
  shuffles of the folds, and nested CV estimates the optimism of tuning itself.
- **Overfitting:** wider, regularisation-focused random searches, refitted with
  a one-standard-error rule that prefers the smallest train-validation gap.
- **Underfitting:** Logistic Regression can search a degree-2 polynomial step,
  and `diagnose_fit` and learning curves show whether capacity is the limit.

Everything here sees the training set only. The final test evaluation stays in
`main.py`, after selection.
"""

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.advanced_models import AdvancedModelTraining
from src.data_preparation import DataPreparation
from src.models import ModelTraining

logger = logging.getLogger(__name__)


def make_one_se_refit(n_folds: int) -> Callable[[Dict[str, np.ndarray]], int]:
    """
    Builds a `refit` callable for a scikit-learn search: among configurations
    within one standard error of the best mean validation score, it picks the
    one with the smallest absolute train-validation gap. Differences inside
    one SE are noise, so nothing real is given up for a less overfitted model.

    Args:
        n_folds (int): Folds per repeat. SE = SD / sqrt(n_folds): repeats reuse
            the same rows, so they must not count as extra independent evidence.
    Returns:
        Callable[[Dict[str, np.ndarray]], int]: Maps `cv_results_` to the chosen index.
    """
    def refit(cv_results: Dict[str, np.ndarray]) -> int:
        means = np.asarray(cv_results['mean_test_score'])
        gaps = np.abs(np.asarray(cv_results['mean_train_score']) - means)
        best = int(np.argmax(means))
        cutoff = means[best] - cv_results['std_test_score'][best] / np.sqrt(n_folds)
        in_range = np.flatnonzero(means >= cutoff)
        return int(in_range[np.argmin(gaps[in_range])])
    return refit


def diagnose_fit(train_mean: float, val_mean: float, val_se: float, best_val_mean: float, max_gap: float) -> str:
    """
    Labels a model's fit from its train and validation scores.

    Args:
        train_mean (float): Mean training score.
        val_mean (float): Mean validation score.
        val_se (float): Standard error of the validation score.
        best_val_mean (float): Best mean validation score among the models compared.
        max_gap (float): Largest acceptable train-validation gap.
    Returns:
        str: 'overfit' if the gap exceeds `max_gap`; 'underfit' if the gap is
            fine but validation is over one SE below the best model; else 'good_fit'.
    """
    if train_mean - val_mean > max_gap:
        return 'overfit'
    if val_mean < best_val_mean - val_se:
        return 'underfit'
    return 'good_fit'


def select_final(scores: Mapping[str, Mapping[str, float]], simplest_first: Sequence[str]) -> Dict[str, Any]:
    """
    Picks the simplest model whose mean validation score is within one standard
    error of the best one. A simpler model that ties is cheaper, easier to
    explain and less likely to overfit.

    Args:
        scores (Mapping[str, Mapping[str, float]]): Key -> `val_mean` and `val_se`.
        simplest_first (Sequence[str]): All the keys, simplest first.
    Returns:
        Dict[str, Any]: The selected key, the best-scoring key, the cutoff and the reason.
    Raises:
        ValueError: If `simplest_first` doesn't list exactly the keys of `scores`.
    """
    if set(simplest_first) != set(scores):
        raise ValueError(f"simplest_first {list(simplest_first)} must list exactly {list(scores)}.")

    best = max(scores, key=lambda key: scores[key]['val_mean'])
    cutoff = scores[best]['val_mean'] - scores[best]['val_se']
    selected = next(key for key in simplest_first if scores[key]['val_mean'] >= cutoff)
    reason = (
        f"{selected} is the best-scoring model" if selected == best else
        f"{selected} is the simplest model within 1 SE of the best ({best}, {scores[best]['val_mean']:.4f})"
    )
    return {'selected': selected, 'best_cv': best, 'cutoff': float(cutoff), 'reason': reason}


class ModelOptimizer:
    """
    Tunes, diagnoses and selects among the baseline and advanced models on the
    training set. Pipelines come from the `ModelTraining` classes, so the
    feature steps and class weighting match the earlier stages.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        settings (Dict[str, Any]): The config's `optimization` block.
        trainers (Dict[str, ModelTraining]): Model name -> the trainer that builds it.
    """

    # Simplest first: linear, then a small additive ensemble, then many deep trees.
    MODELS_SIMPLEST_FIRST = ('logistic_regression', 'gradient_boosting', 'random_forest')

    def __init__(self, config: Dict[str, Any], data_preparation: DataPreparation):
        """
        Initializes the ModelOptimizer with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            data_preparation (DataPreparation): Passed to the model trainers.
        """
        self.config = config
        self.settings = config['optimization']
        self.trainers = {
            model_name: trainer_class(config, data_preparation)
            for trainer_class in (ModelTraining, AdvancedModelTraining)
            for model_name in trainer_class.MODEL_NAMES
        }

    @property
    def repeated_cv(self) -> RepeatedStratifiedKFold:
        """The repeated stratified splitter shared by tuning and fit diagnostics."""
        return RepeatedStratifiedKFold(
            n_splits=self.config['cv'], n_repeats=self.settings['cv_repeats'], random_state=self.config['random_state'],
        )

    @property
    def single_cv(self) -> StratifiedKFold:
        """One stratified k-fold, for the inner loop of nested CV and for learning curves."""
        return StratifiedKFold(n_splits=self.config['cv'], shuffle=True, random_state=self.config['random_state'])

    def build_pipeline(self, model_name: str, glucose_variant: str) -> Pipeline:
        """
        Builds the search pipeline. Logistic Regression gains a `poly` step
        (degree 1 changes nothing) so the search can test whether it underfits,
        and uses `liblinear`, which supports both the L1 and L2 penalties searched.

        Args:
            model_name (str): A key of `trainers`.
            glucose_variant (str): One of `ModelTraining.GLUCOSE_VARIANTS`.
        Returns:
            Pipeline: An unfitted pipeline.
        """
        pipeline = self.trainers[model_name].build_pipeline(model_name, glucose_variant)
        if model_name != 'logistic_regression':
            return pipeline

        poly = PolynomialFeatures(degree=1, include_bias=False)
        pipeline = Pipeline([*pipeline.steps[:-1], ('poly', poly), pipeline.steps[-1]])
        return pipeline.set_params(classifier__solver='liblinear', classifier__random_state=self.config['random_state'])

    def build_search(self, model_name: str, glucose_variant: str, n_iter: int, cv: Any) -> RandomizedSearchCV:
        """
        Builds a seeded random search over the model's `search_spaces`, refitted
        with the one-standard-error rule.

        Args:
            model_name (str): A key of `trainers`.
            glucose_variant (str): One of `ModelTraining.GLUCOSE_VARIANTS`.
            n_iter (int): Candidates to sample, capped at the size of the space.
            cv (Any): The CV splitter.
        Returns:
            RandomizedSearchCV: An unfitted search.
        """
        space = self.config['search_spaces'][model_name]
        return RandomizedSearchCV(
            estimator=self.build_pipeline(model_name, glucose_variant),
            param_distributions=space,
            n_iter=min(n_iter, len(ParameterGrid(space))),
            scoring=self.config['scoring'],
            cv=cv,
            refit=make_one_se_refit(self.config['cv']),
            return_train_score=True,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs'],
        )

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Runs the random search for every model on every glucose variant.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Refitted
                pipelines and search summaries, keyed `"{model}__{variant}"`.
        """
        models, results = {}, {}
        for variant in ModelTraining.GLUCOSE_VARIANTS:
            for model_name in self.MODELS_SIMPLEST_FIRST:
                key = f"{model_name}__{variant}"
                search = self.build_search(model_name, variant, self.settings['n_iter'], self.repeated_cv)
                search.fit(X_train, y_train)

                params = search.best_params_
                models[key] = search.best_estimator_
                results[key] = {
                    'n_candidates': len(search.cv_results_['params']),
                    'best_params': {name.replace('classifier__', ''): value for name, value in params.items()},
                    'top_candidate_val_mean': float(np.max(search.cv_results_['mean_test_score'])),
                    'params_at_search_edge': self._params_at_edge(params, self.config['search_spaces'][model_name]),
                }
                logger.info("%s: tuned %s", key, results[key]['best_params'])
        return models, results

    def evaluate_fit(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Scores a pipeline's configuration on the repeated folds, on both the
        train and validation sides, so any two models are compared on the same splits.

        Args:
            pipeline (Pipeline): A pipeline (cloned, so a fitted one is left as is).
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Dict[str, Any]: Mean train and validation scores, gap, and validation SD and SE.
        """
        scores = cross_validate(
            clone(pipeline), X_train, y_train, cv=self.repeated_cv, scoring=self.config['scoring'],
            return_train_score=True, n_jobs=self.config['n_jobs'],
        )
        train_mean, val_mean = float(np.mean(scores['train_score'])), float(np.mean(scores['test_score']))
        val_std = float(np.std(scores['test_score']))
        return {
            'train_mean': train_mean,
            'val_mean': val_mean,
            'gap': train_mean - val_mean,
            'val_std': val_std,
            'val_se': val_std / float(np.sqrt(self.config['cv'])),
        }

    def nested_cv(
        self, model_name: str, glucose_variant: str, X_train: pd.DataFrame, y_train: pd.Series,
    ) -> Dict[str, Any]:
        """
        Estimates performance of the whole tuning procedure: each outer fold
        repeats the search on the rest of the data and scores the winner on
        rows the search never saw, so the estimate carries no tuning optimism.

        Args:
            model_name (str): A key of `trainers`.
            glucose_variant (str): One of `ModelTraining.GLUCOSE_VARIANTS`.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Dict[str, Any]: Mean, SD and per-fold outer scores.
        """
        search = self.build_search(model_name, glucose_variant, self.settings['nested_n_iter'], self.single_cv)
        outer = StratifiedKFold(n_splits=self.config['cv'], shuffle=True, random_state=self.config['random_state'] + 1)
        scores = cross_val_score(search, X_train, y_train, cv=outer, scoring=self.config['scoring'])
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'fold_scores': [float(score) for score in scores],
        }

    def learning_curves(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, List[float]]:
        """
        Measures train and validation scores as the training set grows. Curves
        that are still apart at full size suggest more data would help; curves
        that have met at a low score suggest more capacity would.

        Args:
            pipeline (Pipeline): A pipeline (cloned by `learning_curve`).
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Dict[str, List[float]]: Training sizes, and mean and SD of the scores at each.
        """
        sizes, train, val = learning_curve(
            pipeline, X_train, y_train, cv=self.single_cv, scoring=self.config['scoring'],
            train_sizes=np.linspace(0.2, 1.0, self.settings['learning_curve_points']),
            shuffle=True, random_state=self.config['random_state'], n_jobs=self.config['n_jobs'],
        )
        return {
            'train_sizes': [int(size) for size in sizes],
            'train_mean': train.mean(axis=1).tolist(),
            'train_std': train.std(axis=1).tolist(),
            'val_mean': val.mean(axis=1).tolist(),
            'val_std': val.std(axis=1).tolist(),
        }

    def optimize(
        self, reference_models: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Tunes every model, compares it with its fixed-grid reference on the same
        repeated folds, diagnoses its fit and selects one model per glucose variant.

        Args:
            reference_models (Dict[str, Pipeline]): Fixed-grid pipelines from
                `ModelTraining.tune_models`, keyed `"{model}__{variant}"`.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Tuned pipelines,
                and per variant the model reports and the selection.
        """
        tuned_models, tuned_results = self.tune(X_train, y_train)
        max_gap = self.settings['max_fit_gap']

        report = {}
        for variant in ModelTraining.GLUCOSE_VARIANTS:
            keys = [f"{model_name}__{variant}" for model_name in self.MODELS_SIMPLEST_FIRST]
            before = {key: self.evaluate_fit(reference_models[key], X_train, y_train) for key in keys}
            after = {key: self.evaluate_fit(tuned_models[key], X_train, y_train) for key in keys}

            best_before = max(fit['val_mean'] for fit in before.values())
            best_after = max(fit['val_mean'] for fit in after.values())
            models_report = {}
            for key in keys:
                model_name = key.split('__')[0]
                models_report[key] = {
                    **tuned_results[key],
                    'before': {**before[key], 'diagnosis': diagnose_fit(
                        before[key]['train_mean'], before[key]['val_mean'], before[key]['val_se'], best_before, max_gap,
                    )},
                    'after': {**after[key], 'diagnosis': diagnose_fit(
                        after[key]['train_mean'], after[key]['val_mean'], after[key]['val_se'], best_after, max_gap,
                    )},
                    'nested_cv': self.nested_cv(model_name, variant, X_train, y_train),
                    'learning_curve': self.learning_curves(tuned_models[key], X_train, y_train),
                }
                logger.info(
                    "%s: gap %+.3f -> %+.3f (%s -> %s), val %.4f -> %.4f, nested %.4f",
                    key, before[key]['gap'], after[key]['gap'], models_report[key]['before']['diagnosis'],
                    models_report[key]['after']['diagnosis'], before[key]['val_mean'], after[key]['val_mean'],
                    models_report[key]['nested_cv']['mean'],
                )

            selection = select_final(after, keys)
            logger.info("Selected for %s: %s", variant, selection['reason'])
            report[variant] = {'models': models_report, 'selection': selection}
        return tuned_models, report

    @staticmethod
    def _params_at_edge(params: Dict[str, Any], space: Dict[str, List[Any]]) -> List[str]:
        """Numeric parameters the search settled on at an end of their range: the space may be too narrow."""
        return [
            name for name, value in params.items()
            if len(space[name]) > 1 and value in (space[name][0], space[name][-1])
            and all(isinstance(option, (int, float)) for option in space[name])
        ]
