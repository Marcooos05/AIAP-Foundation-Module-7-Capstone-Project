"""
Baseline model and the shared training machinery for diabetes risk classification.

The baseline is Logistic Regression (README §5.5): a linear, interpretable
model that sets the benchmark the advanced models in `src.advanced_models`
must beat. A most-frequent dummy gives the floor below it. `ModelTraining`
also holds everything every model shares: pipeline building, stratified CV
tuning, model selection, threshold tuning, test evaluation and explanation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preparation import DataPreparation
from src.evaluation import compute_metrics, metrics_from_scores, threshold_for_target_recall
from src.feature_engineering import FEATURE_SPECS

logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Baseline model training and the shared training and evaluation machinery (README §5).

    Every model is a scikit-learn pipeline `engineer` -> `preprocessor` ->
    `classifier`, so cross-validation recomputes engineered features and refits
    the scaler on each training fold. Class imbalance
    is handled with `class_weight='balanced'` in every model (no resampling,
    README §5.2), and recall is controlled by tuning the decision threshold.

    Each model is trained on two feature variants: with and without fasting
    glucose (README §5.7). Logistic Regression can use its own engineered
    features (`lr_engineered_features`); none currently pass CV validation (EDA §17).

    Subclasses (e.g. `AdvancedModelTraining`) reuse all of this and only
    change `MODEL_NAMES` and `_classifier_factories`.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for model training.
        data_preparation (DataPreparation): Builds a fresh, unfitted feature pipeline per model.
    """

    # Models tuned by this class: the baseline only.
    MODEL_NAMES = ('logistic_regression',)
    GLUCOSE_VARIANTS = ('with_glucose', 'without_glucose')

    # Extra metrics recorded during CV, alongside the selection metric.
    CV_METRICS = ('average_precision', 'roc_auc', 'recall')

    def __init__(self, config: Dict[str, Any], data_preparation: DataPreparation):
        """
        Initializes the ModelTraining class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model training.
            data_preparation (DataPreparation): Provides `create_feature_pipeline` and `feature_variants`.
        """
        self.config = config
        self.data_preparation = data_preparation

    @property
    def cv_splitter(self) -> StratifiedKFold:
        """The stratified k-fold splitter shared by every CV step, so all models see the same folds."""
        return StratifiedKFold(n_splits=self.config['cv'], shuffle=True, random_state=self.config['random_state'])

    def build_pipeline(self, model_name: str, glucose_variant: str) -> Pipeline:
        """
        Wraps an unfitted classifier with a freshly built feature pipeline
        (engineering + preprocessing) for its feature variant.

        Args:
            model_name (str): One of `MODEL_NAMES`.
            glucose_variant (str): One of `GLUCOSE_VARIANTS`.
        Returns:
            Pipeline: An unfitted `engineer` -> `preprocessor` -> `classifier` pipeline.
        Raises:
            ValueError: If the model or variant name is unknown.
        """
        if glucose_variant not in self.GLUCOSE_VARIANTS:
            raise ValueError(f"Unknown glucose variant '{glucose_variant}'; expected one of {self.GLUCOSE_VARIANTS}.")
        model_family = 'logistic_regression' if model_name == 'logistic_regression' else 'trees'
        variant_kwargs = self.data_preparation.feature_variants()[f"{glucose_variant}__{model_family}"]
        feature_pipeline = self.data_preparation.create_feature_pipeline(**variant_kwargs)
        return Pipeline(steps=[*feature_pipeline.steps, ('classifier', self._make_classifier(model_name))])

    def train_baselines(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Cross-validates baseline models that set the floor the tuned models must beat:
        a most-frequent-class dummy (predicts Negative for everyone) and an
        untuned, balanced Logistic Regression for each glucose variant.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Dict[str, Dict[str, Any]]: CV metrics per baseline.
        """
        logger.info("Starting baseline cross-validation.")
        baselines = {
            'dummy_most_frequent': Pipeline(steps=[('classifier', DummyClassifier(strategy='most_frequent'))]),
        }
        for variant in self.GLUCOSE_VARIANTS:
            baselines[f'logistic_regression_default__{variant}'] = self.build_pipeline('logistic_regression', variant)

        results = {}
        for name, pipeline in baselines.items():
            scores = cross_validate(pipeline, X_train, y_train, cv=self.cv_splitter, scoring=list(self.CV_METRICS))
            results[name] = self._summarise_cv(
                {metric: scores[f'test_{metric}'] for metric in self.CV_METRICS}, best_params={},
                fit_time=float(np.mean(scores['fit_time'])),
            )
            logger.info("Baseline %s: CV %s %.4f", name, self.config['scoring'], results[name]['cv_mean'])
        return results

    def tune_models(
        self, X_train: pd.DataFrame, y_train: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Tunes every model on every glucose variant with GridSearchCV
        (stratified k-fold on the training set, selected on `scoring`).
        GridSearchCV refits each best pipeline on the full training set.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Fitted best
                pipelines and their CV results, keyed `"{model}__{variant}"`.
        """
        logger.info("Starting hyperparameter tuning.")
        scoring = self.config['scoring']
        models, results = {}, {}

        for variant in self.GLUCOSE_VARIANTS:
            for model_name in self.MODEL_NAMES:
                key = f"{model_name}__{variant}"
                grid_search = GridSearchCV(
                    estimator=self.build_pipeline(model_name, variant),
                    param_grid=self.config['param_grids'][model_name],
                    cv=self.cv_splitter,
                    scoring=list(self.CV_METRICS),
                    refit=scoring,
                    n_jobs=self.config['n_jobs'],
                )
                grid_search.fit(X_train, y_train)

                best = grid_search.best_index_
                cv = grid_search.cv_results_
                fold_scores = {
                    metric: np.array([cv[f'split{i}_test_{metric}'][best] for i in range(self.config['cv'])])
                    for metric in self.CV_METRICS
                }
                models[key] = grid_search.best_estimator_
                results[key] = self._summarise_cv(
                    fold_scores, best_params=grid_search.best_params_, fit_time=float(cv['mean_fit_time'][best]),
                )
                logger.info(
                    "%s: CV %s %.4f ± %.4f with %s",
                    key, scoring, results[key]['cv_mean'], results[key]['cv_std'], results[key]['best_params'],
                )

        logger.info("Hyperparameter tuning completed.")
        return models, results

    @staticmethod
    def select_best(results: Dict[str, Dict[str, Any]], glucose_variant: str) -> str:
        """
        Picks the pipeline with the highest mean CV score within one glucose variant.

        Args:
            results (Dict[str, Dict[str, Any]]): CV results from `tune_models`.
            glucose_variant (str): Variant to choose within.
        Returns:
            str: The key of the best pipeline.
        Raises:
            ValueError: If no result belongs to the variant.
        """
        candidates = {key: r for key, r in results.items() if key.endswith(f"__{glucose_variant}")}
        if not candidates:
            raise ValueError(f"No tuned models found for variant '{glucose_variant}'.")
        best = max(candidates, key=lambda key: candidates[key]['cv_mean'])
        logger.info("Best %s model: %s (CV %.4f).", glucose_variant, best, candidates[best]['cv_mean'])
        return best

    def tune_threshold(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Chooses the decision threshold from out-of-fold predictions on the
        training set: the highest threshold that still keeps recall at or above
        `target_recall`. Out-of-fold predictions come from models that never
        saw the row, so the threshold isn't tuned on optimistic in-sample
        scores, and the test set stays untouched.

        Args:
            pipeline (Pipeline): The selected pipeline (its hyperparameters are reused; it is cloned).
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
        Returns:
            Dict[str, Any]: The threshold, plus out-of-fold metrics at that threshold and at 0.5.
        """
        y_oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=self.cv_splitter, method='predict_proba')[:, 1]
        chosen = threshold_for_target_recall(y_train, y_oof, self.config['target_recall'])
        result = {
            'threshold': chosen['threshold'],
            'target_recall': self.config['target_recall'],
            'oof_at_threshold': metrics_from_scores(y_train, y_oof, chosen['threshold']),
            'oof_at_0.5': metrics_from_scores(y_train, y_oof, 0.5),
        }
        logger.info(
            "Threshold %.3f: OOF recall %.3f, precision %.3f (at 0.5: recall %.3f, precision %.3f).",
            result['threshold'], result['oof_at_threshold']['recall'], result['oof_at_threshold']['precision'],
            result['oof_at_0.5']['recall'], result['oof_at_0.5']['precision'],
        )
        return result

    def evaluate_on_test(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluates a fitted pipeline on the held-out test set.

        Args:
            pipeline (Pipeline): A pipeline fitted on the full training set.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target variable.
            model_name (str): Name used in the log.
            threshold (float): Decision threshold on the positive-class probability.
        Returns:
            Dict[str, Any]: Test-set metrics.
        """
        metrics = compute_metrics(pipeline, X_test, y_test, threshold)
        logger.info(
            "Test metrics for %s (threshold %.3f): %s",
            model_name, threshold, {k: round(v, 4) for k, v in metrics.items() if k != 'confusion_matrix'},
        )
        return metrics

    def explain_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Estimates feature importance by permutation: each input column the
        model uses is shuffled in turn and the drop in `scoring` is measured.
        This works for every model type, unlike coefficients or impurity
        importances. Raw input columns are permuted, so an engineered feature
        (e.g. family_history_x_bmi) is recomputed from its shuffled source and
        its effect counts toward that source column.

        Args:
            pipeline (Pipeline): A fitted pipeline.
            X (pd.DataFrame): Evaluation features.
            y (pd.Series): Evaluation target variable.
        Returns:
            Dict[str, Dict[str, float]]: Mean and SD importance per input column, largest first.
        """
        model_columns = [
            column
            for name, _, columns in pipeline.named_steps['preprocessor'].transformers_
            if name != 'remainder'
            for column in columns
        ]
        used_columns = list(dict.fromkeys(
            source
            for column in model_columns
            for source in (FEATURE_SPECS[column].sources if column in FEATURE_SPECS else (column,))
        ))
        result = permutation_importance(
            pipeline, X[used_columns], y, scoring=self.config['scoring'], n_repeats=20,
            random_state=self.config['random_state'], n_jobs=self.config['n_jobs'],
        )
        importance = {
            column: {'mean': float(mean), 'std': float(std)}
            for column, mean, std in zip(used_columns, result.importances_mean, result.importances_std)
        }
        return dict(sorted(importance.items(), key=lambda item: item[1]['mean'], reverse=True))

    def _classifier_factories(self) -> Dict[str, Callable[[], Any]]:
        """Model name -> factory for an unfitted classifier with class_weight='balanced'."""
        return {
            'logistic_regression': lambda: LogisticRegression(class_weight='balanced', max_iter=1000),
        }

    def _make_classifier(self, model_name: str) -> Any:
        """Builds an unfitted classifier from `_classifier_factories`."""
        classifiers = self._classifier_factories()
        if model_name not in classifiers:
            raise ValueError(f"Unknown model '{model_name}'; expected one of {list(classifiers)}.")
        return classifiers[model_name]()

    def _summarise_cv(
        self, fold_scores: Dict[str, np.ndarray], best_params: Dict[str, Any], fit_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Turns per-fold CV scores into JSON-safe summary statistics. The per-fold
        scores are kept because every model uses the same folds (`cv_splitter`),
        so models can be compared fold by fold (`compare_to_baseline`).
        """
        selection = fold_scores[self.config['scoring']]
        return {
            'cv_mean': float(np.mean(selection)),
            'cv_std': float(np.std(selection)),
            'cv_metrics': {metric: float(np.mean(scores)) for metric, scores in fold_scores.items()},
            'fold_scores': {metric: [float(v) for v in scores] for metric, scores in fold_scores.items()},
            'mean_fit_time_s': fit_time,
            'best_params': {key.replace('classifier__', ''): value for key, value in best_params.items()},
        }

    @staticmethod
    def used_feature_names(pipeline: Pipeline) -> List[str]:
        """Feature names after preprocessing, as the classifier sees them."""
        return list(pipeline.named_steps['preprocessor'].get_feature_names_out())
