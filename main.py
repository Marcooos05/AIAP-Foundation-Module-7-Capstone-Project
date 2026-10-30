"""Entry point: run the end-to-end diabetes risk detection pipeline."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.advanced_models import AdvancedModelTraining, compare_to_baseline
from src.config_validation import validate_config
from src.data_preparation import DataPreparation
from src.evaluation import (
    bootstrap_ci,
    compute_metrics,
    metrics_from_scores,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.feature_engineering import (
    compare_feature_sets,
    l1_selection,
    passes_retention_rule,
    permutation_ranking,
)
from src.model_optimization import ModelOptimizer
from src.models import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path('./src/config.yaml')
CLASS_LABELS = ['Negative', 'Positive']
MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
FIGURES_DIR = REPORTS_DIR / 'figures'

# Feature selection and comparison run once with and once without the leakage candidates (glucose).
GLUCOSE_SETTINGS = {'with_glucose': True, 'without_glucose': False}

# Architecture of each tuned model, for the comparison report.
ARCHITECTURES = {
    'logistic_regression': 'linear (baseline, src/models.py)',
    'random_forest': 'bagged decision trees (src/advanced_models.py)',
    'gradient_boosting': 'boosted decision trees (src/advanced_models.py)',
}


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the pipeline YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def _load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads the raw dataset referenced by the config and checks required columns exist."""
    file_path = Path(config['file_path'])
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at '{file_path}'.")

    logger.info("Loading data from CSV file.")
    df = pd.read_csv(file_path)
    logger.info("Data loading completed: %d rows, %d columns.", *df.shape)

    required_columns = {config['target_column'], *DataPreparation.REQUIRED_RAW_COLUMNS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Data file is missing required column(s): {sorted(missing_columns)}.")

    return df


def _check_feature_variants(
    data_preparation: DataPreparation,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Fits each feature variant's feature pipeline (engineering + preprocessing)
    on the training set and transforms both sets, as a sanity check before
    modelling: no NaNs, expected columns.
    """
    variants = {}
    for name, kwargs in data_preparation.feature_variants().items():
        feature_pipeline = data_preparation.create_feature_pipeline(**kwargs).fit(X_train)
        Xt_train, Xt_test = feature_pipeline.transform(X_train), feature_pipeline.transform(X_test)
        if np.isnan(Xt_train).any() or np.isnan(Xt_test).any():
            raise ValueError(f"Feature variant '{name}' produced NaN values.")

        feature_names = list(feature_pipeline.get_feature_names_out())
        logger.info("Feature variant %s: %d features %s", name, len(feature_names), feature_names)
        variants[name] = {'n_features': len(feature_names), 'features': feature_names}
    return variants


def _with_model(feature_pipeline: Pipeline, model: ClassifierMixin) -> Pipeline:
    """Appends a model to a feature pipeline."""
    return Pipeline([*feature_pipeline.steps, ('model', model)])


def _logistic_regression(config: Dict[str, Any]) -> LogisticRegression:
    """The baseline model used to judge engineered features (README §5.5)."""
    return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config['random_state'])


def _random_forest(config: Dict[str, Any]) -> RandomForestClassifier:
    """A tree model, to check whether it gains anything from the engineered features."""
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=config['random_state'],
    )


def _applicable_candidates(data_preparation: DataPreparation, include_leakage_candidates: bool) -> List[str]:
    """Candidate features available in a glucose setting (glucose-derived ones are dropped without glucose)."""
    candidates = data_preparation.config['candidate_features']
    available = data_preparation.numeric_features_for(include_leakage_candidates, engineered=candidates)
    return [c for c in candidates if c in available]


def _select_features(
    data_preparation: DataPreparation,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Ranks the raw and candidate features on the training set only, with and
    without glucose: L1-regularised Logistic Regression (regularisation-based
    selection) and permutation importance on held-out CV folds (model-based,
    for Logistic Regression and Random Forest).
    """
    config = data_preparation.config
    cv, seed, scoring = config['cv'], config['random_state'], config['selection_scoring']
    selection = {}
    for setting, include in GLUCOSE_SETTINGS.items():
        candidates = _applicable_candidates(data_preparation, include)
        l1_model = LogisticRegressionCV(
            l1_ratios=(1.0,), solver='liblinear', Cs=20, scoring=scoring, class_weight='balanced',
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed), random_state=seed,
            use_legacy_attributes=False,
        )
        l1 = l1_selection(
            _with_model(data_preparation.create_feature_pipeline(include, engineered=candidates), l1_model),
            X_train, y_train,
        )
        selection[setting] = {
            'l1_logistic_regression': l1,
            'permutation_logistic_regression': permutation_ranking(
                _with_model(data_preparation.create_feature_pipeline(include, engineered=candidates),
                            _logistic_regression(config)),
                X_train, y_train, cv, seed, scoring,
            ),
            'permutation_random_forest': permutation_ranking(
                _with_model(data_preparation.create_feature_pipeline(include), _random_forest(config)),
                X_train, y_train, cv, seed, scoring,
            ),
        }
        logger.info("L1 selection (%s, C=%.4f): kept %s, dropped %s", setting, l1['C'], l1['kept'], l1['dropped'])
    return selection


def _compare_feature_sets(
    data_preparation: DataPreparation,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selection: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cross-validates the raw-feature baseline against engineered feature sets
    on the training set only: each candidate on its own, all candidates, the
    L1-selected ones and the current config. A candidate is recommended if it
    meets the retention rule for Logistic Regression and L1 keeps it, in every
    glucose setting where it applies.
    """
    config = data_preparation.config
    cv, seed, min_gain = config['cv'], config['random_state'], config['min_cv_gain']
    comparison = {}
    for setting, include in GLUCOSE_SETTINGS.items():
        candidates = _applicable_candidates(data_preparation, include)
        l1_kept = selection[setting]['l1_logistic_regression']['kept']
        feature_sets = {
            'baseline': [],
            **{f'+{c}': [c] for c in candidates},
            'all_candidates': candidates,
            'l1_selected': [c for c in candidates if c in l1_kept],
            'config_lr': data_preparation.lr_engineered_features(),
        }
        lr_results = compare_feature_sets(
            {name: _with_model(data_preparation.create_feature_pipeline(include, engineered=engineered),
                               _logistic_regression(config))
             for name, engineered in feature_sets.items()},
            X_train, y_train, cv, seed, primary_metric=config['selection_scoring'],
        )
        rf_results = compare_feature_sets(
            {name: _with_model(data_preparation.create_feature_pipeline(include, engineered=feature_sets[name]),
                               _random_forest(config))
             for name in ('baseline', 'all_candidates')},
            X_train, y_train, cv, seed, primary_metric=config['selection_scoring'],
        )
        retained = [
            c for c in candidates if c in l1_kept and passes_retention_rule(lr_results[f'+{c}'], cv, min_gain)
        ]
        comparison[setting] = {
            'feature_sets': feature_sets,
            'logistic_regression': lr_results,
            'random_forest': rf_results,
            'retained': retained,
        }

        logger.info("Feature set comparison (%s, Logistic Regression, %d-fold CV on train):", setting, cv)
        for name, result in lr_results.items():
            logger.info(
                "  %-28s PR-AUC %.4f ± %.4f  ROC-AUC %.4f  gain %+.4f  folds improved %d/%d",
                name, result['average_precision_mean'], result['average_precision_std'],
                result['roc_auc_mean'], result['gain_vs_baseline'], result['n_folds_improved'], cv,
            )

    recommended = [
        c for c in config['candidate_features']
        if all(c in result['retained'] for setting, result in comparison.items()
               if c in _applicable_candidates(data_preparation, GLUCOSE_SETTINGS[setting]))
    ]
    configured = data_preparation.lr_engineered_features()
    logger.info("Recommended engineered features for Logistic Regression: %s", recommended)
    if set(configured) != set(recommended):
        logger.warning(
            "Configured Logistic Regression features %s differ from the CV recommendation %s; "
            "update lr_engineered_features in config.yaml.", configured, recommended,
        )

    return {
        'retention_rule': (
            f"mean {config['selection_scoring']} gain over baseline >= {min_gain} in Logistic Regression, "
            f"improved in more than half of {cv} folds, and kept by L1, in every glucose setting"
        ),
        **comparison,
        'recommended_features': recommended,
    }


def _train_and_evaluate(
    config: Dict[str, Any],
    data_preparation: DataPreparation,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Pipeline]:
    """
    Stage 2: baseline vs advanced models on the fixed grids (README §5).

    1. Cross-validate the reference floor: a most-frequent dummy and an untuned
       Logistic Regression.
    2. Tune the baseline (Logistic Regression, src/models.py) and the advanced
       models (Random Forest, Gradient Boosting, src/advanced_models.py) on both
       glucose variants with GridSearchCV over the same stratified folds.
    3. Compare each advanced model with the baseline fold by fold.
    4. Per glucose variant, for every model: tune the decision threshold on
       out-of-fold predictions for the target recall, then evaluate once on the
       test set with bootstrap confidence intervals.
    5. Plot and save the comparisons. The final choice is made in Stage 3.

    Returns:
        Dict[str, Pipeline]: The tuned pipelines, the reference for Stage 3.
    """
    baseline_training = ModelTraining(config, data_preparation)
    advanced_training = AdvancedModelTraining(config, data_preparation)

    start = time.perf_counter()
    baseline_results = baseline_training.train_baselines(X_train, y_train)
    logger.info("Baseline cross-validation completed in %.2fs.", time.perf_counter() - start)

    tuned_models, tuned_results = {}, {}
    for trainer in (baseline_training, advanced_training):
        start = time.perf_counter()
        models, results = trainer.tune_models(X_train, y_train)
        tuned_models.update(models)
        tuned_results.update(results)
        logger.info("Tuning %s completed in %.2fs.", list(trainer.MODEL_NAMES), time.perf_counter() - start)

    _save_json({'baselines': baseline_results, 'tuned': tuned_results}, REPORTS_DIR / 'cv_results.json')
    plot_model_comparison(
        {**baseline_results, **tuned_results}, config['scoring'], FIGURES_DIR / 'model_comparison_cv.png',
    )

    comparison = {}
    for variant in ModelTraining.GLUCOSE_VARIANTS:
        baseline_key = f'logistic_regression__{variant}'
        variant_keys = [key for key in tuned_results if key.endswith(f'__{variant}')]
        paired = compare_to_baseline(
            {key: tuned_results[key] for key in variant_keys}, baseline_key, config['scoring'],
        )
        best_name = baseline_training.select_best(tuned_results, variant)

        models_report = {}
        for key in variant_keys:
            model = tuned_models[key]
            threshold = baseline_training.tune_threshold(model, X_train, y_train)

            start = time.perf_counter()
            y_score = model.predict_proba(X_test)[:, 1]
            predict_ms = (time.perf_counter() - start) * 1000

            models_report[key] = {
                'architecture': ARCHITECTURES[key.split('__')[0]],
                'cv_mean': tuned_results[key]['cv_mean'],
                'cv_std': tuned_results[key]['cv_std'],
                'cv_metrics': tuned_results[key]['cv_metrics'],
                'vs_baseline': paired.get(key),
                'threshold': threshold['threshold'],
                'test_at_tuned_threshold': metrics_from_scores(y_test, y_score, threshold['threshold']),
                'test_95ci_at_tuned_threshold': bootstrap_ci(
                    y_test, y_score, threshold['threshold'], random_state=config['random_state'],
                ),
                'mean_fit_time_s': tuned_results[key]['mean_fit_time_s'],
                'predict_time_ms': predict_ms,
            }

        comparison[variant] = {'baseline': baseline_key, 'best_cv': best_name, 'models': models_report}
        _log_comparison(variant, comparison[variant], config['scoring'])

        variant_models = {key.split('__')[0]: tuned_models[key] for key in variant_keys}
        plot_roc_curves(
            variant_models, X_test, y_test, FIGURES_DIR / f'roc_curve_test__{variant}.png', title=f"ROC curve ({variant})",
        )
        plot_precision_recall_curves(
            variant_models, X_test, y_test, FIGURES_DIR / f'pr_curve_test__{variant}.png',
            title=f"Precision-recall curve ({variant})",
        )
        plot_calibration_curves(
            variant_models, X_test, y_test, FIGURES_DIR / f'calibration_curve_test__{variant}.png',
            title=f"Calibration curve ({variant})",
        )

    _save_json(comparison, REPORTS_DIR / 'model_comparison.json')
    return tuned_models


def _optimize_and_select(
    config: Dict[str, Any],
    data_preparation: DataPreparation,
    reference_models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Stage 3: model optimization and final selection (README, Model optimization).

    1. Tune every model with a repeated-CV random search, diagnose over- and
       underfitting before and after, and select one model per glucose variant.
       All of this sees the training set only.
    2. Tune each selected model's threshold on out-of-fold predictions.
    3. Evaluate only the selected models on the test set, with confidence intervals.
    4. Save the final decision: reports, importances, figures and pipelines.
    """
    optimizer = ModelOptimizer(config, data_preparation)
    start = time.perf_counter()
    tuned_models, report = optimizer.optimize(reference_models, X_train, y_train)
    logger.info("Model optimization completed in %.2fs.", time.perf_counter() - start)

    final_metrics, selected_models, feature_importance = {}, {}, {}
    for variant, variant_report in report.items():
        key = variant_report['selection']['selected']
        model = tuned_models[key]
        threshold = optimizer.trainers[key.split('__')[0]].tune_threshold(model, X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        final_metrics[variant] = {
            'model_name': key,
            'test_at_tuned_threshold': metrics_from_scores(y_test, y_score, threshold['threshold']),
            'test_95ci_at_tuned_threshold': bootstrap_ci(
                y_test, y_score, threshold['threshold'], random_state=config['random_state'],
            ),
            'test_at_default_threshold': metrics_from_scores(y_test, y_score),
        }
        selected_models[variant] = {
            'model_name': key,
            'selection': variant_report['selection'],
            'hyperparameters': variant_report['models'][key]['best_params'],
            'features': ModelTraining.used_feature_names(model),
            'cv_score': variant_report['models'][key]['after']['val_mean'],
            'nested_cv_score': variant_report['models'][key]['nested_cv']['mean'],
            'decision_threshold': threshold,
        }
        feature_importance[variant] = optimizer.trainers[key.split('__')[0]].explain_model(model, X_test, y_test)

        plot_confusion_matrix(
            model, X_test, y_test, FIGURES_DIR / f'confusion_matrix_test__{variant}.png',
            labels=CLASS_LABELS, threshold=threshold['threshold'],
            title=f"Test set: {key}\nthreshold {threshold['threshold']:.3f} (target recall "
                  f"{config['target_recall']:.2f})",
        )
        plot_learning_curves(
            {name.split('__')[0]: entry['learning_curve'] for name, entry in variant_report['models'].items()},
            config['scoring'], FIGURES_DIR / f'learning_curves__{variant}.png', title=f"Learning curves ({variant})",
        )
        joblib.dump(model, MODELS_DIR / f'best_pipeline__{variant}.joblib')

    # The dummy predicts Negative for everyone: the floor every model must beat on test.
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    final_metrics['dummy_most_frequent'] = compute_metrics(dummy, X_test, y_test)

    _save_json(report, REPORTS_DIR / 'optimization_results.json')
    _save_json(final_metrics, REPORTS_DIR / 'final_metrics.json')
    _save_json(selected_models, REPORTS_DIR / 'selected_models.json')
    _save_json(feature_importance, REPORTS_DIR / 'feature_importance.json')
    logger.info("Models saved to %s, reports saved to %s", MODELS_DIR.resolve(), REPORTS_DIR.resolve())


def _log_comparison(variant: str, comparison: Dict[str, Any], metric: str) -> None:
    """Logs one row per model: CV score, paired gain over the baseline, and test metrics with 95% CIs."""
    logger.info("Baseline vs advanced models (%s); best CV score: %s", variant, comparison['best_cv'])
    for key, report in comparison['models'].items():
        vs = report['vs_baseline']
        gain = (f"{vs['mean_diff']:+.4f} ({vs['n_folds_improved']}/{vs['n_folds']} folds, p={vs['paired_t_p_value']:.2f})"
                if vs else "baseline")
        ci = report['test_95ci_at_tuned_threshold']
        logger.info(
            "  %-38s CV %s %.4f ± %.4f  vs LR %-30s test recall %.2f [%.2f-%.2f]  PR-AUC %.3f [%.3f-%.3f]",
            key, metric, report['cv_mean'], report['cv_std'], gain,
            ci['recall']['estimate'], ci['recall']['lower'], ci['recall']['upper'],
            ci['average_precision']['estimate'], ci['average_precision']['lower'], ci['average_precision']['upper'],
        )


def main():
    """
    Runs the end-to-end pipeline.

    Stage 1: data preparation and feature engineering
    1. Load and validate the YAML config.
    2. Load the raw dataset and verify required columns are present.
    3. Generate a data validation report on the raw data.
    4. Clean the data (duplicates, missing targets, category levels, rows
       below the minimum age).
    5. Split into stratified train/test sets.
    6. Fit and check the feature pipeline (engineering + preprocessing) for
       every feature variant (with / without glucose, trees / Logistic Regression).
    7. Rank features on the training set (L1 and permutation importance).
    8. Cross-validate the baseline against engineered feature sets on the
       training set and recommend which engineered features to keep.
    9. Save the reports to reports/. The test set is not used by steps 7-8.

    Stage 2: baseline vs advanced models (see `_train_and_evaluate`).

    Stage 3: model optimization and final selection (see `_optimize_and_select`).
    """
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    config = _load_config(CONFIG_PATH)
    df = _load_data(config)

    data_preparation = DataPreparation(config)

    start = time.perf_counter()
    validation_report = data_preparation.generate_validation_report(df)
    _save_json(validation_report, REPORTS_DIR / 'data_validation_report.json')
    logger.info("Data validation completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    df_cleaned = data_preparation.clean_data(df)
    logger.info("Data cleaning completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    X_train, X_test, y_train, y_test = data_preparation.split_data(df_cleaned)
    logger.info("Data splitting completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    variants = _check_feature_variants(data_preparation, X_train, X_test)
    logger.info("Preprocessor checks completed in %.2fs.", time.perf_counter() - start)

    summary = {
        'n_rows_raw': int(len(df)),
        'n_rows_cleaned': int(len(df_cleaned)),
        'splits': {
            name: {'n_rows': int(len(y_split)), 'n_positive': int(y_split.sum()),
                   'positive_share': round(float(y_split.mean()), 4)}
            for name, y_split in (('train', y_train), ('test', y_test))
        },
        'feature_variants': variants,
    }
    _save_json(summary, REPORTS_DIR / 'data_preparation_summary.json')
    logger.info("Data preparation summary saved to %s", (REPORTS_DIR / 'data_preparation_summary.json').resolve())

    start = time.perf_counter()
    selection = _select_features(data_preparation, X_train, y_train)
    _save_json(selection, REPORTS_DIR / 'feature_selection.json')
    logger.info("Feature selection completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    comparison = _compare_feature_sets(data_preparation, X_train, y_train, selection)
    _save_json(comparison, REPORTS_DIR / 'feature_comparison.json')
    logger.info("Feature set comparison completed in %.2fs.", time.perf_counter() - start)

    reference_models = _train_and_evaluate(config, data_preparation, X_train, X_test, y_train, y_test)
    _optimize_and_select(config, data_preparation, reference_models, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
