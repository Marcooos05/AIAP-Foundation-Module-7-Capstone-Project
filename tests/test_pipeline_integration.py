import pytest

from src.advanced_models import AdvancedModelTraining, compare_to_baseline
from src.data_preparation import DataPreparation
from src.models import ModelTraining

METRIC_KEYS = (
    "recall", "specificity", "precision", "f1", "f2", "roc_auc", "average_precision", "brier_score", "accuracy",
)


@pytest.fixture
def trained(base_config, learnable_df):
    data_prep = DataPreparation(base_config)
    X_train, X_test, y_train, y_test = data_prep.split_data(data_prep.clean_data(learnable_df))
    model_training = ModelTraining(base_config, data_prep)
    models, results = model_training.tune_models(X_train, y_train)
    advanced_models, advanced_results = AdvancedModelTraining(base_config, data_prep).tune_models(X_train, y_train)
    models.update(advanced_models)
    results.update(advanced_results)
    return model_training, models, results, (X_train, X_test, y_train, y_test)


def test_tunes_baseline_and_advanced_models_on_both_variants(trained):
    _, models, results, _ = trained
    names = ModelTraining.MODEL_NAMES + AdvancedModelTraining.MODEL_NAMES
    expected = {f"{m}__{v}" for m in names for v in ModelTraining.GLUCOSE_VARIANTS}
    assert set(models) == set(results) == expected
    for result in results.values():
        assert 0.0 <= result["cv_mean"] <= 1.0
        assert set(result["cv_metrics"]) == set(ModelTraining.CV_METRICS)


def test_baselines_include_dummy_at_prevalence(base_config, trained):
    model_training, _, _, (X_train, _, y_train, _) = trained
    baselines = model_training.train_baselines(X_train, y_train)
    # A constant prediction has average precision equal to the positive rate.
    assert baselines["dummy_most_frequent"]["cv_mean"] == pytest.approx(y_train.mean(), abs=0.02)
    assert baselines["logistic_regression_default__with_glucose"]["cv_mean"] > baselines["dummy_most_frequent"]["cv_mean"]


@pytest.mark.parametrize("variant", ModelTraining.GLUCOSE_VARIANTS)
def test_advanced_models_compare_against_baseline_on_same_folds(base_config, trained, variant):
    _, _, results, _ = trained
    variant_results = {key: r for key, r in results.items() if key.endswith(variant)}
    comparison = compare_to_baseline(variant_results, f"logistic_regression__{variant}", base_config["scoring"])
    assert set(comparison) == {f"{m}__{variant}" for m in AdvancedModelTraining.MODEL_NAMES}
    for result in comparison.values():
        assert result["n_folds"] == base_config["cv"]
        assert 0.0 <= result["paired_t_p_value"] <= 1.0


def test_end_to_end_selection_threshold_and_test_evaluation(trained):
    model_training, models, results, (X_train, X_test, y_train, y_test) = trained

    for variant in ModelTraining.GLUCOSE_VARIANTS:
        best_name = model_training.select_best(results, variant)
        assert best_name.endswith(variant)

        threshold = model_training.tune_threshold(models[best_name], X_train, y_train)
        metrics = model_training.evaluate_on_test(models[best_name], X_test, y_test, best_name, threshold["threshold"])
        assert set(METRIC_KEYS) <= set(metrics)
        assert sum(metrics["confusion_matrix"].values()) == len(y_test)


def test_learns_signal_with_glucose(trained):
    model_training, models, results, (_, X_test, _, y_test) = trained
    best_name = model_training.select_best(results, "with_glucose")
    metrics = model_training.evaluate_on_test(models[best_name], X_test, y_test, best_name)
    assert metrics["roc_auc"] > 0.75
