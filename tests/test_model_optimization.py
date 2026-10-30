import numpy as np
import pytest

from src.advanced_models import AdvancedModelTraining
from src.data_preparation import DataPreparation
from src.model_optimization import ModelOptimizer, diagnose_fit, make_one_se_refit, select_final
from src.models import ModelTraining


@pytest.fixture
def data_prep(base_config):
    return DataPreparation(base_config)


@pytest.fixture
def optimizer(base_config, data_prep):
    return ModelOptimizer(base_config, data_prep)


@pytest.fixture
def train_test(data_prep, learnable_df):
    return data_prep.split_data(data_prep.clean_data(learnable_df))


class TestOneSeRefit:
    def test_picks_smallest_gap_within_one_se(self):
        cv_results = {
            "mean_test_score": np.array([0.70, 0.69, 0.60]),
            "std_test_score": np.array([0.05, 0.05, 0.05]),
            "mean_train_score": np.array([0.95, 0.72, 0.60]),
        }
        # SE = 0.05 / 2 with 4 folds: index 2 is out of range despite having no gap.
        assert make_one_se_refit(4)(cv_results) == 1

    def test_keeps_best_when_nothing_else_is_in_range(self):
        cv_results = {
            "mean_test_score": np.array([0.70, 0.50]),
            "std_test_score": np.array([0.01, 0.01]),
            "mean_train_score": np.array([0.99, 0.50]),
        }
        assert make_one_se_refit(5)(cv_results) == 0


class TestDiagnoseFit:
    @pytest.mark.parametrize("train, val, best, expected", [
        (0.90, 0.60, 0.60, "overfit"),
        (0.52, 0.50, 0.70, "underfit"),
        (0.71, 0.69, 0.70, "good_fit"),
        (0.60, 0.70, 0.70, "good_fit"),  # validation above train is not overfitting
    ])
    def test_labels(self, train, val, best, expected):
        assert diagnose_fit(train, val, val_se=0.02, best_val_mean=best, max_gap=0.1) == expected


class TestSelectFinal:
    ORDER = ("logistic_regression", "gradient_boosting", "random_forest")

    def test_picks_simplest_within_one_se(self):
        scores = {
            "logistic_regression": {"val_mean": 0.69, "val_se": 0.03},
            "gradient_boosting": {"val_mean": 0.70, "val_se": 0.03},
            "random_forest": {"val_mean": 0.65, "val_se": 0.03},
        }
        result = select_final(scores, self.ORDER)
        assert result["selected"] == "logistic_regression"
        assert result["best_cv"] == "gradient_boosting"

    def test_picks_best_when_simpler_models_are_out_of_range(self):
        scores = {
            "logistic_regression": {"val_mean": 0.60, "val_se": 0.01},
            "gradient_boosting": {"val_mean": 0.62, "val_se": 0.01},
            "random_forest": {"val_mean": 0.70, "val_se": 0.01},
        }
        assert select_final(scores, self.ORDER)["selected"] == "random_forest"

    def test_order_must_cover_every_model(self):
        with pytest.raises(ValueError, match="simplest_first"):
            select_final({"random_forest": {"val_mean": 0.7, "val_se": 0.01}}, ("logistic_regression",))


class TestBuildPipeline:
    def test_logistic_regression_gets_a_poly_step(self, optimizer):
        pipeline = optimizer.build_pipeline("logistic_regression", "with_glucose")
        assert list(pipeline.named_steps) == ["engineer", "preprocessor", "poly", "classifier"]
        assert pipeline.named_steps["classifier"].solver == "liblinear"

    def test_ensembles_are_unchanged(self, optimizer):
        pipeline = optimizer.build_pipeline("random_forest", "with_glucose")
        assert list(pipeline.named_steps) == ["engineer", "preprocessor", "classifier"]

    def test_poly_features_are_reported(self, optimizer, train_test):
        X_train, _, y_train, _ = train_test
        pipeline = optimizer.build_pipeline("logistic_regression", "without_glucose")
        pipeline.set_params(poly__degree=2).fit(X_train, y_train)
        assert "bmi^2" in ModelTraining.used_feature_names(pipeline)


class TestTune:
    def test_covers_every_model_and_variant(self, optimizer, train_test):
        X_train, _, y_train, _ = train_test
        models, results = optimizer.tune(X_train, y_train)
        expected = {f"{m}__{v}" for m in ModelOptimizer.MODELS_SIMPLEST_FIRST for v in ModelTraining.GLUCOSE_VARIANTS}
        assert set(models) == set(results) == expected
        assert all(result["n_candidates"] <= 2 for result in results.values())

    def test_is_reproducible(self, base_config, data_prep, train_test):
        X_train, _, y_train, _ = train_test
        _, first = ModelOptimizer(base_config, data_prep).tune(X_train, y_train)
        _, second = ModelOptimizer(base_config, data_prep).tune(X_train, y_train)
        assert first == second


def test_nested_cv_gives_one_score_per_outer_fold(base_config, optimizer, train_test):
    X_train, _, y_train, _ = train_test
    result = optimizer.nested_cv("logistic_regression", "with_glucose", X_train, y_train)
    assert len(result["fold_scores"]) == base_config["cv"]
    assert result["mean"] == pytest.approx(np.mean(result["fold_scores"]))


def test_learning_curves_grow_to_the_training_folds(base_config, optimizer, train_test):
    X_train, _, y_train, _ = train_test
    pipeline = optimizer.build_pipeline("logistic_regression", "with_glucose")
    curve = optimizer.learning_curves(pipeline, X_train, y_train)
    assert len(curve["train_sizes"]) == len(curve["val_mean"]) == base_config["optimization"]["learning_curve_points"]
    assert curve["train_sizes"][0] < curve["train_sizes"][-1]


def test_optimize_reports_before_after_and_a_selection(base_config, data_prep, optimizer, train_test):
    X_train, _, y_train, _ = train_test
    reference = {
        **ModelTraining(base_config, data_prep).tune_models(X_train, y_train)[0],
        **AdvancedModelTraining(base_config, data_prep).tune_models(X_train, y_train)[0],
    }
    models, report = optimizer.optimize(reference, X_train, y_train)

    assert set(report) == set(ModelTraining.GLUCOSE_VARIANTS)
    for variant, variant_report in report.items():
        assert variant_report["selection"]["selected"] in variant_report["models"]
        assert variant_report["selection"]["selected"].endswith(variant)
        for model_report in variant_report["models"].values():
            for side in ("before", "after"):
                assert model_report[side]["diagnosis"] in {"overfit", "underfit", "good_fit"}
                assert model_report[side]["gap"] == pytest.approx(
                    model_report[side]["train_mean"] - model_report[side]["val_mean"],
                )
