import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.advanced_models import AdvancedModelTraining, compare_to_baseline
from src.data_preparation import DataPreparation
from src.evaluation import CI_METRICS, bootstrap_ci, metrics_from_scores
from src.models import ModelTraining


@pytest.fixture
def advanced(base_config):
    return AdvancedModelTraining(base_config, DataPreparation(base_config))


@pytest.fixture
def train_test(base_config, learnable_df):
    data_prep = DataPreparation(base_config)
    return data_prep.split_data(data_prep.clean_data(learnable_df))


class TestAdvancedModelTraining:
    def test_two_distinct_architectures(self, advanced):
        assert AdvancedModelTraining.MODEL_NAMES == ("random_forest", "gradient_boosting")
        assert isinstance(advanced.build_pipeline("random_forest", "with_glucose")[-1], RandomForestClassifier)
        assert isinstance(
            advanced.build_pipeline("gradient_boosting", "with_glucose")[-1], HistGradientBoostingClassifier,
        )

    def test_shares_the_baseline_machinery(self):
        assert issubclass(AdvancedModelTraining, ModelTraining)
        assert AdvancedModelTraining.GLUCOSE_VARIANTS == ModelTraining.GLUCOSE_VARIANTS

    @pytest.mark.parametrize("model_name", AdvancedModelTraining.MODEL_NAMES)
    def test_uses_balanced_class_weight_and_fixed_seed(self, base_config, advanced, model_name):
        classifier = advanced.build_pipeline(model_name, "with_glucose").named_steps["classifier"]
        assert classifier.class_weight == "balanced"
        assert classifier.random_state == base_config["random_state"]

    def test_baseline_model_is_not_an_advanced_model(self, advanced):
        with pytest.raises(ValueError, match="Unknown model"):
            advanced.build_pipeline("logistic_regression", "with_glucose")

    def test_tunes_both_models_on_both_variants_with_raw_features(self, advanced, train_test):
        X_train, _, y_train, _ = train_test
        models, results = advanced.tune_models(X_train, y_train)
        expected = {f"{m}__{v}" for m in AdvancedModelTraining.MODEL_NAMES for v in ModelTraining.GLUCOSE_VARIANTS}
        assert set(models) == set(results) == expected
        for key, model in models.items():
            names = ModelTraining.used_feature_names(model)
            assert "sqrt_physical_activity" not in names
            assert ("fasting_glucose_level" in names) == key.endswith("__with_glucose")


class TestCompareToBaseline:
    @pytest.fixture
    def results(self):
        return {
            "baseline": {"fold_scores": {"average_precision": [0.60, 0.70, 0.65, 0.55, 0.50]}},
            "better": {"fold_scores": {"average_precision": [0.65, 0.72, 0.70, 0.60, 0.52]}},
            "mixed": {"fold_scores": {"average_precision": [0.62, 0.66, 0.66, 0.53, 0.51]}},
        }

    def test_paired_differences(self, results):
        comparison = compare_to_baseline(results, "baseline", "average_precision")
        assert set(comparison) == {"better", "mixed"}
        assert comparison["better"]["mean_diff"] == pytest.approx(np.mean([0.05, 0.02, 0.05, 0.05, 0.02]))
        assert comparison["better"]["n_folds_improved"] == 5
        assert comparison["mixed"]["n_folds_improved"] == 3
        assert comparison["better"]["paired_t_p_value"] < comparison["mixed"]["paired_t_p_value"]

    def test_identical_scores_give_p_value_one(self, results):
        results["same"] = {"fold_scores": {"average_precision": results["baseline"]["fold_scores"]["average_precision"]}}
        same = compare_to_baseline(results, "baseline", "average_precision")["same"]
        assert same["mean_diff"] == 0
        assert same["paired_t_p_value"] == 1.0

    def test_missing_baseline_raises(self, results):
        with pytest.raises(ValueError, match="Baseline 'lr' not found"):
            compare_to_baseline(results, "lr", "average_precision")

    def test_unequal_fold_counts_raise(self, results):
        results["short"] = {"fold_scores": {"average_precision": [0.6, 0.7]}}
        with pytest.raises(ValueError, match="folds"):
            compare_to_baseline(results, "baseline", "average_precision")


class TestBootstrapCI:
    @pytest.fixture
    def scored(self):
        rng = np.random.default_rng(0)
        y_true = pd.Series(np.r_[np.ones(30), np.zeros(170)].astype(int))
        y_score = np.clip(0.4 * y_true.to_numpy() + 0.6 * rng.random(200), 0, 1)
        return y_true, y_score

    def test_interval_brackets_point_estimate(self, scored):
        ci = bootstrap_ci(*scored, threshold=0.5, n_boot=200)
        point = metrics_from_scores(*scored, threshold=0.5)
        assert set(ci) == set(CI_METRICS)
        for metric, bounds in ci.items():
            assert bounds["estimate"] == pytest.approx(point[metric])
            assert bounds["lower"] <= bounds["estimate"] <= bounds["upper"]

    def test_reproducible_with_seed(self, scored):
        assert bootstrap_ci(*scored, n_boot=100, random_state=3) == bootstrap_ci(*scored, n_boot=100, random_state=3)

    def test_lower_confidence_gives_narrower_interval(self, scored):
        wide = bootstrap_ci(*scored, n_boot=300, confidence=0.95)["roc_auc"]
        narrow = bootstrap_ci(*scored, n_boot=300, confidence=0.5)["roc_auc"]
        assert narrow["upper"] - narrow["lower"] < wide["upper"] - wide["lower"]
