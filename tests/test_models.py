import pytest

from src.advanced_models import AdvancedModelTraining
from src.data_preparation import DataPreparation
from src.models import ModelTraining


@pytest.fixture
def model_training(base_config):
    return ModelTraining(base_config, DataPreparation(base_config))


@pytest.fixture
def train_test(base_config, learnable_df):
    data_prep = DataPreparation(base_config)
    return data_prep.split_data(data_prep.clean_data(learnable_df))


class TestBuildPipeline:
    def test_baseline_is_logistic_regression_only(self):
        assert ModelTraining.MODEL_NAMES == ("logistic_regression",)

    def test_baseline_uses_balanced_class_weight(self, model_training):
        pipeline = model_training.build_pipeline("logistic_regression", "with_glucose")
        assert pipeline.named_steps["classifier"].class_weight == "balanced"
        assert list(pipeline.named_steps) == ["engineer", "preprocessor", "classifier"]

    def test_returns_independent_pipelines(self, model_training):
        first = model_training.build_pipeline("logistic_regression", "with_glucose")
        second = model_training.build_pipeline("logistic_regression", "with_glucose")
        assert first.named_steps["preprocessor"] is not second.named_steps["preprocessor"]

    @pytest.mark.parametrize("trainer_cls, model_name, variant, has_glucose, has_lr_features", [
        (ModelTraining, "logistic_regression", "with_glucose", True, True),
        (ModelTraining, "logistic_regression", "without_glucose", False, True),
        (AdvancedModelTraining, "random_forest", "with_glucose", True, False),
        (AdvancedModelTraining, "gradient_boosting", "without_glucose", False, False),
    ])
    def test_uses_matching_feature_variant(
        self, base_config, train_test, trainer_cls, model_name, variant, has_glucose, has_lr_features,
    ):
        X_train, _, y_train, _ = train_test
        trainer = trainer_cls(base_config, DataPreparation(base_config))
        pipeline = trainer.build_pipeline(model_name, variant).fit(X_train, y_train)
        names = ModelTraining.used_feature_names(pipeline)
        assert ("fasting_glucose_level" in names) == has_glucose
        assert ("sqrt_physical_activity" in names) == has_lr_features
        assert ("physical_activity_frequency" in names) == (not has_lr_features)

    def test_unknown_variant_raises(self, model_training):
        with pytest.raises(ValueError, match="Unknown glucose variant"):
            model_training.build_pipeline("logistic_regression", "glucose_only")

    @pytest.mark.parametrize("model_name", ["knn", "random_forest"])
    def test_unknown_model_raises(self, model_training, model_name):
        # The baseline trainer only knows the baseline; ensembles live in AdvancedModelTraining.
        with pytest.raises(ValueError, match="Unknown model"):
            model_training.build_pipeline(model_name, "with_glucose")


class TestTuneModels:
    def test_tunes_baseline_on_both_variants_and_keeps_fold_scores(self, base_config, model_training, train_test):
        X_train, _, y_train, _ = train_test
        models, results = model_training.tune_models(X_train, y_train)
        assert set(models) == set(results) == {"logistic_regression__with_glucose", "logistic_regression__without_glucose"}
        for result in results.values():
            assert len(result["fold_scores"]["average_precision"]) == base_config["cv"]
            assert result["cv_mean"] == pytest.approx(sum(result["fold_scores"]["average_precision"]) / base_config["cv"])
            assert result["mean_fit_time_s"] > 0


class TestSelectBest:
    def test_picks_highest_cv_score_within_variant(self):
        results = {
            "logistic_regression__with_glucose": {"cv_mean": 0.70},
            "random_forest__with_glucose": {"cv_mean": 0.65},
            "random_forest__without_glucose": {"cv_mean": 0.90},
        }
        assert ModelTraining.select_best(results, "with_glucose") == "logistic_regression__with_glucose"
        assert ModelTraining.select_best(results, "without_glucose") == "random_forest__without_glucose"

    def test_no_candidates_raises(self):
        with pytest.raises(ValueError, match="No tuned models"):
            ModelTraining.select_best({"random_forest__with_glucose": {"cv_mean": 0.5}}, "without_glucose")


class TestTuneThreshold:
    def test_out_of_fold_recall_meets_target(self, model_training, train_test):
        X_train, _, y_train, _ = train_test
        pipeline = model_training.build_pipeline("logistic_regression", "with_glucose")
        result = model_training.tune_threshold(pipeline, X_train, y_train)
        assert result["oof_at_threshold"]["recall"] >= 0.9
        assert 0.0 < result["threshold"] < 1.0

    def test_does_not_fit_the_given_pipeline(self, model_training, train_test):
        X_train, _, y_train, _ = train_test
        pipeline = model_training.build_pipeline("logistic_regression", "with_glucose")
        model_training.tune_threshold(pipeline, X_train, y_train)
        assert not hasattr(pipeline.named_steps["classifier"], "coef_")


class TestExplainModel:
    def test_reports_only_columns_the_model_uses(self, base_config, model_training, train_test):
        X_train, X_test, y_train, y_test = train_test
        advanced = AdvancedModelTraining(base_config, DataPreparation(base_config))
        pipeline = advanced.build_pipeline("random_forest", "without_glucose").fit(X_train, y_train)
        importance = model_training.explain_model(pipeline, X_test, y_test)
        assert "fasting_glucose_level" not in importance
        assert set(importance) == {
            "age", "bmi", "blood_pressure", "cholesterol_level", "physical_activity_frequency",
            "family_history_of_diabetes",
        }
        means = [value["mean"] for value in importance.values()]
        assert means == sorted(means, reverse=True)

    def test_engineered_features_count_toward_their_raw_sources(self, model_training, train_test):
        X_train, X_test, y_train, y_test = train_test
        pipeline = model_training.build_pipeline("logistic_regression", "with_glucose").fit(X_train, y_train)
        assert "sqrt_physical_activity" in ModelTraining.used_feature_names(pipeline)
        importance = model_training.explain_model(pipeline, X_test, y_test)
        assert set(importance) == {
            "age", "bmi", "blood_pressure", "fasting_glucose_level", "cholesterol_level",
            "physical_activity_frequency", "family_history_of_diabetes",
        }
