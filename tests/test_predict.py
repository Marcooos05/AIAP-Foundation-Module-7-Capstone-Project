import pytest

from src.predict import VARIANTS, InvalidInputError, ModelNotAvailableError, RiskPredictor


@pytest.fixture
def load(base_config, saved_artifacts):
    models_dir, reports_dir = saved_artifacts

    def _load(variant="with_glucose"):
        return RiskPredictor.load(variant, config=base_config, models_dir=models_dir, reports_dir=reports_dir)
    return _load


class TestLoad:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_loads_pipeline_and_threshold(self, load, variant):
        predictor = load(variant)
        assert predictor.threshold == 0.4
        assert predictor.model_name == f"logistic_regression__{variant}"

    def test_unknown_variant_raises(self, base_config):
        with pytest.raises(ValueError, match="Unknown variant"):
            RiskPredictor.load("glucose_only", config=base_config)

    def test_missing_model_says_how_to_fix_it(self, base_config, tmp_path):
        with pytest.raises(ModelNotAvailableError, match="python main.py"):
            RiskPredictor.load("with_glucose", config=base_config, models_dir=tmp_path, reports_dir=tmp_path)

    def test_missing_variant_in_report_raises(self, base_config, saved_artifacts):
        models_dir, reports_dir = saved_artifacts
        (reports_dir / "selected_models.json").write_text("{}")
        with pytest.raises(ModelNotAvailableError, match="not in"):
            RiskPredictor.load("with_glucose", config=base_config, models_dir=models_dir, reports_dir=reports_dir)


class TestPredict:
    def test_returns_score_flag_and_warnings(self, load, valid_record):
        result = load().predict([valid_record])[0]
        assert 0.0 <= result["risk_score"] <= 1.0
        assert isinstance(result["flagged"], bool)
        assert result["warnings"] == []

    def test_returns_one_result_per_record_in_order(self, load, valid_record):
        other = {**valid_record, "fasting_glucose_level": 140.0}
        results = load().predict([valid_record, other])
        assert [r["risk_score"] for r in results] == [load().predict([r])[0]["risk_score"] for r in (valid_record, other)]

    @pytest.mark.parametrize("threshold, expected", [(0.0, True), (1.01, False)])
    def test_flag_applies_the_threshold(self, load, valid_record, threshold, expected):
        predictor = load()
        predictor.threshold = threshold
        assert predictor.predict([valid_record])[0]["flagged"] is expected

    def test_without_glucose_needs_no_glucose(self, load, valid_record):
        del valid_record["fasting_glucose_level"]
        assert "fasting_glucose_level" not in load("without_glucose").required_columns
        assert 0.0 <= load("without_glucose").predict([valid_record])[0]["risk_score"] <= 1.0

    def test_without_glucose_ignores_a_supplied_glucose_value(self, load, valid_record):
        predictor = load("without_glucose")
        low = predictor.predict([{**valid_record, "fasting_glucose_level": 60.0}])
        high = predictor.predict([{**valid_record, "fasting_glucose_level": 140.0}])
        assert low[0]["risk_score"] == high[0]["risk_score"]

    def test_implausible_value_warns_but_still_scores(self, load, valid_record):
        result = load().predict([{**valid_record, "age": 104}])[0]
        assert 0.0 <= result["risk_score"] <= 1.0
        assert "age=104" in result["warnings"][0]

    def test_padded_level_is_accepted(self, load, valid_record):
        assert load().predict([{**valid_record, "family_history_of_diabetes": " Present "}])


class TestValidation:
    def test_no_records_raises(self, load):
        with pytest.raises(InvalidInputError, match="No records"):
            load().predict([])

    def test_missing_field_raises(self, load, valid_record):
        del valid_record["bmi"]
        with pytest.raises(InvalidInputError, match="Missing required field"):
            load().predict([valid_record])

    def test_missing_glucose_raises_for_with_glucose(self, load, valid_record):
        valid_record["fasting_glucose_level"] = None
        with pytest.raises(InvalidInputError, match="fasting_glucose_level"):
            load().predict([valid_record])

    @pytest.mark.parametrize("value", ["high", float("inf")])
    def test_non_numeric_raises(self, load, valid_record, value):
        with pytest.raises(InvalidInputError, match="finite number"):
            load().predict([{**valid_record, "bmi": value}])

    def test_unknown_level_raises(self, load, valid_record):
        with pytest.raises(InvalidInputError, match="Unexpected value"):
            load().predict([{**valid_record, "family_history_of_diabetes": "Unknown"}])

    def test_below_min_age_raises(self, load, valid_record):
        with pytest.raises(InvalidInputError, match="minimum age"):
            load().predict([{**valid_record, "age": 16}])
