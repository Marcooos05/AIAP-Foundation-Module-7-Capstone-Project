import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline

from src.data_preparation import DataPreparation
from src.feature_engineering import (
    FEATURE_SPECS,
    ClinicalFeatureEngineer,
    compare_feature_sets,
    features_derived_from,
    l1_selection,
    passes_retention_rule,
    permutation_ranking,
)


@pytest.fixture
def edge_df():
    """Rows on and around every clinical cut-off."""
    return pd.DataFrame({
        "age": [18, 34, 35, 44, 45, 64, 65],
        "bmi": [18.4, 18.5, 24.9, 25.0, 29.9, 30.0, 40.0],
        "blood_pressure": [110.0, 129.9, 130.0, 140.0, 100.0, 129.0, 131.0],
        "fasting_glucose_level": [80.0, 99.9, 100.0, 126.0, 90.0, 99.0, 101.0],
        "cholesterol_level": [200.0] * 7,
        "physical_activity_frequency": [0.0, 0.04, 0.25, 0.49, 0.81, 1.0, 0.16],
        "family_history_of_diabetes": ["Present", "Absent", "Present", "Absent", "Absent", "Present", "Absent"],
    })


def _transform(df, features):
    return ClinicalFeatureEngineer(features=features).fit_transform(df)


class TestClinicalFeatureEngineer:
    def test_sqrt_physical_activity(self, edge_df):
        out = _transform(edge_df, ["sqrt_physical_activity"])
        np.testing.assert_allclose(out["sqrt_physical_activity"], [0, 0.2, 0.5, 0.7, 0.9, 1.0, 0.4])

    def test_family_history_x_bmi(self, edge_df):
        out = _transform(edge_df, ["family_history_x_bmi"])
        present = edge_df["family_history_of_diabetes"] == "Present"
        np.testing.assert_allclose(out["family_history_x_bmi"], np.where(present, edge_df["bmi"], 0))

    def test_bmi_squared(self, edge_df):
        np.testing.assert_allclose(_transform(edge_df, ["bmi_squared"])["bmi_squared"], edge_df["bmi"] ** 2)

    def test_bmi_band_uses_who_cutoffs(self, edge_df):
        assert _transform(edge_df, ["bmi_band"])["bmi_band"].tolist() == [0, 1, 1, 2, 2, 3, 3]

    def test_age_band_uses_ada_cutoffs(self, edge_df):
        assert _transform(edge_df, ["age_band"])["age_band"].tolist() == [0, 0, 1, 1, 2, 2, 3]

    def test_elevated_bp_flag(self, edge_df):
        assert _transform(edge_df, ["elevated_bp"])["elevated_bp"].tolist() == [0, 0, 1, 1, 0, 0, 1]

    def test_prediabetic_glucose_flag(self, edge_df):
        assert _transform(edge_df, ["prediabetic_glucose"])["prediabetic_glucose"].tolist() == [0, 0, 1, 1, 0, 0, 1]

    def test_adds_all_features_by_default_after_the_raw_columns(self, edge_df):
        out = ClinicalFeatureEngineer().fit_transform(edge_df)
        assert list(out.columns) == [*edge_df.columns, *FEATURE_SPECS]

    def test_empty_feature_list_is_a_no_op(self, edge_df):
        pd.testing.assert_frame_equal(_transform(edge_df, []), edge_df)

    def test_does_not_modify_input(self, edge_df):
        before = edge_df.copy()
        ClinicalFeatureEngineer().fit_transform(edge_df)
        pd.testing.assert_frame_equal(edge_df, before)

    def test_feature_names_out_match_transform(self, edge_df):
        engineer = ClinicalFeatureEngineer(features=["bmi_band", "elevated_bp"]).fit(edge_df)
        assert list(engineer.get_feature_names_out()) == list(engineer.transform(edge_df).columns)

    def test_is_a_cloneable_sklearn_estimator(self):
        engineer = ClinicalFeatureEngineer(features=["bmi_band"])
        assert clone(engineer).get_params() == {"features": ["bmi_band"]}

    def test_works_inside_a_pipeline(self, base_config, synthetic_df):
        data_prep = DataPreparation(base_config)
        X_train, X_test, y_train, _ = data_prep.split_data(data_prep.clean_data(synthetic_df))
        pipeline = Pipeline([
            *data_prep.create_feature_pipeline(engineered=["bmi_band", "elevated_bp"]).steps,
            ("model", LogisticRegression()),
        ]).fit(X_train, y_train)
        assert pipeline.predict_proba(X_test).shape == (len(X_test), 2)

    def test_transform_before_fit_raises(self, edge_df):
        with pytest.raises(Exception, match="not fitted"):
            ClinicalFeatureEngineer().transform(edge_df)

    def test_unknown_feature_raises(self, edge_df):
        with pytest.raises(ValueError, match="Unknown engineered feature"):
            _transform(edge_df, ["age_x_bmi"])

    def test_missing_source_column_raises(self, edge_df):
        with pytest.raises(ValueError, match="missing column"):
            _transform(edge_df.drop(columns=["blood_pressure"]), ["elevated_bp"])

    def test_missing_value_raises(self, edge_df):
        edge_df.loc[0, "bmi"] = np.nan
        with pytest.raises(ValueError, match="Missing value"):
            _transform(edge_df, ["bmi_band"])

    def test_negative_activity_raises(self, edge_df):
        edge_df.loc[0, "physical_activity_frequency"] = -0.1
        with pytest.raises(ValueError, match="negative physical_activity_frequency"):
            _transform(edge_df, ["sqrt_physical_activity"])

    def test_rejects_non_dataframe(self, edge_df):
        with pytest.raises(TypeError, match="DataFrame"):
            ClinicalFeatureEngineer().fit(edge_df.to_numpy())


def test_features_derived_from_glucose():
    assert features_derived_from(["fasting_glucose_level"]) == ["prediabetic_glucose"]
    assert features_derived_from(["bmi"], ["sqrt_physical_activity", "bmi_band"]) == ["bmi_band"]


class TestSelectionAndComparison:
    @pytest.fixture
    def train(self, base_config, synthetic_df):
        data_prep = DataPreparation(base_config)
        X_train, _, y_train, _ = data_prep.split_data(data_prep.clean_data(synthetic_df))
        return data_prep, X_train, y_train

    def _lr(self, data_prep, engineered):
        return Pipeline([
            *data_prep.create_feature_pipeline(engineered=engineered).steps,
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])

    def test_l1_selection_covers_every_model_input(self, train):
        data_prep, X_train, y_train = train
        pipeline = Pipeline([
            *data_prep.create_feature_pipeline(engineered=["bmi_band"]).steps,
            ("model", LogisticRegressionCV(l1_ratios=(1.0,), solver="liblinear", cv=3, use_legacy_attributes=False)),
        ])
        result = l1_selection(pipeline, X_train, y_train)
        assert set(result["kept"]) | set(result["dropped"]) == set(result["coefficients"])
        assert "bmi_band" in result["coefficients"]
        assert all(result["coefficients"][name] != 0 for name in result["kept"])

    def test_permutation_ranking_is_sorted_and_reproducible(self, train):
        data_prep, X_train, y_train = train
        pipeline = self._lr(data_prep, ["bmi_band"])
        first = permutation_ranking(pipeline, X_train, y_train, cv=3, random_state=0, n_repeats=2)
        second = permutation_ranking(pipeline, X_train, y_train, cv=3, random_state=0, n_repeats=2)
        assert first == second
        assert len(first) == 8
        means = [v["mean"] for v in first.values()]
        assert means == sorted(means, reverse=True)

    def test_compare_feature_sets_gains_are_relative_to_baseline(self, train):
        data_prep, X_train, y_train = train
        results = compare_feature_sets(
            {"baseline": self._lr(data_prep, []), "+bmi_band": self._lr(data_prep, ["bmi_band"])},
            X_train, y_train, cv=3, random_state=0,
        )
        assert results["baseline"]["gain_vs_baseline"] == 0
        assert results["baseline"]["n_folds_improved"] == 0
        assert {"average_precision_mean", "roc_auc_mean", "recall_mean"} <= set(results["+bmi_band"])
        assert results["+bmi_band"]["gain_vs_baseline"] == pytest.approx(
            results["+bmi_band"]["average_precision_mean"] - results["baseline"]["average_precision_mean"], abs=2e-4,
        )

    def test_compare_feature_sets_requires_baseline(self, train):
        data_prep, X_train, y_train = train
        with pytest.raises(ValueError, match="Baseline"):
            compare_feature_sets({"+bmi_band": self._lr(data_prep, ["bmi_band"])}, X_train, y_train, 3, 0)


@pytest.mark.parametrize("gain, n_improved, expected", [
    (0.01, 3, True), (0.01, 2, False), (0.001, 5, False), (0.005, 3, True),
])
def test_retention_rule(gain, n_improved, expected):
    result = {"gain_vs_baseline": gain, "n_folds_improved": n_improved}
    assert passes_retention_rule(result, n_folds=5, min_gain=0.005) is expected
