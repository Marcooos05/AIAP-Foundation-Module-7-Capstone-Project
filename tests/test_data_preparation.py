import numpy as np
import pandas as pd
import pytest

from src.data_preparation import DataPreparation


class TestGenerateValidationReport:
    def test_reports_data_quality_counts(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["n_rows"] == 7
        assert report["n_duplicate_rows"] == 1
        assert report["missing_values_per_column"] == {"diabetes_diagnosis": 1}
        assert report["n_below_min_age"] == 1

    def test_reports_out_of_range_values(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["n_outside_plausible_range"]["age"] == {"below": 1, "above": 1}
        assert report["n_outside_plausible_range"]["bmi"] == {"below": 1, "above": 0}

    def test_reports_unexpected_levels(self, base_config, raw_df):
        raw_df.loc[0, "family_history_of_diabetes"] = "Unknown"
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["unexpected_categorical_levels"] == {"family_history_of_diabetes": {"Unknown": 1}}

    def test_padded_level_is_not_unexpected(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["unexpected_categorical_levels"] == {}

    def test_reports_target_distribution(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["target_distribution"] == {"Positive": 4, "Negative": 2, "nan": 1}


class TestCleanData:
    def test_drops_duplicate_missing_target_and_minor(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        # 7 rows - 1 duplicate - 1 missing target - 1 minor
        assert len(cleaned) == 4
        assert cleaned["age"].min() >= 18

    def test_keeps_ages_above_100_and_extreme_bmi(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert 104 in cleaned["age"].values
        assert 11.8 in cleaned["bmi"].values

    def test_does_not_modify_input(self, base_config, raw_df):
        original = raw_df.copy()
        DataPreparation(base_config).clean_data(raw_df)
        pd.testing.assert_frame_equal(raw_df, original)

    def test_strips_whitespace_from_levels(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert set(cleaned["family_history_of_diabetes"]) == {"Absent", "Present"}

    def test_unexpected_level_raises(self, base_config, raw_df):
        raw_df.loc[0, "family_history_of_diabetes"] = "Unknown"
        with pytest.raises(ValueError, match="Unexpected value"):
            DataPreparation(base_config).clean_data(raw_df)

    def test_missing_feature_value_raises(self, base_config, raw_df):
        raw_df.loc[0, "bmi"] = np.nan
        with pytest.raises(ValueError, match="Missing feature value"):
            DataPreparation(base_config).clean_data(raw_df)

    def test_empty_dataframe_raises(self, base_config, raw_df):
        with pytest.raises(ValueError, match="empty"):
            DataPreparation(base_config).clean_data(raw_df.iloc[0:0])

    def test_missing_required_column_raises(self, base_config, raw_df):
        with pytest.raises(ValueError, match="missing required raw column"):
            DataPreparation(base_config).clean_data(raw_df.drop(columns=["bmi"]))

    def test_engineers_sqrt_activity(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        np.testing.assert_allclose(cleaned["sqrt_physical_activity"], np.sqrt(cleaned["physical_activity_frequency"]))

    def test_engineers_family_history_x_bmi(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        present = cleaned["family_history_of_diabetes"] == "Present"
        assert (cleaned.loc[present, "family_history_x_bmi"] == cleaned.loc[present, "bmi"]).all()
        assert (cleaned.loc[~present, "family_history_x_bmi"] == 0).all()

    def test_negative_activity_raises(self, base_config, raw_df):
        raw_df.loc[0, "physical_activity_frequency"] = -0.1
        with pytest.raises(ValueError, match="negative physical_activity_frequency"):
            DataPreparation(base_config).clean_data(raw_df)


class TestSplitData:
    @pytest.fixture
    def split(self, base_config, synthetic_df):
        data_prep = DataPreparation(base_config)
        cleaned = data_prep.clean_data(synthetic_df)
        return cleaned, data_prep.split_data(cleaned)

    def test_sizes_add_up(self, split):
        cleaned, (X_train, X_test, y_train, y_test) = split
        assert len(X_train) + len(X_test) == len(cleaned)
        assert len(X_test) == pytest.approx(0.2 * len(cleaned), abs=1)

    def test_target_is_binary_and_removed_from_features(self, split):
        _, (X_train, X_test, y_train, y_test) = split
        assert set(y_train.unique()) == {0, 1}
        assert "diabetes_diagnosis" not in X_train.columns

    def test_split_is_stratified(self, split):
        _, (_, _, y_train, y_test) = split
        assert y_test.mean() == pytest.approx(y_train.mean(), abs=0.02)

    def test_single_class_raises(self, base_config, synthetic_df):
        synthetic_df["diabetes_diagnosis"] = "Negative"
        data_prep = DataPreparation(base_config)
        with pytest.raises(ValueError, match="must match some, but not all"):
            data_prep.split_data(data_prep.clean_data(synthetic_df))


class TestCreatePreprocessor:
    @pytest.fixture
    def train_test(self, base_config, synthetic_df):
        data_prep = DataPreparation(base_config)
        X_train, X_test, _, _ = data_prep.split_data(data_prep.clean_data(synthetic_df))
        return X_train, X_test

    def test_returns_independent_instances(self, base_config):
        data_prep = DataPreparation(base_config)
        assert data_prep.create_preprocessor() is not data_prep.create_preprocessor()

    @pytest.mark.parametrize("variant, n_features", [
        ("with_glucose__trees", 7),
        ("with_glucose__logistic_regression", 8),
        ("without_glucose__trees", 6),
        ("without_glucose__logistic_regression", 7),
    ])
    def test_feature_variants(self, base_config, train_test, variant, n_features):
        X_train, X_test = train_test
        data_prep = DataPreparation(base_config)
        preprocessor = data_prep.create_preprocessor(**data_prep.feature_variants()[variant]).fit(X_train)
        names = list(preprocessor.get_feature_names_out())

        assert len(names) == n_features
        assert "family_history_of_diabetes_Present" in names
        assert ("fasting_glucose_level" in names) == variant.startswith("with_glucose")
        if variant.endswith("logistic_regression"):
            assert "physical_activity_frequency" not in names
            assert {"sqrt_physical_activity", "family_history_x_bmi"} <= set(names)
        else:
            assert not {"sqrt_physical_activity", "family_history_x_bmi"} & set(names)
        assert not np.isnan(preprocessor.transform(X_test)).any()

    def test_scaler_fitted_on_training_data_only(self, base_config, train_test):
        X_train, X_test = train_test
        preprocessor = DataPreparation(base_config).create_preprocessor().fit(X_train)
        scaler = preprocessor.named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, X_train[scaler.feature_names_in_].mean().values)

        Xt_train = preprocessor.transform(X_train)
        np.testing.assert_allclose(Xt_train[:, :len(scaler.mean_)].mean(axis=0), 0, atol=1e-10)

    def test_binary_feature_encoded_as_single_column(self, base_config, train_test):
        X_train, _ = train_test
        preprocessor = DataPreparation(base_config).create_preprocessor().fit(X_train)
        encoded = preprocessor.transform(X_train)[:, -1]
        expected = (X_train["family_history_of_diabetes"] == "Present").astype(int).values
        np.testing.assert_array_equal(encoded, expected)
