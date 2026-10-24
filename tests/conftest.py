import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def base_config():
    """A valid pipeline config, shared across test modules."""
    return {
        "file_path": "./data/synthetic_data_166.csv",
        "target_column": "diabetes_diagnosis",
        "positive_class": "Positive",
        "min_age": 18,
        "test_size": 0.2,
        "random_state": 42,
        "cv": 5,
        "numeric_features": [
            "age", "bmi", "blood_pressure", "fasting_glucose_level", "cholesterol_level",
            "physical_activity_frequency",
        ],
        "binary_features": ["family_history_of_diabetes"],
        "categorical_levels": {
            "family_history_of_diabetes": ["Absent", "Present"],
            "diabetes_diagnosis": ["Negative", "Positive"],
        },
        "leakage_candidate_features": ["fasting_glucose_level"],
        "lr_feature_swaps": {"physical_activity_frequency": "sqrt_physical_activity"},
        "lr_extra_features": ["family_history_x_bmi"],
        "plausible_ranges": {
            "age": [18, 100],
            "bmi": [12, 70],
            "blood_pressure": [70, 220],
            "fasting_glucose_level": [50, 400],
            "cholesterol_level": [80, 400],
            "physical_activity_frequency": [0, 1],
        },
    }


@pytest.fixture
def raw_df():
    """
    Hand-built raw rows, each exercising one case from the EDA: a minor (row 1),
    an age above 100 that is kept (row 2), a padded category level (row 3),
    an exact duplicate of row 0 (row 5) and a missing target (row 6).
    """
    return pd.DataFrame({
        "age": [45, 16, 104, 60, 30, 45, 50],
        "bmi": [25.0, 22.0, 37.0, 31.0, 11.8, 25.0, 28.0],
        "blood_pressure": [120.0, 105.0, 164.0, 140.0, 107.0, 120.0, 130.0],
        "fasting_glucose_level": [90.0, 89.0, 114.0, 120.0, 51.0, 90.0, 100.0],
        "cholesterol_level": [200.0, 201.0, 208.0, 250.0, 138.0, 200.0, 210.0],
        "physical_activity_frequency": [0.25, 0.16, 0.15, 0.04, 0.43, 0.25, 0.30],
        "family_history_of_diabetes": ["Present", "Present", "Absent", " Absent ", "Absent", "Present", "Absent"],
        "diabetes_diagnosis": ["Positive", "Positive", "Negative", "Negative", "Positive", "Positive", np.nan],
    })


@pytest.fixture
def synthetic_df():
    """~300 clean raw rows shaped like the real CSV (about 15% Positive), for split and preprocessor tests."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "bmi": rng.normal(30, 5, n).round(2),
        "blood_pressure": rng.normal(130, 15, n).round(2),
        "fasting_glucose_level": rng.normal(105, 15, n).round(2),
        "cholesterol_level": rng.normal(206, 35, n).round(2),
        "physical_activity_frequency": rng.uniform(0.01, 0.8, n).round(3),
        "family_history_of_diabetes": rng.choice(["Absent", "Present"], n, p=[0.6, 0.4]),
        "diabetes_diagnosis": rng.choice(["Negative", "Positive"], n, p=[0.85, 0.15]),
    })
