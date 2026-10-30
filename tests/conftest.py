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
        "cv": 3,
        "scoring": "average_precision",
        "target_recall": 0.9,
        "n_jobs": 1,
        "param_grids": {  # one or two values each, so tests stay fast
            "logistic_regression": {"classifier__C": [0.1, 1.0]},
            "random_forest": {"classifier__n_estimators": [20], "classifier__max_depth": [3]},
            "gradient_boosting": {"classifier__max_iter": [20], "classifier__max_depth": [2]},
        },
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
        "lr_engineered_features": ["sqrt_physical_activity", "family_history_x_bmi"],
        "candidate_features": [
            "sqrt_physical_activity", "family_history_x_bmi", "bmi_squared", "bmi_band", "age_band",
            "elevated_bp", "prediabetic_glucose",
        ],
        "selection_scoring": "average_precision",
        "min_cv_gain": 0.005,
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


@pytest.fixture
def learnable_df(synthetic_df):
    """
    `synthetic_df` with a label models can learn: low glucose and a family
    history raise the chance of Positive, mirroring the direction found in
    the real data (EDA §13). About 15-20% Positive.
    """
    rng = np.random.default_rng(1)
    df = synthetic_df.copy()
    logit = -0.15 * (df["fasting_glucose_level"] - 105) + 1.2 * (df["family_history_of_diabetes"] == "Present") - 2.6
    df["diabetes_diagnosis"] = np.where(rng.random(len(df)) < 1 / (1 + np.exp(-logit)), "Positive", "Negative")
    return df
