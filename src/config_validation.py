"""Validation for the pipeline's YAML configuration."""

from typing import Any, Dict

from sklearn.metrics import get_scorer_names

from src.feature_engineering import FEATURE_SPECS

REQUIRED_KEYS = {
    "file_path": str,
    "target_column": str,
    "positive_class": str,
    "min_age": (int, float),
    "test_size": float,
    "random_state": int,
    "cv": int,
    "numeric_features": list,
    "binary_features": list,
    "categorical_levels": dict,
    "leakage_candidate_features": list,
    "lr_engineered_features": list,
    "candidate_features": list,
    "selection_scoring": str,
    "min_cv_gain": (int, float),
    "plausible_ranges": dict,
}

FEATURE_ROLE_KEYS = ("numeric_features", "binary_features")

ENGINEERED_FEATURES = tuple(FEATURE_SPECS)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the pipeline configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
    Raises:
        ValueError: If a required key is missing, has the wrong type, or an
            out-of-range value, or if the feature roles are inconsistent.
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")

    for key, expected_type in REQUIRED_KEYS.items():
        value = config[key]
        # bool is a subclass of int, so reject it explicitly.
        if not isinstance(value, expected_type) or isinstance(value, bool):
            types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
            raise ValueError(
                f"Config key '{key}' must be of type {' or '.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__}."
            )

    if not 0.0 < config["test_size"] < 1.0:
        raise ValueError(f"Config key 'test_size' must be in (0, 1), got {config['test_size']}.")

    if config["cv"] < 2:
        raise ValueError(f"Config key 'cv' must be >= 2, got {config['cv']}.")

    if config["min_cv_gain"] < 0:
        raise ValueError(f"Config key 'min_cv_gain' must be >= 0, got {config['min_cv_gain']}.")

    if config["selection_scoring"] not in get_scorer_names():
        raise ValueError(f"Config key 'selection_scoring' must be a scikit-learn scorer, got '{config['selection_scoring']}'.")

    if config["min_age"] < 0:
        raise ValueError(f"Config key 'min_age' must be >= 0, got {config['min_age']}.")

    for key in (*FEATURE_ROLE_KEYS, "leakage_candidate_features", "lr_engineered_features", "candidate_features"):
        if not all(isinstance(item, str) for item in config[key]):
            raise ValueError(f"Config key '{key}' must be a list of strings.")

    numeric, binary = config["numeric_features"], config["binary_features"]
    if not numeric and not binary:
        raise ValueError("Config must list at least one feature in numeric_features or binary_features.")

    overlap = sorted(set(numeric) & set(binary))
    if overlap:
        raise ValueError(f"Feature(s) {overlap} are listed in both 'numeric_features' and 'binary_features'.")

    target = config["target_column"]
    if target in numeric or target in binary:
        raise ValueError(f"Target column '{target}' must not be listed as a feature.")

    not_numeric = sorted(set(config["leakage_candidate_features"]) - set(numeric))
    if not_numeric:
        raise ValueError(f"Leakage candidate feature(s) {not_numeric} must be listed in 'numeric_features'.")

    for key in ("lr_engineered_features", "candidate_features"):
        unknown = sorted(set(config[key]) - set(ENGINEERED_FEATURES))
        if unknown:
            raise ValueError(f"{key} {unknown} must be engineered features: {list(ENGINEERED_FEATURES)}.")
        for name in config[key]:
            missing_sources = sorted(set(FEATURE_SPECS[name].sources) - set(numeric) - set(binary))
            if missing_sources:
                raise ValueError(f"{key} feature '{name}' needs feature(s) {missing_sources}, which are not configured.")

    levels = config["categorical_levels"]
    for column in (*binary, target):
        column_levels = levels.get(column)
        if not isinstance(column_levels, list) or len(column_levels) != 2 or len(set(column_levels)) != 2:
            raise ValueError(f"'categorical_levels.{column}' must list exactly 2 distinct levels.")

    if config["positive_class"] not in levels[target]:
        raise ValueError(
            f"positive_class '{config['positive_class']}' must be one of the target levels {levels[target]}."
        )

    for column, bounds in config["plausible_ranges"].items():
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds)
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"'plausible_ranges.{column}' must be [low, high] numbers with low < high.")
