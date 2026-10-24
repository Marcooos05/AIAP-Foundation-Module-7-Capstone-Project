import pytest
import yaml

from src.config_validation import validate_config


def test_valid_config_passes(base_config):
    validate_config(base_config)


def test_project_config_file_is_valid():
    with open("src/config.yaml") as file:
        validate_config(yaml.safe_load(file))


def test_missing_required_key_raises(base_config):
    del base_config["cv"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_config(base_config)


def test_wrong_type_raises(base_config):
    base_config["random_state"] = "42"
    with pytest.raises(ValueError, match="must be of type int"):
        validate_config(base_config)


def test_bool_rejected_for_int_key(base_config):
    base_config["cv"] = True
    with pytest.raises(ValueError, match="cv"):
        validate_config(base_config)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_test_size_out_of_range_raises(base_config, value):
    base_config["test_size"] = value
    with pytest.raises(ValueError, match="test_size"):
        validate_config(base_config)


def test_cv_below_two_raises(base_config):
    base_config["cv"] = 1
    with pytest.raises(ValueError, match="cv"):
        validate_config(base_config)


def test_negative_min_age_raises(base_config):
    base_config["min_age"] = -1
    with pytest.raises(ValueError, match="min_age"):
        validate_config(base_config)


def test_feature_in_two_roles_raises(base_config):
    base_config["binary_features"].append("age")
    with pytest.raises(ValueError, match="listed in both"):
        validate_config(base_config)


def test_target_as_feature_raises(base_config):
    base_config["numeric_features"].append("diabetes_diagnosis")
    with pytest.raises(ValueError, match="must not be listed as a feature"):
        validate_config(base_config)


def test_leakage_candidate_not_numeric_raises(base_config):
    base_config["leakage_candidate_features"] = ["family_history_of_diabetes"]
    with pytest.raises(ValueError, match="Leakage candidate"):
        validate_config(base_config)


def test_unknown_swap_target_raises(base_config):
    base_config["lr_feature_swaps"] = {"physical_activity_frequency": "log_activity"}
    with pytest.raises(ValueError, match="engineered feature"):
        validate_config(base_config)


def test_swap_of_unconfigured_feature_raises(base_config):
    base_config["lr_feature_swaps"] = {"weight": "sqrt_physical_activity"}
    with pytest.raises(ValueError, match="lr_feature_swaps key"):
        validate_config(base_config)


def test_unknown_extra_feature_raises(base_config):
    base_config["lr_extra_features"] = ["age_x_bmi"]
    with pytest.raises(ValueError, match="lr_extra_features"):
        validate_config(base_config)


def test_binary_feature_without_two_levels_raises(base_config):
    base_config["categorical_levels"]["family_history_of_diabetes"] = ["Absent", "Present", "Unknown"]
    with pytest.raises(ValueError, match="exactly 2 distinct levels"):
        validate_config(base_config)


def test_positive_class_not_a_target_level_raises(base_config):
    base_config["positive_class"] = "Yes"
    with pytest.raises(ValueError, match="positive_class"):
        validate_config(base_config)


@pytest.mark.parametrize("bounds", [[100, 18], [18], ["a", "b"]])
def test_invalid_plausible_range_raises(base_config, bounds):
    base_config["plausible_ranges"]["age"] = bounds
    with pytest.raises(ValueError, match="plausible_ranges.age"):
        validate_config(base_config)
