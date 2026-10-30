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


def test_unknown_lr_engineered_feature_raises(base_config):
    base_config["lr_engineered_features"] = ["age_x_bmi"]
    with pytest.raises(ValueError, match="lr_engineered_features"):
        validate_config(base_config)


def test_engineered_feature_from_unconfigured_column_raises(base_config):
    base_config["numeric_features"].remove("physical_activity_frequency")
    with pytest.raises(ValueError, match="needs feature"):
        validate_config(base_config)


def test_unknown_candidate_feature_raises(base_config):
    base_config["candidate_features"] = ["bmi_band", "age_x_bmi"]
    with pytest.raises(ValueError, match="candidate_features"):
        validate_config(base_config)


def test_negative_min_cv_gain_raises(base_config):
    base_config["min_cv_gain"] = -0.01
    with pytest.raises(ValueError, match="min_cv_gain"):
        validate_config(base_config)


def test_unknown_selection_scoring_raises(base_config):
    base_config["selection_scoring"] = "pr_auc"
    with pytest.raises(ValueError, match="selection_scoring"):
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


def test_unsupported_scoring_raises(base_config):
    base_config["scoring"] = "accuracy"
    with pytest.raises(ValueError, match="scoring"):
        validate_config(base_config)


@pytest.mark.parametrize("value", [0.0, -0.5, 1.1])
def test_target_recall_out_of_range_raises(base_config, value):
    base_config["target_recall"] = value
    with pytest.raises(ValueError, match="target_recall"):
        validate_config(base_config)


def test_target_recall_of_one_passes(base_config):
    base_config["target_recall"] = 1.0
    validate_config(base_config)


@pytest.mark.parametrize("value", [0, -2])
def test_invalid_n_jobs_raises(base_config, value):
    base_config["n_jobs"] = value
    with pytest.raises(ValueError, match="n_jobs"):
        validate_config(base_config)


def test_missing_param_grid_raises(base_config):
    del base_config["param_grids"]["random_forest"]
    with pytest.raises(ValueError, match="missing an entry"):
        validate_config(base_config)


def test_unknown_param_grid_model_raises(base_config):
    base_config["param_grids"]["knn"] = {"classifier__n_neighbors": [5]}
    with pytest.raises(ValueError, match="Unknown model 'knn'"):
        validate_config(base_config)


@pytest.mark.parametrize("grid", [{"classifier__C": []}, {"classifier__C": 1.0}, [1.0]])
def test_invalid_param_grid_raises(base_config, grid):
    base_config["param_grids"]["logistic_regression"] = grid
    with pytest.raises(ValueError, match="param_grids.logistic_regression"):
        validate_config(base_config)


def test_missing_search_space_raises(base_config):
    del base_config["search_spaces"]["gradient_boosting"]
    with pytest.raises(ValueError, match="'search_spaces' is missing an entry"):
        validate_config(base_config)


def test_invalid_search_space_raises(base_config):
    base_config["search_spaces"]["random_forest"] = {"classifier__max_depth": []}
    with pytest.raises(ValueError, match="search_spaces.random_forest"):
        validate_config(base_config)


@pytest.mark.parametrize("key, value", [("cv_repeats", 0), ("n_iter", 1.5), ("nested_n_iter", True)])
def test_invalid_optimization_count_raises(base_config, key, value):
    base_config["optimization"][key] = value
    with pytest.raises(ValueError, match=f"optimization.{key}"):
        validate_config(base_config)


def test_learning_curve_needs_two_points(base_config):
    base_config["optimization"]["learning_curve_points"] = 1
    with pytest.raises(ValueError, match="learning_curve_points"):
        validate_config(base_config)


@pytest.mark.parametrize("value", [0, -0.1, "high"])
def test_invalid_max_fit_gap_raises(base_config, value):
    base_config["optimization"]["max_fit_gap"] = value
    with pytest.raises(ValueError, match="max_fit_gap"):
        validate_config(base_config)
