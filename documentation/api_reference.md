# API reference

For developers calling the model. To run the service, see the [Deployment guide](deployment_guide.md).

> **Not for clinical use.** Scores reflect a synthetic dataset whose label runs against clinical knowledge. See [Limitations](model_explanation.md#limitations).

## REST API

The base URL is `http://localhost:8000` when run locally. There's no authentication yet, so don't expose the service publicly (see [Security](deployment_guide.md#security)).

Interactive documentation is generated at `/docs`, and the OpenAPI schema is at `/openapi.json`.

### `GET /health`

Reports whether both models are loaded.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "models": {"with_glucose": true, "without_glucose": true}}
```

| Status | `status` | Meaning |
|---|---|---|
| 200 | `ok` | Both models are loaded |
| 503 | `degraded` | At least one model is missing. `models` shows which. Run `python main.py` |

### `POST /predict`

Scores 1 to 1,000 records with one model.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `variant` | string | No | `with_glucose` (default) or `without_glucose` |
| `records` | array | Yes | 1 to 1,000 records |

**Record fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `age` | number | Yes | Years. Must be at least 18 |
| `bmi` | number | Yes | kg/m² |
| `blood_pressure` | number | Yes | Assumed systolic, mmHg |
| `fasting_glucose_level` | number | `with_glucose` only | mg/dL. Ignored by `without_glucose` |
| `cholesterol_level` | number | Yes | Assumed total, mg/dL |
| `physical_activity_frequency` | number | Yes | Proportion from 0 to 1 |
| `family_history_of_diabetes` | string | Yes | `Absent` or `Present` |

Units are inferred, as the dataset doesn't document them. Unknown fields are rejected, which catches misspelled names.

**Response fields**

| Field | Type | Description |
|---|---|---|
| `variant` | string | Variant used |
| `model` | string | Model that produced the scores |
| `threshold` | number | Decision threshold of that model |
| `predictions` | array | One entry per record, in request order |
| `predictions[].risk_score` | number | Score from 0 to 1. It ranks people, and is **not** a calibrated probability |
| `predictions[].flagged` | boolean | `true` if `risk_score` ≥ `threshold`: refer for follow-up |
| `predictions[].warnings` | array | Values outside the plausible ranges. The record is still scored |
| `notice` | string | The not-for-clinical-use notice |

**Example: with glucose**

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "variant": "with_glucose",
    "records": [{
      "age": 50, "bmi": 28.0, "blood_pressure": 125.0, "fasting_glucose_level": 95.0,
      "cholesterol_level": 200.0, "physical_activity_frequency": 0.3,
      "family_history_of_diabetes": "Present"
    }]
  }'
```

```json
{
  "variant": "with_glucose",
  "model": "logistic_regression__with_glucose",
  "threshold": 0.41845013187363406,
  "predictions": [
    {"risk_score": 0.6605342767918021, "flagged": true, "warnings": []}
  ],
  "notice": "Technical demonstration only, not for clinical use. In this synthetic dataset the label runs against clinical knowledge for 6 of 7 features, so scores are not real diabetes risk."
}
```

**Example: without glucose, batch with a warning**

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "variant": "without_glucose",
    "records": [
      {"age": 50, "bmi": 28.0, "blood_pressure": 125.0, "cholesterol_level": 200.0,
       "physical_activity_frequency": 0.3, "family_history_of_diabetes": "Absent"},
      {"age": 104, "bmi": 37.0, "blood_pressure": 164.0, "cholesterol_level": 208.0,
       "physical_activity_frequency": 0.15, "family_history_of_diabetes": "Absent"}
    ]
  }'
```

```json
{
  "variant": "without_glucose",
  "model": "logistic_regression__without_glucose",
  "threshold": 0.40405721466180566,
  "predictions": [
    {"risk_score": 0.41353813770627806, "flagged": true, "warnings": []},
    {"risk_score": 0.007565907001056425, "flagged": false,
     "warnings": ["age=104.0 is outside the plausible range [18, 100]"]}
  ],
  "notice": "Technical demonstration only, not for clinical use. In this synthetic dataset the label runs against clinical knowledge for 6 of 7 features, so scores are not real diabetes risk."
}
```

**Example: Python client**

This needs `httpx`, which is in `requirements-dev.txt`.

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict",
    json={"variant": "without_glucose", "records": [{
        "age": 50, "bmi": 28.0, "blood_pressure": 125.0, "cholesterol_level": 200.0,
        "physical_activity_frequency": 0.3, "family_history_of_diabetes": "Absent",
    }]},
    timeout=10,
)
response.raise_for_status()
print(response.json()["predictions"][0])
```

### Errors

| Status | When | Body |
|---|---|---|
| 422 | Malformed request: wrong type, unknown field or variant, NaN, or an empty or oversized batch | `detail` is a list of `type`, `loc`, `msg` |
| 422 | Valid types, but the record can't be scored: below the minimum age, or glucose missing for `with_glucose` | `detail` is a message |
| 503 | The requested model isn't available | `detail` is a message |

Error responses never repeat the values submitted, because they may be health data.

Wrong type:

```json
{"detail": [{"type": "float_parsing", "loc": ["body", "records", 0, "age"], "msg": "Input should be a valid number, unable to parse string as a number"}]}
```

Below the minimum age:

```json
{"detail": "1 record(s) below the minimum age of 18: out of scope."}
```

Glucose missing for `with_glucose`:

```json
{"detail": "Missing value(s) per field: {'fasting_glucose_level': 1}."}
```

A single invalid record fails the whole batch, so no partial results are returned.

## Python API

Use these from the project root, so that `src` is importable.

### Scoring: `src.predict`

```python
from src.predict import RiskPredictor

predictor = RiskPredictor.load("with_glucose")
results = predictor.predict([{
    "age": 50, "bmi": 28.0, "blood_pressure": 125.0, "fasting_glucose_level": 95.0,
    "cholesterol_level": 200.0, "physical_activity_frequency": 0.3,
    "family_history_of_diabetes": "Present",
}])
print(results[0]["flagged"], round(results[0]["risk_score"], 3))   # True 0.661
```

| Member | Description |
|---|---|
| `RiskPredictor.load(variant, config=None, models_dir=..., reports_dir=...)` | Builds a predictor from saved artifacts. Raises `ModelNotAvailableError` if they're missing |
| `predict(records)` | Returns `risk_score`, `flagged` and `warnings` for each record. Raises `InvalidInputError` if a record can't be scored |
| `required_columns` | Input columns this variant needs |
| `threshold`, `variant`, `model_name` | The loaded model's details |

### Data: `src.data_preparation.DataPreparation(config)`

| Method | Description |
|---|---|
| `generate_validation_report(df)` | Data-quality statistics on raw data |
| `clean_data(df)` | Cleans raw data. Does not engineer features |
| `split_data(df)` | Returns `X_train, X_test, y_train, y_test`, stratified |
| `numeric_features_for(include_leakage_candidates, linear_model, engineered)` | Numeric features for a variant |
| `create_preprocessor(...)` | Unfitted scaler and encoder |
| `create_feature_pipeline(...)` | Unfitted `engineer` → `preprocessor` pipeline |
| `lr_engineered_features()` | Engineered features used by Logistic Regression |
| `feature_variants()` | Names the four feature variants |

### Features: `src.feature_engineering`

| Name | Description |
|---|---|
| `ClinicalFeatureEngineer(features=None)` | scikit-learn transformer that adds engineered features to a DataFrame |
| `FEATURE_SPECS` | Registry of engineered features |
| `features_derived_from(columns, features=None)` | Engineered features built from the given columns |
| `l1_selection(pipeline, X, y)` | L1-based feature selection |
| `permutation_ranking(pipeline, X, y, cv, random_state, ...)` | Held-out permutation importance |
| `compare_feature_sets(pipelines, X, y, cv, random_state, ...)` | Paired CV comparison with a baseline |
| `passes_retention_rule(result, n_folds, min_gain)` | Whether a feature set's gain is large and consistent |

### Models: `src.models.ModelTraining(config, data_preparation)`

`AdvancedModelTraining` in `src.advanced_models` has the same interface for Random Forest and Gradient Boosting.

| Method | Description |
|---|---|
| `build_pipeline(model_name, glucose_variant)` | Unfitted pipeline |
| `train_baselines(X_train, y_train)` | CV of the dummy and untuned Logistic Regression |
| `tune_models(X_train, y_train)` | Grid search per model and variant |
| `select_best(results, glucose_variant)` | Highest mean CV score within a variant |
| `tune_threshold(pipeline, X_train, y_train)` | Threshold from out-of-fold predictions |
| `evaluate_on_test(pipeline, X_test, y_test, model_name, threshold=0.5)` | Test metrics |
| `explain_model(pipeline, X, y)` | Permutation importance per input column |
| `used_feature_names(pipeline)` | Feature names as the classifier sees them |

`compare_to_baseline(results, baseline_key, metric)` in `src.advanced_models` compares models fold by fold.

### Optimization: `src.model_optimization`

| Name | Description |
|---|---|
| `ModelOptimizer(config, data_preparation)` | Tunes, diagnoses and selects, on training data only |
| `ModelOptimizer.optimize(reference_models, X_train, y_train)` | Runs everything; returns tuned pipelines and the report |
| `ModelOptimizer.tune`, `evaluate_fit`, `nested_cv`, `learning_curves` | The individual steps |
| `make_one_se_refit(n_folds)` | `refit` rule: smallest gap within 1 SE of the best |
| `diagnose_fit(train_mean, val_mean, val_se, best_val_mean, max_gap)` | `overfit`, `underfit` or `good_fit` |
| `select_final(scores, simplest_first)` | Simplest model within 1 SE of the best |

### Evaluation: `src.evaluation`

| Function | Description |
|---|---|
| `metrics_from_scores(y_true, y_score, threshold=0.5)` | All classification metrics |
| `compute_metrics(model, X, y, threshold=0.5)` | The same, for a fitted model |
| `threshold_for_target_recall(y_true, y_score, target_recall)` | Highest threshold that keeps the target recall |
| `bootstrap_ci(y_true, y_score, threshold, n_boot=1000, ...)` | Stratified bootstrap confidence intervals |
| `predict_with_threshold(model, X, threshold=0.5)` | Labels at a threshold |
| `plot_*` | ROC, PR, calibration, confusion matrix, comparison and learning-curve figures |
