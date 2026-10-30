"""
REST API for the diabetes risk models.

Run from the project root, after `python main.py` has created the models:

    uvicorn src.api:app --host 0.0.0.0 --port 8000

See documentation/api_reference.md. Interactive docs are served at /docs.
"""

import logging
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.predict import VARIANTS, InvalidInputError, ModelNotAvailableError, RiskPredictor

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1000

NOTICE = (
    "Technical demonstration only, not for clinical use. In this synthetic dataset the label runs against "
    "clinical knowledge for 6 of 7 features, so scores are not real diabetes risk."
)

app = FastAPI(title="Diabetes Risk Detection API", version="1.0.0", description=NOTICE)


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """
    Returns 422 without echoing the submitted values: they may be health data,
    and a NaN or infinity in them can't be written as JSON.
    """
    details = [{key: item[key] for key in ('type', 'loc', 'msg')} for item in error.errors()]
    return JSONResponse(status_code=422, content={'detail': details})


class PatientRecord(BaseModel):
    """One person's measurements, using the column names of the training data."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    age: float = Field(description="Age in years. Must be at least 18.", examples=[50])
    bmi: float = Field(description="Body mass index, kg/m².", examples=[28.0])
    blood_pressure: float = Field(description="Systolic blood pressure, mmHg.", examples=[125.0])
    fasting_glucose_level: Optional[float] = Field(
        default=None, description="Fasting glucose, mg/dL. Required for `with_glucose`.", examples=[95.0],
    )
    cholesterol_level: float = Field(description="Total cholesterol, mg/dL.", examples=[200.0])
    physical_activity_frequency: float = Field(description="Activity frequency, as a proportion.", examples=[0.3])
    family_history_of_diabetes: Literal['Absent', 'Present'] = Field(description="Family history of diabetes.")


class PredictionRequest(BaseModel):
    """A batch of records and the model variant to score them with."""

    model_config = ConfigDict(extra='forbid')

    variant: Literal['with_glucose', 'without_glucose'] = 'with_glucose'
    records: List[PatientRecord] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    """The result for one record, in request order."""

    risk_score: float = Field(description="Model score from 0 to 1. Ranks people; it is not a calibrated risk.")
    flagged: bool = Field(description="True if `risk_score` is at or above `threshold`.")
    warnings: List[str] = Field(description="Values outside the plausible ranges. The record is still scored.")


class PredictionResponse(BaseModel):
    """Predictions and the model that produced them."""

    variant: str
    model: str
    threshold: float
    predictions: List[Prediction]
    notice: str


class HealthResponse(BaseModel):
    """Service status and which variants can be scored."""

    status: Literal['ok', 'degraded']
    models: Dict[str, bool]


@lru_cache(maxsize=len(VARIANTS))
def _load_predictor(variant: str) -> RiskPredictor:
    """Loads a predictor once. A failed load is not cached, so new models are picked up without a restart."""
    return RiskPredictor.load(variant)


def get_predictors() -> Dict[str, RiskPredictor]:
    """FastAPI dependency: every predictor that can be loaded, keyed by variant."""
    predictors = {}
    for variant in VARIANTS:
        try:
            predictors[variant] = _load_predictor(variant)
        except ModelNotAvailableError as error:
            logger.warning("Model not available: %s", error)
    return predictors


Predictors = Annotated[Dict[str, RiskPredictor], Depends(get_predictors)]


@app.get('/health', response_model=HealthResponse, responses={503: {'model': HealthResponse}})
def health(predictors: Predictors) -> Any:
    """Reports whether every model is loaded. Returns 503 if any is missing, so orchestrators can see it."""
    models = {variant: variant in predictors for variant in VARIANTS}
    healthy = all(models.values())
    body = {'status': 'ok' if healthy else 'degraded', 'models': models}
    return body if healthy else JSONResponse(status_code=503, content=body)


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest, predictors: Predictors) -> Dict[str, Any]:
    """Scores up to 1,000 records with the chosen variant. `flagged` applies that variant's recall-tuned threshold."""
    predictor = predictors.get(request.variant)
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model '{request.variant}' is not available.")

    records = [record.model_dump() for record in request.records]
    try:
        predictions = predictor.predict(records)
    except InvalidInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        'variant': predictor.variant,
        'model': predictor.model_name,
        'threshold': predictor.threshold,
        'predictions': predictions,
        'notice': NOTICE,
    }
