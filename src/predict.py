"""
Inference for the saved diabetes risk models: load a selected pipeline with its
decision threshold, validate new records and score them.

The pipelines come from `python main.py`. The threshold is stored separately in
`reports/selected_models.json`, so `RiskPredictor` keeps the two together.
Not for clinical use: see documentation/model_explanation.md.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config_validation import validate_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'src' / 'config.yaml'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'

VARIANTS = ('with_glucose', 'without_glucose')


class ModelNotAvailableError(RuntimeError):
    """A saved pipeline or its threshold is missing, so `python main.py` has not been run."""


class InvalidInputError(ValueError):
    """A record cannot be scored."""


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads and validates the pipeline configuration.

    Args:
        path (Path): Path to config.yaml.
    Returns:
        Dict[str, Any]: The validated configuration.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


class RiskPredictor:
    """
    Scores records with one selected pipeline and its decision threshold.

    Attributes:
        pipeline (Any): The fitted scikit-learn pipeline.
        threshold (float): Score at or above which a record is flagged.
        variant (str): 'with_glucose' or 'without_glucose'.
        model_name (str): Name of the selected model.
        config (Dict[str, Any]): Configuration used for validation.
    """

    def __init__(self, pipeline: Any, threshold: float, variant: str, model_name: str, config: Dict[str, Any]):
        """
        Initializes the predictor. Use `load` to build one from saved artifacts.

        Args:
            pipeline (Any): A fitted pipeline exposing `predict_proba`.
            threshold (float): Decision threshold on the positive-class probability.
            variant (str): One of `VARIANTS`.
            model_name (str): Name of the selected model.
            config (Dict[str, Any]): Configuration used for validation.
        """
        self.pipeline = pipeline
        self.threshold = threshold
        self.variant = variant
        self.model_name = model_name
        self.config = config

    @classmethod
    def load(
        cls,
        variant: str,
        config: Optional[Dict[str, Any]] = None,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
    ) -> 'RiskPredictor':
        """
        Loads a saved pipeline and its threshold.

        Args:
            variant (str): One of `VARIANTS`.
            config (Optional[Dict[str, Any]]): Configuration; read from config.yaml if None.
            models_dir (Path): Folder holding `best_pipeline__{variant}.joblib`.
            reports_dir (Path): Folder holding `selected_models.json`.
        Returns:
            RiskPredictor: A predictor ready to score records.
        Raises:
            ValueError: If the variant is unknown.
            ModelNotAvailableError: If the pipeline or its threshold is missing.
        """
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'; expected one of {VARIANTS}.")

        model_path = Path(models_dir) / f'best_pipeline__{variant}.joblib'
        selection_path = Path(reports_dir) / 'selected_models.json'
        missing = [str(path) for path in (model_path, selection_path) if not path.exists()]
        if missing:
            raise ModelNotAvailableError(f"Missing {missing}. Run `python main.py` to create them.")

        selection = json.loads(selection_path.read_text())
        if variant not in selection:
            raise ModelNotAvailableError(f"'{variant}' is not in {selection_path}. Run `python main.py`.")

        return cls(
            pipeline=joblib.load(model_path),
            threshold=float(selection[variant]['decision_threshold']['threshold']),
            variant=variant,
            model_name=selection[variant]['model_name'],
            config=load_config() if config is None else config,
        )

    @property
    def required_columns(self) -> List[str]:
        """Raw input columns this variant needs. Leakage candidates are left out without glucose."""
        columns = [*self.config['numeric_features'], *self.config['binary_features']]
        if self.variant == 'without_glucose':
            columns = [c for c in columns if c not in self.config['leakage_candidate_features']]
        return columns

    def predict(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates and scores records.

        Args:
            records (Sequence[Mapping[str, Any]]): One mapping per person, with raw column names.
        Returns:
            List[Dict[str, Any]]: For each record, `risk_score`, `flagged` and
                `warnings` (values outside the plausible ranges, scored anyway).
        Raises:
            InvalidInputError: If there are no records, a required value is
                missing or not numeric, a category level is unknown, or a person
                is below `min_age`.
        """
        frame = self._validate(records)
        scores = self.pipeline.predict_proba(frame)[:, 1]
        return [
            {
                'risk_score': float(score),
                'flagged': bool(score >= self.threshold),
                'warnings': self._plausibility_warnings(row),
            }
            for score, (_, row) in zip(scores, frame.iterrows())
        ]

    def _validate(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Builds the model input, applying the same rules as `DataPreparation.clean_data`."""
        if not records:
            raise InvalidInputError("No records to score.")

        frame = pd.DataFrame([dict(record) for record in records])
        columns = self.required_columns
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            raise InvalidInputError(f"Missing required field(s): {absent}.")
        frame = frame[columns].copy()

        missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
        if missing:
            raise InvalidInputError(f"Missing value(s) per field: {missing}.")

        for column in self.config['numeric_features']:
            if column not in columns:
                continue
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
            if not np.isfinite(frame[column]).all():
                raise InvalidInputError(f"Field '{column}' must be a finite number.")

        for column in self.config['binary_features']:
            frame[column] = frame[column].astype(str).str.strip()
            unexpected = sorted(set(frame[column]) - set(self.config['categorical_levels'][column]))
            if unexpected:
                raise InvalidInputError(
                    f"Unexpected value(s) in '{column}': {unexpected}. "
                    f"Expected one of {self.config['categorical_levels'][column]}."
                )

        below_min_age = int((frame['age'] < self.config['min_age']).sum())
        if below_min_age:
            raise InvalidInputError(
                f"{below_min_age} record(s) below the minimum age of {self.config['min_age']}: out of scope."
            )
        return frame

    def _plausibility_warnings(self, row: pd.Series) -> List[str]:
        """Lists values outside the plausible ranges. As in training, they are flagged but kept."""
        return [
            f"{column}={row[column]} is outside the plausible range [{low}, {high}]"
            for column, (low, high) in self.config['plausible_ranges'].items()
            if column in row.index and not low <= row[column] <= high
        ]
