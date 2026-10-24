"""Entry point: run the diabetes risk detection pipeline (data preparation stage)."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.config_validation import validate_config
from src.data_preparation import DataPreparation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path('./src/config.yaml')
REPORTS_DIR = Path('reports')


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the pipeline YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def _load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads the raw dataset referenced by the config and checks required columns exist."""
    file_path = Path(config['file_path'])
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at '{file_path}'.")

    logger.info("Loading data from CSV file.")
    df = pd.read_csv(file_path)
    logger.info("Data loading completed: %d rows, %d columns.", *df.shape)

    required_columns = {config['target_column'], *DataPreparation.REQUIRED_RAW_COLUMNS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Data file is missing required column(s): {sorted(missing_columns)}.")

    return df


def _check_feature_variants(
    data_preparation: DataPreparation,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Fits each feature variant's preprocessor on the training set and transforms
    both sets, as a sanity check before modelling: no NaNs, expected columns.
    """
    variants = {}
    for name, kwargs in data_preparation.feature_variants().items():
        preprocessor = data_preparation.create_preprocessor(**kwargs).fit(X_train)
        Xt_train, Xt_test = preprocessor.transform(X_train), preprocessor.transform(X_test)
        if np.isnan(Xt_train).any() or np.isnan(Xt_test).any():
            raise ValueError(f"Feature variant '{name}' produced NaN values.")

        feature_names = list(preprocessor.get_feature_names_out())
        logger.info("Feature variant %s: %d features %s", name, len(feature_names), feature_names)
        variants[name] = {'n_features': len(feature_names), 'features': feature_names}
    return variants


def main():
    """
    Runs the data preparation stage of the pipeline:

    1. Load and validate the YAML config.
    2. Load the raw dataset and verify required columns are present.
    3. Generate a data validation report on the raw data.
    4. Clean the data (duplicates, missing targets, category levels, rows
       below the minimum age) and engineer features.
    5. Split into stratified train/test sets.
    6. Fit and check the preprocessor for every feature variant
       (with / without glucose, trees / Logistic Regression).
    7. Save the validation report and a data preparation summary to reports/.

    Model training and evaluation will be added as the next stage.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    config = _load_config(CONFIG_PATH)
    df = _load_data(config)

    data_preparation = DataPreparation(config)

    start = time.perf_counter()
    validation_report = data_preparation.generate_validation_report(df)
    _save_json(validation_report, REPORTS_DIR / 'data_validation_report.json')
    logger.info("Data validation completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    df_cleaned = data_preparation.clean_data(df)
    logger.info("Data cleaning completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    X_train, X_test, y_train, y_test = data_preparation.split_data(df_cleaned)
    logger.info("Data splitting completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    variants = _check_feature_variants(data_preparation, X_train, X_test)
    logger.info("Preprocessor checks completed in %.2fs.", time.perf_counter() - start)

    summary = {
        'n_rows_raw': int(len(df)),
        'n_rows_cleaned': int(len(df_cleaned)),
        'splits': {
            name: {'n_rows': int(len(y_split)), 'n_positive': int(y_split.sum()),
                   'positive_share': round(float(y_split.mean()), 4)}
            for name, y_split in (('train', y_train), ('test', y_test))
        },
        'feature_variants': variants,
    }
    _save_json(summary, REPORTS_DIR / 'data_preparation_summary.json')
    logger.info("Data preparation summary saved to %s", (REPORTS_DIR / 'data_preparation_summary.json').resolve())


if __name__ == "__main__":
    main()
