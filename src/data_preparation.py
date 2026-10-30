"""Data validation, cleaning, splitting and preprocessing for diabetes risk detection."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FEATURE_SPECS, ClinicalFeatureEngineer, features_derived_from

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Class for data validation, cleaning, splitting and preprocessing of the
    diabetes dataset, following the decisions made in eda.ipynb (Part 3,
    §15-21). Feature engineering itself lives in `src.feature_engineering`;
    this class decides which engineered features each variant uses.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for data preparation.
    """

    REQUIRED_RAW_COLUMNS = (
        'age', 'bmi', 'blood_pressure', 'fasting_glucose_level', 'cholesterol_level',
        'physical_activity_frequency', 'family_history_of_diabetes',
    )

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for data preparation.
        """
        self.config = config

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes data quality statistics on the raw DataFrame before cleaning,
        so every run is auditable.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
        Returns:
            Dict[str, Any]: Row/column counts, missing-value counts, duplicate
                count, unexpected categorical levels, values outside the
                plausible ranges, rows below the minimum age, and the target
                class distribution.
        """
        report = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "missing_values_per_column": {
                col: int(count) for col, count in df.isna().sum().items() if count > 0
            },
            "n_duplicate_rows": int(df.duplicated().sum()),
            "unexpected_categorical_levels": {},
            "n_outside_plausible_range": {},
        }

        for col, levels in self.config['categorical_levels'].items():
            if col not in df.columns:
                continue
            values = df[col].dropna().astype(str).str.strip()
            unexpected = values[~values.isin(levels)].value_counts()
            if not unexpected.empty:
                report["unexpected_categorical_levels"][col] = {
                    str(value): int(count) for value, count in unexpected.items()
                }

        for col, (low, high) in self.config['plausible_ranges'].items():
            if col in df.columns:
                report["n_outside_plausible_range"][col] = {
                    "below": int((df[col] < low).sum()),
                    "above": int((df[col] > high).sum()),
                }

        if 'age' in df.columns:
            report["n_below_min_age"] = int((df['age'] < self.config['min_age']).sum())

        target = self.config['target_column']
        if target in df.columns:
            report["target_distribution"] = {
                str(label): int(count) for label, count in df[target].value_counts(dropna=False).items()
            }

        logger.info("Data validation report: %s", report)
        return report

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw data. Engineered features are added later, inside the
        model pipeline (`create_feature_pipeline`).

        Steps (see eda.ipynb for the evidence behind each one):
            1. Drop exact duplicate rows and rows with a missing target.
            2. Strip whitespace from text columns and check every value is an
               expected level (EDA §3.3).
            3. Drop rows below `min_age`: out of scope for adult screening (EDA §15).

        Every step uses only values from the same row, so it is safe to run
        before the train/test split. Statistical outliers are kept on purpose:
        35% of them are Positive (EDA §5).

        Args:
            df (pd.DataFrame): Input DataFrame containing raw data.
        Returns:
            pd.DataFrame: The cleaned DataFrame (a new object; the input is not modified).
        Raises:
            ValueError: If the DataFrame is empty, is missing a required column,
                has a missing feature value, or contains an unexpected category level.
        """
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")

        target = self.config['target_column']
        required_columns = [target, *self.REQUIRED_RAW_COLUMNS]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required raw column(s): {missing_columns}.")

        logger.info("Starting data cleaning process. Rows before cleaning: %d", len(df))

        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        logger.info("Removed %d duplicate row(s).", n_duplicates)

        n_missing_target = int(df[target].isna().sum())
        df = df.dropna(subset=[target])
        logger.info("Removed %d row(s) with a missing target.", n_missing_target)

        # The EDA found no missing feature values, so the pipeline has no imputer;
        # fail loudly rather than pass NaNs to the model.
        missing_features = {
            col: int(count) for col, count in df[list(self.REQUIRED_RAW_COLUMNS)].isna().sum().items() if count
        }
        if missing_features:
            raise ValueError(f"Missing feature value(s) found: {missing_features}.")

        df = self._standardise_categorical_levels(df)
        df = self._drop_below_min_age(df)

        df = df.reset_index(drop=True)
        logger.info("Data cleaning process completed. Rows after cleaning: %d", len(df))
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Encodes the target as 1 (positive class) / 0 (other class) and splits
        the data into training and test sets, stratified on the target so both
        keep the same class ratio. There is no separate validation set: with
        ~150 positives, model selection uses stratified k-fold CV on the
        training set instead (EDA §16).

        Args:
            df (pd.DataFrame): Input DataFrame containing cleaned data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
        Raises:
            ValueError: If the positive class does not split the target into two classes.
        """
        logger.info("Starting data splitting process.")
        target = self.config['target_column']
        X = df.drop(columns=[target])
        y = (df[target].astype(str) == str(self.config['positive_class'])).astype(int)
        if y.nunique() < 2:
            raise ValueError(
                f"Positive class '{self.config['positive_class']}' must match some, but not all, "
                f"values of '{target}'."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=self.config['random_state'], stratify=y,
        )
        for split_name, y_split in (('train', y_train), ('test', y_test)):
            logger.info(
                "%s set: %d rows, %d positive (share %.3f)",
                split_name, len(y_split), int(y_split.sum()), y_split.mean(),
            )
        logger.info("Data splitting process completed.")
        return X_train, X_test, y_train, y_test

    def lr_engineered_features(self) -> List[str]:
        """
        Lists the engineered features the Logistic Regression variants use,
        chosen from the CV comparison in main.py (EDA §17).

        Returns:
            List[str]: Engineered feature names.
        """
        return list(self.config['lr_engineered_features'])

    def numeric_features_for(
        self,
        include_leakage_candidates: bool = True,
        linear_model: bool = False,
        engineered: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Lists the numeric features for one feature variant.

        Args:
            include_leakage_candidates (bool): Keep `leakage_candidate_features`
                (fasting glucose) and features derived from them. False gives the
                "without glucose" variant (README §5.7).
            linear_model (bool): Use the Logistic Regression engineered features
                (`lr_engineered_features`); ignored when `engineered` is given.
            engineered (Optional[Sequence[str]]): Engineered features to use instead,
                e.g. one candidate at a time when comparing feature sets. A feature
                whose `FeatureSpec.replaces` is set takes the place of that raw
                feature; any other is appended.
        Returns:
            List[str]: Numeric feature names, in config order.
        """
        if engineered is None:
            engineered = self.lr_engineered_features() if linear_model else []

        features = list(self.config['numeric_features'])
        for name in engineered:
            replaces = FEATURE_SPECS[name].replaces
            if replaces in features:
                features = [name if f == replaces else f for f in features]
            elif name not in features:
                features.append(name)

        if not include_leakage_candidates:
            leakage = self.config['leakage_candidate_features']
            excluded = {*leakage, *features_derived_from(leakage)}
            features = [f for f in features if f not in excluded]
        return features

    def create_preprocessor(
        self,
        include_leakage_candidates: bool = True,
        linear_model: bool = False,
        engineered: Optional[Sequence[str]] = None,
    ) -> ColumnTransformer:
        """
        Builds a new, unfitted preprocessor: `StandardScaler` for numeric
        features and a single 0/1 column per binary feature. Called once per
        model pipeline, so each pipeline owns an independent instance and the
        scaler is fitted on training data (or each CV training fold) only.
        Columns not selected are dropped, so excluded features never reach the model.
        Engineered features must already be present (see `create_feature_pipeline`).

        Args:
            include_leakage_candidates (bool): See `numeric_features_for`.
            linear_model (bool): See `numeric_features_for`.
            engineered (Optional[Sequence[str]]): See `numeric_features_for`.
        Returns:
            ColumnTransformer: A preprocessor for data transformation.
        """
        numeric_features = self.numeric_features_for(include_leakage_candidates, linear_model, engineered)
        binary_features = self.config['binary_features']

        transformers = []
        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))
        if binary_features:
            # drop='if_binary' keeps one column per two-level feature (e.g. family_history_of_diabetes_Present).
            # Unknown levels raise, but clean_data has already rejected them.
            transformers.append(('bin', OneHotEncoder(drop='if_binary'), binary_features))

        return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)

    def create_feature_pipeline(
        self,
        include_leakage_candidates: bool = True,
        linear_model: bool = False,
        engineered: Optional[Sequence[str]] = None,
    ) -> Pipeline:
        """
        Builds a new, unfitted feature pipeline for raw (cleaned) data: a
        `ClinicalFeatureEngineer` that adds only the engineered features this
        variant uses, followed by `create_preprocessor`. Append a model to get
        a full pipeline that cross-validation refits on every fold.

        Args:
            include_leakage_candidates (bool): See `numeric_features_for`.
            linear_model (bool): See `numeric_features_for`.
            engineered (Optional[Sequence[str]]): See `numeric_features_for`.
        Returns:
            Pipeline: Steps 'engineer' and 'preprocess'.
        """
        numeric_features = self.numeric_features_for(include_leakage_candidates, linear_model, engineered)
        return Pipeline([
            ('engineer', ClinicalFeatureEngineer(features=[f for f in numeric_features if f in FEATURE_SPECS])),
            ('preprocess', self.create_preprocessor(include_leakage_candidates, linear_model, engineered)),
        ])

    @staticmethod
    def feature_variants() -> Dict[str, Dict[str, bool]]:
        """
        Names the four feature variants: with / without the leakage candidate
        features (fasting glucose), each for tree models and for Logistic Regression.

        Returns:
            Dict[str, Dict[str, bool]]: Variant name -> keyword arguments for `create_feature_pipeline`.
        """
        return {
            'with_glucose__trees': {'include_leakage_candidates': True, 'linear_model': False},
            'with_glucose__logistic_regression': {'include_leakage_candidates': True, 'linear_model': True},
            'without_glucose__trees': {'include_leakage_candidates': False, 'linear_model': False},
            'without_glucose__logistic_regression': {'include_leakage_candidates': False, 'linear_model': True},
        }

    def _standardise_categorical_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips whitespace from the configured text columns and checks every
        value is an expected level.

        Args:
            df (pd.DataFrame): DataFrame after duplicate / missing-target removal.
        Returns:
            pd.DataFrame: The DataFrame with stripped text columns.
        Raises:
            ValueError: If a column contains a value outside `categorical_levels`.
        """
        for col, levels in self.config['categorical_levels'].items():
            if col not in df.columns:
                continue
            stripped = df[col].astype(str).str.strip()
            n_changed = int((stripped != df[col].astype(str)).sum())
            unexpected = sorted(set(stripped) - set(levels))
            if unexpected:
                raise ValueError(f"Unexpected value(s) in '{col}': {unexpected}. Expected one of {levels}.")
            df[col] = stripped
            if n_changed:
                logger.info("Stripped whitespace from %d '%s' value(s).", n_changed, col)
        return df

    def _drop_below_min_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows below `min_age` (EDA §15). Ages above 100 and extreme BMIs
        are kept: rare, but still in scope.

        Args:
            df (pd.DataFrame): DataFrame after level checks.
        Returns:
            pd.DataFrame: The DataFrame without under-age rows.
        """
        below = df['age'] < self.config['min_age']
        dropped_classes = df.loc[below, self.config['target_column']].value_counts().to_dict()
        logger.info(
            "Dropped %d row(s) with age < %s (target classes: %s).",
            int(below.sum()), self.config['min_age'], dropped_classes,
        )
        return df.loc[~below]
