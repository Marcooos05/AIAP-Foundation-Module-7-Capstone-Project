# AIAP Module 7 Capstone Project

**Early detection of diabetes risk: a binary classification problem on tabular clinical data**

## Dataset Overview

**File:** `data/synthetic_data_166.csv`, with 1,000 rows (one per individual) × 8 columns (7 features + 1 target). The filename indicates the data is **synthetic**, and its near-perfectly normal distributions support this (see [EDA.ipynb](EDA.ipynb)).

### Features

Units are not documented. The ones below are inferred from the value ranges and standard clinical conventions.

| Feature | Type | Assumed unit | Range (mean ± SD) | Description | Expected link to diabetes |
|---|---|---|---|---|---|
| `age` | Numeric (int) | years | 12 – 106 (54.6 ± 14.9) | Age of the individual | Risk rises with age; ADA screening starts at 35 |
| `bmi` | Numeric | kg/m² | 11.8 – 47.7 (30.0 ± 5.0) | Body mass index | Obesity (BMI ≥ 30) is the strongest modifiable risk factor |
| `blood_pressure` | Numeric | mmHg (likely systolic) | 87.7 – 181.0 (129.7 ± 14.8) | Blood pressure reading | Hypertension often co-occurs with type 2 diabetes |
| `fasting_glucose_level` | Numeric | mg/dL | 50.9 – 149.9 (104.6 ± 15.4) | Fasting plasma glucose | **Diagnostic criterion** (≥ 126 = diabetes), so this is a leakage risk (§5.7) |
| `cholesterol_level` | Numeric | mg/dL (likely total) | 90.3 – 330.9 (206.1 ± 35.1) | Blood cholesterol | Dyslipidaemia is linked to insulin resistance |
| `physical_activity_frequency` | Numeric | proportion (0–1) | 0.01 – 0.79 (0.28 ± 0.16) | How often the person exercises; exact definition unknown | Higher activity means lower risk |
| `family_history_of_diabetes` | Categorical (binary) | Absent / Present | 60% Absent, 40% Present | Diabetes in a first-degree relative | Strong non-modifiable risk factor |

### Target

| Column | Values | Distribution |
|---|---|---|
| `diabetes_diagnosis` | Negative / Positive | 850 Negative (85%) / 150 Positive (**15%**) |

### Initial observations (univariate EDA)

- **Data quality:** there are no missing values, blank strings, duplicate rows or inconsistent category labels, so no structural cleaning is needed.
- **Implausible values:** 7 rows fall outside adult clinical ranges: ages 12, 16, 17, 101, 104 and 106, and a BMI of 11.8. In data preparation, the **3 minors (ages 12, 16, 17) are dropped** as outside the adult screening population. The ages above 100 and the BMI of 11.8 are kept (see Data preparation below).
- **Distributions:** five numeric features are approximately normal (|skew| < 0.15, Shapiro–Wilk p > 0.05). `physical_activity_frequency` is the exception: it is right-skewed (skew 0.57).
- **Outliers:** 34 rows (3.4%) contain at least one IQR outlier, and 15 are confirmed by the z-score rule. Most are clinically plausible extremes. 35% of outlier rows are Positive (vs 15% overall), so they are kept.
- **Feature–target direction (preliminary):** apart from family history, the class means go **against** clinical expectations:

  | Feature | Negative (mean) | Positive (mean) | Expected |
  |---|---|---|---|
  | age | 56.3 | 44.9 | Positive older |
  | bmi | 30.7 | 25.8 | Positive higher |
  | blood_pressure | 131.4 | 120.4 | Positive higher |
  | fasting_glucose_level | 107.9 | 86.0 | Positive higher |
  | cholesterol_level | 207.9 | 195.5 | Positive higher |
  | physical_activity_frequency | 0.26 | 0.38 | Positive lower |
  | family history Present → % Positive | | 24.8% (vs 8.5% Absent) | ✓ as expected |

  The bivariate analysis below examines this further.

### Bivariate & correlation findings

Full analysis: [EDA.ipynb](EDA.ipynb), Part 2 (§8–14).

| Rank | Feature | AUC alone | Corr. with Positive | Effect size | Matches clinical? |
|---|---|---|---|---|---|
| 1 | fasting_glucose_level | 0.90 | −0.51 | d = −1.65 (large) | ✗ |
| 2 | bmi | 0.78 | −0.34 | d = −1.02 (large) | ✗ |
| 3 | blood_pressure | 0.72 | −0.26 | d = −0.77 (medium) | ✗ |
| 4 | age | 0.71 | −0.27 | d = −0.80 (medium) | ✗ |
| 5 | physical_activity_frequency | 0.71 | +0.26 | d = +0.76 (medium) | ✗ |
| 6 | family_history | 0.65 | +0.22 | OR = 3.54 (95% CI 2.46–5.10) | ✓ |
| 7 | cholesterol_level | 0.61 | −0.13 | d = −0.36 (small) | ✗ |

*AUC alone* is the ROC-AUC of each feature used as the only predictor, taken in whichever direction works better. Correlation is point-biserial (Pearson against Positive = 1).

- **Strong signal:** every feature differs significantly between the classes (p < 1e-4), and five have a single-feature AUC above 0.7. Even a simple model should perform well.
- **No multicollinearity:** no feature pair has |r| > 0.7 and every VIF is below 2. Pearson ≈ Spearman, so the relationships are monotonic and near-linear. All 7 features can be kept.
- **Glucose dominates, but it isn't classic leakage.** One cut-off (≤ 94 mg/dL) gives sensitivity 0.81 and specificity 0.85, but the classes still overlap. The label was clearly *not* derived from the ADA ≥ 126 rule, since none of those 86 rows are Positive.
- **Reversed risk gradients:** binned positive rates fall steadily as glucose, BMI, age and blood pressure rise (e.g. BMI < 18.5 → 54% Positive, BMI ≥ 30 → 4%). A FINDRISC-style risk-factor count is perfectly inverted, falling from 50% Positive with 0 factors to 0% with all 6.
- **The anomaly lies in the label, not the features.** The features correlate with each other in clinically sensible ways (BMI with glucose, r = 0.54, and with BP, r = 0.51), but 6 of 7 relate to the label in the reverse direction. Simply swapping the labels doesn't explain it: family history would become protective and prevalence would jump to 85%. A synthetic generator with non-clinical rules is the most likely cause.
- **Decision:** model the label **as given**, report the anomaly as a limitation (TRIPOD+AI requires the outcome definition), and **raise it with the data provider or instructor**.

### Data preparation

Explored and justified in [EDA.ipynb](EDA.ipynb), Part 3 (§15–21), to be ported into the `src/` pipeline.

| Step | Decision | Leakage safety |
|---|---|---|
| **Cleaning** | Drop the 3 rows with age < 18 (2 Positive, 1 Negative), giving **997 rows, 148 Positive (14.8%)**. Keep ages > 100, BMI 11.8 and all statistical outliers (35% of outlier rows are Positive) | Fixed rule, applied before the split |
| **Feature engineering** | `sqrt_physical_activity` (skew 0.57 → −0.06) and `family_history_x_bmi`, for Logistic Regression only. Kept only if CV shows a gain: on the training set the interaction is weak alone (AUC 0.59). Clinical bins are for interpretability only | Row-wise, applied before the split |
| **Splitting** | Stratified 80/20 train/test (`random_state=42`): train 797 (118 Positive), test 200 (30 Positive). No separate validation set; stratified 5-fold CV on train (23–24 positives per fold). The test set is used once | |
| **Encoding** | `family_history_of_diabetes` → `OneHotEncoder(drop="if_binary")` (one 0/1 column). Target: Positive = 1 | Inside the pipeline |
| **Scaling** | `StandardScaler` on numeric features, since they are near-normal with few outliers (so `RobustScaler` isn't needed) | Fitted on each training fold only, inside the pipeline |
| **Feature sets** | `with_glucose` (7 features for trees, 8 for LR) and `without_glucose` (6 / 7) | |

## Literature Review

> **Status:** first draft, written after the literature review and updated after the univariate and bivariate EDA ([EDA.ipynb](EDA.ipynb)). The open question is the label definition (see Dataset Overview).

### 1. Problem statement and rationale

Diabetes often goes undetected for years, and by the time it's diagnosed, complications (cardiovascular disease, kidney disease, retinopathy, neuropathy) may already have started. Spotting people at high risk early enables lifestyle changes and earlier clinical follow-up, which can delay or prevent these complications.

This project builds a model that flags individuals at risk of diabetes from routinely collected health indicators.

| Item | Value |
|---|---|
| Task | Binary classification (`diabetes_diagnosis`: Positive / Negative) |
| Data | 1,000 rows, tabular, no missing values or duplicates |
| Features (7) | `age`, `bmi`, `blood_pressure`, `fasting_glucose_level`, `cholesterol_level`, `physical_activity_frequency`, `family_history_of_diabetes` |
| Class balance | 850 Negative / 150 Positive (**15% positive**, imbalanced) |
| Intended use | Screening: flag people for follow-up testing, not diagnose them |

### 2. Review methodology

- **Sources:** PubMed, Google Scholar, arXiv / NeurIPS proceedings, plus clinical guidelines (ADA *Standards of Care in Diabetes*, WHO diagnostic criteria).
- **Search terms:** "diabetes prediction machine learning", "clinical prediction model logistic regression vs machine learning", "class imbalance risk prediction", "SMOTE calibration", "precision-recall imbalanced", "TRIPOD+AI", "sample size clinical prediction model".
- **Inclusion focus:** (1) clinical diabetes risk scores; (2) methodology papers on model choice, class imbalance, evaluation and reporting for clinical prediction models on tabular data. Papers reporting only accuracy on imbalanced data, or with unclear validation, were read critically (see §6).

### 3. Clinical background

- **Diagnostic thresholds (ADA / WHO), fasting plasma glucose:** < 100 mg/dL is normal, 100–125 mg/dL is prediabetes, and **≥ 126 mg/dL is diabetes**. Fasting glucose is therefore not just a risk factor but part of the **definition** of the outcome (see §5.7).
- **Established risk scores:** the Finnish Diabetes Risk Score (FINDRISC; Lindström & Tuomilehto, 2003) predicts 10-year type 2 diabetes risk from age, BMI, waist circumference, antihypertensive medication, history of high blood glucose, physical activity and diet. It needs no laboratory tests, and its predictors overlap heavily with our features. Scores like this are logistic-regression-based, which supports LR as a strong, interpretable baseline.
- **Known risk factors** for type 2 diabetes include older age, higher BMI, hypertension, dyslipidaemia, physical inactivity and family history. This gives the **expected direction** of each feature, which we check against our data in EDA.

### 4. Key findings from the literature

| # | Source | Finding | Implication for this project |
|---|---|---|---|
| 1 | Christodoulou et al., 2019 (*J Clin Epidemiol*) | Systematic review of 71 studies: **no performance benefit of ML over logistic regression** for clinical prediction. 68% of studies had potential validation bias, and 79% didn't address calibration. | Logistic regression as the baseline. Any ML model must beat it under fair validation. Report calibration. |
| 2 | Grinsztajn, Oyallon & Varoquaux, 2022 (*NeurIPS*) | On tabular data, **tree-based ensembles (RF, XGBoost) outperform deep learning**, especially on small or medium datasets. | Use tree-based models (DT, RF, gradient boosting). No deep learning for 1,000 rows. |
| 3 | van den Goorbergh et al., 2022 (*JAMIA*) | Random under-/oversampling and **SMOTE produced poorly calibrated models** (minority-class risk overestimated) and **did not improve AUC**. Shifting the decision threshold gave the same sensitivity/specificity balance. | **No resampling and no SMOTE.** Handle imbalance with cost-sensitive learning and threshold tuning. |
| 4 | Carriero et al., 2024 (arXiv preprint) | Extends #3 to ML models: models without imbalance correction had **equal or better calibration**. | Confirms #3 also applies to our tree-based models. |
| 5 | Chawla et al., 2002 (*JAIR*) | Introduced SMOTE (synthetic minority oversampling), the most widely used imbalance technique. | Reviewed as the common practice we are deliberately **not** following, given #3–4. |
| 6 | Saito & Rehmsmeier, 2015 (*PLoS ONE*) | On imbalanced data, the **precision-recall plot is more informative than the ROC plot**. ROC can look optimistic. | Report **PR-AUC** alongside ROC-AUC. |
| 7 | Van Calster et al., 2019 (*BMC Medicine*) | **Calibration is the "Achilles heel"** of predictive analytics. A miscalibrated model can mislead clinical decisions even when its AUC is high. | Report the Brier score and a calibration curve, not just discrimination. |
| 8 | Riley et al., 2020 (*BMJ*) | Many prediction models are built on datasets that are too small. They recommend a formal sample-size calculation (`pmsampsize`) rather than rules of thumb. | We have about 150 events and 7 predictors (~21 events per variable). Keep models simple, avoid too many engineered features, and use regularisation. |
| 9 | Lundberg & Lee, 2017 (*NeurIPS*) | SHAP: a unified, model-agnostic way to attribute predictions to features. | Use SHAP (or LR coefficients) to explain predictions to clinicians. |
| 10 | Collins et al., 2024 (*BMJ*), **TRIPOD+AI** | Updated reporting guideline for prediction models built with regression **or** ML. It replaces TRIPOD 2015 with a 27-item checklist. | Use TRIPOD+AI as the reporting checklist for this project. |

### 5. Project decisions (initial notes)

#### 5.1 Cost of errors: false negatives matter most

- A **false negative** (a person at risk is told they're fine) means a missed chance for early intervention and a higher risk of complications. **This is the costly error.**
- A **false positive** (a healthy person is flagged) leads to a follow-up test such as fasting glucose or HbA1c, which is cheap and low-risk. **False positives are acceptable.**
- **Goal:** minimise false negatives, i.e. maximise recall (sensitivity), while keeping precision at a level that doesn't make screening pointless.

#### 5.2 Handling class imbalance: no resampling

- **We will not use oversampling, undersampling or SMOTE.** The evidence (#3, #4) shows they distort calibration without improving discrimination.
- Instead we use **cost-sensitive learning**:
  - `class_weight="balanced"` (or a custom FN:FP cost ratio) in LR, DT and RF, and the equivalent (`scale_pos_weight` / sample weights) in gradient boosting.
  - **Decision-threshold tuning:** instead of the default 0.5, pick the threshold inside cross-validation that reaches the target recall (or maximises F2).

#### 5.3 Evaluation metrics

| Role | Metric | Why |
|---|---|---|
| **Primary** | **Recall / sensitivity** | Directly measures missed cases (FN), which are the costly error |
| Secondary | PR-AUC | Threshold-free and informative under imbalance (#6) |
| Secondary | F2 score | Combines precision and recall, weighting recall more heavily |
| Secondary | Precision, specificity, confusion matrix | Shows the false-positive burden at the chosen threshold |
| Secondary | ROC-AUC | Standard discrimination measure, for comparison with the literature |
| Secondary | Brier score, calibration curve | Checks that predicted risks can be trusted (#7) |
| **Not used** | Accuracy | Predicting "Negative" for everyone already scores 85% |

#### 5.4 Validation strategy

- **Stratified** train/test split (e.g. 80/20), so both sets keep about 15% positives. The test set is used **once**, at the end.
- **Stratified k-fold cross-validation** (e.g. 5-fold, optionally repeated) on the training set for model selection, hyperparameter tuning and threshold tuning.
- All preprocessing (scaling, encoding, any outlier capping) goes **inside a scikit-learn `Pipeline`**, so it's fitted on training folds only and doesn't leak information from validation folds.

#### 5.5 Candidate models

| Order | Model | Role |
|---|---|---|
| 1 | **Logistic Regression** (L2-regularised) | **Baseline.** Interpretable, well calibrated, the clinical standard (#1) |
| 2 | Decision Tree | Simple non-linear model and interpretable rules. Prone to overfitting |
| 3 | Random Forest | Robust tree ensemble, strong on tabular data (#2) |
| 4 | Gradient Boosting (e.g. scikit-learn `HistGradientBoostingClassifier` / XGBoost) | Usually the strongest tabular learner (#2). Needs careful tuning on small data |

Every model is compared against the LR baseline using the same CV folds and metrics.

#### 5.6 Reporting standard

We follow the **TRIPOD+AI** checklist (#10): data source, outcome definition, predictors, sample size, handling of missing data and outliers, model development, performance (discrimination and calibration), and limitations.

#### 5.7 Fasting glucose: potential data leakage

- Fasting glucose ≥ 126 mg/dL is one of the criteria used to **diagnose** diabetes (§3). If the label was produced from glucose, a model using glucose would just learn the diagnostic rule. That is **target leakage**, and it doesn't help *early* detection.
- **Plan:**
  1. Train every model **with** and **without** `fasting_glucose_level` and compare.
  2. In the bivariate EDA, check how strongly glucose separates the classes. A near-deterministic relationship (e.g. almost every case above a cut-off belongs to one class) counts as **leakage**, and glucose is then **excluded** from the final model.
- **Bivariate EDA result:**
  - Glucose is the dominant predictor (AUC 0.90 alone) but is **not near-deterministic**. At the best single cut-off (≤ 94 mg/dL), sensitivity is 0.81 and specificity 0.85, so the classes still overlap.
  - The label was **not** derived from the diagnostic rule: none of the 86 rows with glucose ≥ 126 are Positive. Its relationship to the label is actually *reversed* (Positive mean ≈ 86 vs Negative ≈ 108 mg/dL), as it is for 5 other features (see Dataset Overview).
  - **Verdict: glucose is not classified as leakage. Keep it,** but still train every model with and without it, to report how well the non-laboratory features alone perform (more relevant for early screening).
  - The verdict will be revisited if the data provider's label definition shows that glucose was used to construct the label.

#### 5.8 Feature engineering *(updated after bivariate EDA)*

Kept to a small number given the sample size (#8):

| Candidate | Decision | Reason |
|---|---|---|
| Raw continuous features | **Keep** | Monotonic, near-linear relationships (Pearson ≈ Spearman); low VIF |
| Binary encoding of `family_history_of_diabetes` (Present = 1) | **Keep** | Strong, clinically consistent (OR 3.54) |
| Square-root transform of `physical_activity_frequency` | **Try** (LR only) | Right-skewed (skew 0.57); trees are unaffected |
| `family_history × bmi` interaction | **Try** (LR only) | The family-history effect varies across BMI bands, but cells are small |
| BMI category (WHO), BP category, age bands, glucose band | **Interpretability only** | Relationships are monotonic without thresholds, so there is little predictive gain over raw values |
| Clinical risk-factor count (FINDRISC-style) | **Drop** | Perfectly inverted on this data (50% → 0% Positive), so it would mislead |

### 6. Pitfalls identified in the literature

- **Data leakage through preprocessing:** scaling or resampling before the train/test split inflates performance. We fit everything inside the CV pipeline.
- **Reporting only accuracy** on imbalanced data. We use recall, PR-AUC and calibration instead.
- **No baseline comparison:** many ML papers don't compare against logistic regression (#1).
- **Ignoring calibration:** 79% of the studies reviewed in #1 didn't assess it.
- **Removing outliers blindly:** in our EDA, 35% of rows with outliers are Positive (vs 15% overall). Dropping them would remove the cases we most need to detect. We flag them instead and decide in preprocessing.
- **Target leakage** from features that define the outcome (glucose, §5.7).

### 7. References

1. Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol.* 2019;110:12–22.
2. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 (Datasets and Benchmarks).* arXiv:2207.08815.
3. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *J Am Med Inform Assoc.* 2022;29(9):1525–1534. doi:10.1093/jamia/ocac093
4. Carriero A, et al. The harms of class imbalance corrections for machine learning based prediction models: a simulation study. arXiv:2404.19494, 2024.
5. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic Minority Over-sampling Technique. *J Artif Intell Res.* 2002;16:321–357.
6. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE.* 2015;10(3):e0118432. doi:10.1371/journal.pone.0118432
7. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17:230. doi:10.1186/s12916-019-1466-7
8. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ.* 2020;368:m441. doi:10.1136/bmj.m441
9. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *NeurIPS 2017.*
10. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378.
11. Lindström J, Tuomilehto J. The Diabetes Risk Score: a practical tool to predict type 2 diabetes risk. *Diabetes Care.* 2003;26(3):725–731. doi:10.2337/diacare.26.3.725
12. American Diabetes Association Professional Practice Committee. Diagnosis and classification of diabetes: *Standards of Care in Diabetes* (current edition). *Diabetes Care.*

---

## Structure

```
AIAP_Mod7_Capstone-Project/
├── data/
│   └── synthetic_data_166.csv      # raw dataset (not committed, see .gitignore)
├── src/
│   ├── config.yaml                 # all pipeline settings (paths, split, features, levels)
│   ├── config_validation.py        # validates config.yaml before the pipeline runs
│   └── data_preparation.py         # DataPreparation: validation report, cleaning, feature engineering,
│                                   #   stratified split, preprocessor (scaling + encoding)
├── tests/                          # pytest suite (config validation, data preparation)
├── reports/
│   ├── figures/                    # EDA plots
│   ├── data_validation_report.json # written by main.py
│   └── data_preparation_summary.json
├── EDA.ipynb                       # exploratory analysis and data preparation rationale
├── main.py                         # pipeline entry point
├── requirements.txt / requirements-dev.txt
└── pytest.ini
```

## Setup

### Requirements

Python 3.10+ with the following packages (see `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| pandas | 3.0.3 | Data manipulation and analysis |
| numpy | 2.4.6 | Numerical computing |
| scikit-learn | 1.8.0 | Machine learning models and evaluation |
| scipy | 1.17.1 | Statistical tests (Shapiro–Wilk, chi-squared, etc.) |
| matplotlib | 3.10.9 | Visualisation |
| PyYAML | 6.0.3 | Configuration file parsing |
| joblib | 1.5.3 | Model serialisation |

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

Run the tests with `pip install -r requirements-dev.txt` and then `python -m pytest`.

## Pipeline (`python main.py`)

Run from the project root. All settings live in [src/config.yaml](src/config.yaml), which is validated before anything runs.

**Stage 1: data preparation** (implemented)

| Step | What happens | Output |
|---|---|---|
| 1. Load config | Read and validate `src/config.yaml` | |
| 2. Load data | Read the CSV and check the required columns | |
| 3. Validation report | Count missing values, duplicates, unexpected category levels, values outside plausible ranges, rows below `min_age`, and the target distribution, on the raw data | `reports/data_validation_report.json` |
| 4. Clean & engineer | Drop duplicates and missing targets, strip and check category levels, drop rows with age < 18, add `sqrt_physical_activity` and `family_history_x_bmi` | 1000 → 997 rows |
| 5. Split | Stratified 80/20 train/test (`random_state=42`) | train 797 (14.8% Positive), test 200 (15.0%) |
| 6. Preprocessor check | Fit `StandardScaler` + `OneHotEncoder(drop="if_binary")` on train for each feature variant, transform both sets, check for NaNs | `reports/data_preparation_summary.json` |

**Feature variants** (`DataPreparation.feature_variants()`):

| Variant | Features |
|---|---|
| `with_glucose__trees` | 6 raw numeric + family history (7) |
| `with_glucose__logistic_regression` | activity → sqrt, + family history × BMI (8) |
| `without_glucose__trees` | the same as trees, minus `fasting_glucose_level` (6) |
| `without_glucose__logistic_regression` | the same as LR, minus `fasting_glucose_level` (7) |

The preprocessor is always fitted on training data only. In the modelling stage it sits inside a scikit-learn `Pipeline`, so cross-validation refits it on every fold.

**Stage 2: model training & evaluation** (next): Logistic Regression baseline → Decision Tree → Random Forest → Gradient Boosting, with stratified 5-fold CV, `class_weight="balanced"` and threshold tuning for recall (see §5 of the literature review).

## Results and discussion
