# Project report: early detection of diabetes risk

AIAP Foundation, Module 7 capstone.

## 1. Summary

This project built a classifier that flags people for follow-up diabetes screening from seven routine health indicators, on a synthetic dataset of 1,000 records. False negatives were treated as the costly error, so the work aimed for 90% recall.

The final model is **Logistic Regression**, in two variants. On the test set, both caught 26 of 30 positives (recall 0.87). ROC-AUC was 0.944 with fasting glucose and 0.849 without it. Random Forest and Gradient Boosting did not improve on this simple baseline, and neither did seven engineered features or wider hyperparameter tuning.

The most important finding is about the data. Six of seven features relate to the label in the opposite direction to clinical knowledge: people labelled Positive are younger, leaner and have lower glucose. The models learn this pattern accurately, so the results demonstrate a sound method, not a clinically useful tool. The label must be confirmed before any real use.

## 2. Problem and objectives

Diabetes often goes undetected for years, and complications may have begun by diagnosis. Identifying people at risk early allows lifestyle changes and earlier follow-up.

The errors aren't equal. A false negative means a missed chance to intervene, while a false positive leads to a cheap follow-up test. The objectives were therefore to:

1. Maximise recall, with a 90% target, while keeping precision usable
2. Assess whether laboratory glucose is needed
3. Compare a simple baseline against more complex models under fair validation
4. Package the result so that it's reproducible and can be served

## 3. Literature review

The full review is in the main [README](../README.md#literature-review). The findings that shaped this project:

| Finding | Source | Effect on this project |
|---|---|---|
| ML shows no clear benefit over logistic regression in clinical prediction | Christodoulou et al., 2019 | Logistic Regression as baseline |
| Tree ensembles lead on small tabular data | Grinsztajn et al., 2022 | Random Forest and Gradient Boosting as challengers; no deep learning |
| Resampling harms calibration without improving AUC | van den Goorbergh et al., 2022; Carriero et al., 2024 | No SMOTE; class weights and threshold tuning instead |
| PR curves are more informative under imbalance | Saito & Rehmsmeier, 2015 | PR-AUC as the selection metric |
| Calibration matters as much as discrimination | Van Calster et al., 2019 | Brier score and calibration curves reported |
| Datasets are often too small for their models | Riley et al., 2020 | Few features, strong regularisation |
| TRIPOD+AI sets reporting standards | Collins et al., 2024 | Used as a reporting guide |

## 4. Data

The dataset has 1,000 rows and 7 features, with a 15% positive class. It has no missing values, duplicates or inconsistent labels.

- **Implausible values:** 7 rows fall outside adult ranges. The 3 under-18s were dropped as out of scope, leaving **997 rows and 148 positives**. Ages above 100 and a BMI of 11.8 were kept.
- **Distributions:** five numeric features are almost perfectly normal, which suggests synthetic generation. Physical activity is right-skewed.
- **Outliers:** 34 rows have at least one IQR outlier. They were kept, because 35% are Positive against 15% overall.

## 5. Exploratory analysis

The full analysis is in [eda.ipynb](../eda.ipynb).

**Strong signal.** Fasting glucose alone reaches ROC-AUC 0.90, and BMI, blood pressure, age and activity each exceed 0.70.

**Clean structure.** Relationships are monotonic and near-linear (Pearson ≈ Spearman), with no multicollinearity (every VIF below 2). This pointed to a linear model doing well.

**Reversed direction.** Only family history behaves as expected (24.8% Positive with it, 8.5% without). For the other six features, the direction is reversed:

| Measure | Negative mean | Positive mean | Expected |
|---|---|---|---|
| Age | 56.3 | 44.9 | Positive older |
| BMI | 30.7 | 25.8 | Positive higher |
| Fasting glucose | 107.9 | 86.0 | Positive higher |
| Blood pressure | 131.4 | 120.4 | Positive higher |

None of the 86 people at or above the diagnostic glucose level of 126 mg/dL is labelled Positive, and a FINDRISC-style risk count is perfectly inverted, falling from 50% Positive with no risk factors to 0% with all six.

The features agree with each other in clinically sensible ways, which points to the label as the source. Swapping the labels doesn't fix it, because family history would become protective and prevalence would be 85%. **Decision:** model the label as given, report it, and raise it with the data provider.

**Glucose isn't classic leakage.** It's dominant but not deterministic, and the label wasn't derived from the ≥ 126 rule. Every model was still built with and without it, because the non-laboratory version better suits early screening.

## 6. Method

**Split:** stratified 80/20, giving 797 training rows (118 positive) and 200 test rows (30 positive).

**Preprocessing:** standard scaling and one-hot encoding inside a scikit-learn `Pipeline`, refitted on each training fold.

**Imbalance:** `class_weight="balanced"` and a tuned decision threshold, with no resampling.

**Metrics:** PR-AUC for selection, with recall as the operating goal. Accuracy wasn't used, because predicting Negative for everyone scores 0.85.

**Threshold:** the highest threshold keeping out-of-fold recall at or above 0.90 on the training set.

The work ran in stages, each tested against the one before:

1. **Feature engineering.** A reusable scikit-learn transformer and seven clinically motivated candidates, each tested alone, with L1 and permutation-based selection.
2. **Baseline model.** A dummy floor and Logistic Regression.
3. **Advanced models.** Random Forest (bagging) and Gradient Boosting (boosting), compared with the baseline fold by fold, with bootstrap confidence intervals.
4. **Optimization.** Repeated stratified CV, random search over wider spaces, a one-standard-error refit that favours smaller train−validation gaps, nested CV and learning curves.
5. **Productionisation.** Inference code, REST API, container and documentation.

A single Decision Tree was tried early and dropped as the weakest and least stable model.

## 7. Results

### Feature engineering

No candidate met the retention rule set in advance (a mean PR-AUC gain of at least 0.005, in most folds, and kept by L1). The closest was `sqrt_physical_activity`, at +0.003 without glucose. Fold-to-fold variation is about 0.04. Since relationships are already monotonic and near-linear, the models use the raw features.

### Baseline vs advanced models (fixed grids)

Mean PR-AUC, 5-fold CV on the training set:

| Model | With glucose | Without glucose |
|---|---|---|
| Dummy | 0.148 | 0.148 |
| Logistic Regression | 0.693 | 0.510 |
| Random Forest | 0.672 | 0.496 |
| Gradient Boosting | 0.693 | 0.480 |

Paired comparisons showed no reliable gain from either ensemble (p ≥ 0.28, indicative only). Without glucose, Logistic Regression led on test PR-AUC (0.612 vs 0.503 and 0.474).

### Optimization

Train−validation PR-AUC gap, and validation score, on the same repeated folds:

| Model | Variant | Gap: before → after | Validation: before → after | Diagnosis after |
|---|---|---|---|---|
| Logistic Regression | With glucose | −0.002 → −0.001 | 0.688 → 0.690 | Good fit |
| Gradient Boosting | With glucose | +0.097 → +0.027 | 0.690 → 0.656 | Good fit |
| Random Forest | With glucose | +0.260 → +0.023 | 0.665 → 0.676 | Good fit |
| Logistic Regression | Without glucose | −0.015 → −0.011 | 0.503 → 0.493 | Good fit |
| Gradient Boosting | Without glucose | +0.257 → +0.055 | 0.463 → 0.449 | Underfit |
| Random Forest | Without glucose | +0.191 → +0.048 | 0.491 → 0.475 | Good fit |

- **Overfitting was largely removed,** with all three overfitted cases now under the 0.10 limit.
- **Scores did not rise.** All changes were inside about one standard error. The refit rule also went too far for Gradient Boosting without glucose, which ended up underfitting.
- **Logistic Regression isn't underfitting.** Degree-2 terms scored lower than the plain model.
- **Tuning optimism is small.** Nested CV gave 0.679 with glucose and 0.495 without for Logistic Regression, close to the repeated-CV scores.

The final rule, set in advance, was the simplest model within one standard error of the best. Logistic Regression had the highest score in both variants, so it was selected on both counts.

### Final test performance

| Metric | With glucose | Without glucose | Dummy |
|---|---|---|---|
| Recall | 0.87 [0.73–0.97] | 0.87 [0.73–0.97] | 0.00 |
| Specificity | 0.82 [0.76–0.88] | 0.68 [0.62–0.75] | 1.00 |
| Precision | 0.46 [0.38–0.56] | 0.33 [0.27–0.39] | 0.00 |
| F2 | 0.73 [0.64–0.82] | 0.65 [0.56–0.73] | 0.00 |
| ROC-AUC | 0.944 [0.909–0.973] | 0.849 [0.780–0.915] | 0.50 |
| PR-AUC | 0.759 [0.636–0.882] | 0.606 [0.470–0.763] | 0.15 |
| Brier score | 0.100 | 0.167 | 0.15 |
| Missed / false alarms | 4 / 31 | 4 / 54 | 30 / 0 |

Glucose is the most important input (permutation importance 0.43). Without it, BMI leads (0.22), and recall holds at the cost of 23 more false alarms.

## 8. Discussion

**Simple won, and that's a useful result.** The data is near-linear, with only 118 training positives, so extra flexibility added variance and little else. This agrees with Christodoulou et al. (2019). Logistic Regression is also faster and easier to explain.

**Method mattered more than model.** Moving to repeated folds and paired comparisons changed the conclusion: one 5-fold split had Gradient Boosting ahead by 0.0003, which is noise.

**Negative results are still results.** Feature engineering and tuning gave no gain. Each had a pre-set rule, so these outcomes are evidence and not a failure to find something.

**Honest shortfalls.** Test recall missed the 0.90 target by one person. The tuned threshold's benefit on test was also small: with glucose it added 8 false alarms and caught no extra positives.

## 9. Limitations

- The label contradicts clinical knowledge, so results don't describe real diabetes risk.
- The data is synthetic, and real-world performance is unknown.
- 30 test positives give wide intervals.
- Test results for all models were seen in the advanced-model stage, before final selection. Selection used CV only, but the test set isn't fully fresh.
- `class_weight="balanced"` inflates scores, so they rank people but aren't probabilities.
- Fairness couldn't be assessed, because the data has no group attributes.

## 10. Future work

1. Confirm the label definition with the data provider. This comes first, because everything else depends on it.
2. Validate on real clinical data, with a new test set.
3. Recalibrate the scores if they're to be shown as risks.
4. Assess performance across patient groups.
5. Add SHAP explanations for individual predictions.
6. Add what the service needs for real use: authentication, monitoring and CI ([checklist](deployment_guide.md#before-real-use)).

## 11. Reproducing the results

```bash
pip install -r requirements-dev.txt
python -m pytest
python main.py
```

The seed is fixed, and repeat runs give identical optimization results. All tables come from `reports/`.

## References

Full references are in the main [README](../README.md#7-references).
