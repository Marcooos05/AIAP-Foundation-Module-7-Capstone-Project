# Model explanation

What the model does, how it decides, how well it works and when not to use it. Written for reviewers and readers without a machine learning background. It follows the structure of a model card, and the reporting topics of TRIPOD+AI.

> **Do not use this model for real clinical decisions.** The data is synthetic, and people labelled Positive look *healthier* than people labelled Negative on almost every measure. Until the meaning of the label is confirmed, the scores describe this dataset and not real diabetes risk.

## Summary

| | |
|---|---|
| **What it does** | Gives each person a score and flags those above a threshold for follow-up |
| **Model** | Logistic Regression |
| **Inputs** | Up to 7 routine health measures |
| **Variants** | `with_glucose` (7 inputs) and `without_glucose` (6 inputs, no lab test needed) |
| **Goal** | Catch at least 90% of positives, accepting extra false alarms |
| **Test result** | Catches 26 of 30 positives (recall 0.87) in both variants |

## Intended use

**As designed:** a first-pass screen that flags people for a cheap follow-up test, such as fasting glucose or HbA1c. It doesn't diagnose anyone.

**Out of scope:**

- Any real clinical decision, until the limitations below are resolved
- Diagnosing diabetes
- Telling someone they're *not* at risk: about 1 in 8 positives is missed
- Reading the score as a probability
- People under 18, who were excluded from the data
- Populations other than the one in this dataset

## Data

- **Source:** `synthetic_data_166.csv`, 1,000 synthetic records. The generator and the label definition are unknown.
- **Cleaning:** 3 under-18s were dropped, leaving 997 rows, 148 of them Positive (14.8%). Outliers were kept, because 35% of outlier rows are Positive.
- **Split:** 797 training rows (118 Positive) and 200 test rows (30 Positive), stratified.

| Input | Assumed unit |
|---|---|
| `age` | Years |
| `bmi` | kg/m² |
| `blood_pressure` | mmHg, likely systolic |
| `fasting_glucose_level` | mg/dL (`with_glucose` only) |
| `cholesterol_level` | mg/dL, likely total |
| `physical_activity_frequency` | Proportion, 0 to 1 |
| `family_history_of_diabetes` | Absent / Present |

Units aren't documented, so they're inferred. Seven engineered features were tested and none improved cross-validated performance, so the model uses these raw inputs.

## How the model decides

Logistic Regression adds up a weighted score from the inputs and converts it to a number from 0 to 1. Each input is first rescaled, so that one unit means one standard deviation. Weights can then be compared directly.

The table shows how each input changes the odds of being labelled Positive. A multiplier above 1 raises the odds, and one below 1 lowers them.

| Input | Odds multiplier: with glucose | Without glucose | Direction expected clinically? |
|---|---|---|---|
| Fasting glucose, +1 SD | **0.13** | not used | ✗ Opposite |
| Family history present (vs absent) | **3.31** | **2.39** | ✓ Yes |
| Age, +1 SD | 0.50 | 0.60 | ✗ Opposite |
| BMI, +1 SD | 0.63 | **0.43** | ✗ Opposite |
| Physical activity, +1 SD | 1.24 | 1.44 | ✗ Opposite |
| Blood pressure, +1 SD | 0.82 | 0.74 | ✗ Opposite |
| Cholesterol, +1 SD | 1.20 | 1.08 | ✓ Yes here, though weak. On its own it runs the opposite way |

Multipliers are the exponential of each coefficient in the saved models, and they hold the other inputs constant.

**How much each input matters overall.** Permutation importance shuffles one input and measures how much test PR-AUC falls:

| Input | With glucose | Without glucose |
|---|---|---|
| Fasting glucose | 0.43 | not used |
| Age | 0.07 | 0.10 |
| Family history | 0.06 | 0.08 |
| BMI | 0.03 | 0.22 |
| Physical activity | 0.03 | 0.06 |
| Blood pressure | 0.01 | 0.01 |
| Cholesterol | 0.00 | 0.00 |

Glucose dominates when it's available. Without it, BMI takes over.

**A worked example.** These two people were scored by the running API with the `without_glucose` model. The threshold is 0.404.

| | Person A | Person B |
|---|---|---|
| Age | 50 | 104 |
| BMI | 28 | 37 |
| Blood pressure | 125 | 164 |
| Family history | Absent | Absent |
| **Score** | **0.414, flagged** | **0.008, not flagged** |

Person B is older, heavier and hypertensive, which medicine would treat as higher risk, yet the model gives a far lower score. This is what the limitation below means in practice.

## The decision threshold

The score alone doesn't say who to flag, so a threshold turns it into a yes or no. A lower threshold catches more positives but raises more false alarms.

A missed case costs more than a follow-up test, so the threshold is the highest one that still reaches 90% recall on out-of-fold training predictions:

| | With glucose | Without glucose |
|---|---|---|
| Threshold | 0.418 | 0.404 |
| Out-of-fold recall | 0.907 | 0.907 |

## Performance

Measured on the 200 test rows, with 95% bootstrap intervals in brackets.

| Metric | Meaning | With glucose | Without glucose | Always "Negative" |
|---|---|---|---|---|
| **Recall** | Share of positives caught | 0.87 [0.73–0.97] | 0.87 [0.73–0.97] | 0.00 |
| Specificity | Share of negatives left unflagged | 0.82 [0.76–0.88] | 0.68 [0.62–0.75] | 1.00 |
| Precision | Share of flags that are positive | 0.46 [0.38–0.56] | 0.33 [0.27–0.39] | 0.00 |
| F2 | Balance favouring recall | 0.73 [0.64–0.82] | 0.65 [0.56–0.73] | 0.00 |
| ROC-AUC | Ranking quality | 0.944 [0.909–0.973] | 0.849 [0.780–0.915] | 0.50 |
| PR-AUC | Ranking quality for rare positives | 0.759 [0.636–0.882] | 0.606 [0.470–0.763] | 0.15 |
| Brier score | Probability error; lower is better | 0.100 | 0.167 | 0.15 |

**In practical terms**, out of these 200 people, 30 of them Positive:

| | With glucose | Without glucose |
|---|---|---|
| Positives caught | 26 | 26 |
| Positives missed | 4 | 4 |
| False alarms | 31 | 54 |

Adding the glucose test keeps recall the same with 23 fewer false alarms.

**Compared with other models.** Random Forest and Gradient Boosting were tuned the same way and did not beat Logistic Regression. Logistic Regression also had the highest repeated-CV PR-AUC: 0.690 with glucose and 0.493 without.

## Calibration: why the score isn't a probability

Training gave extra weight to the rarer Positive class (`class_weight="balanced"`), which pushes scores upward. A score of 0.66 doesn't mean a 66% chance. Scores are good for ranking people, and should not be quoted as risks. See `reports/figures/calibration_curve_test__with_glucose.png`.

## Limitations

- **The label is suspect.** 6 of 7 features relate to the label in the opposite direction to clinical knowledge. None of the 86 people with glucose ≥ 126 mg/dL, the diagnostic threshold, is labelled Positive. Flipping the labels doesn't explain it, because family history would then become protective and 85% of people would be Positive. The likely cause is a synthetic generator with non-clinical rules, but this is unconfirmed and should be raised with the data provider.
- **Synthetic data.** Nothing here has been checked on real patients. Performance in the real world is unknown.
- **Small sample.** 30 test positives give wide intervals, and one person moves recall by 3.3 points.
- **Recall is below target on test.** It reached 0.87 against the 0.90 target, one person short and well within the interval.
- **The threshold's value isn't proven.** On test, the tuned threshold added 8 false alarms with glucose and caught no extra positives, and without glucose it added 10 and caught one more.
- **The test set isn't fully fresh.** Test results for all models were reported before the final selection. The selection itself used training data only, with its rule fixed in advance.
- **Feature importance uses the test set,** and with so few positives its small values are noisy.
- **Fairness wasn't assessed.** The data has no sex, ethnicity or other group attributes, so performance across groups is unknown.

## What must happen before real use

1. Confirm what the label means.
2. Retrain and validate on real clinical data, ideally from another site.
3. Assess performance across patient groups.
4. Recalibrate the scores if they'll be shown as risks.
5. Follow the relevant regulations for clinical decision support.

Re-running this project on real data is a separate exercise, and these results shouldn't be carried over.
