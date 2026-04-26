# AI-Powered Credit Scoring Model

A machine learning pipeline for credit default prediction and scorecard generation, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The model produces an interpretable 300–900 credit score (Indian bureau style) with per-applicant SHAP explanations and a built-in fairness audit.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Evaluation](#evaluation)
- [Credit Score Output](#credit-score-output)
- [Fairness & Explainability](#fairness--explainability)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)

---

## Overview

The notebook trains an ensemble of gradient boosting models to predict loan default risk, then maps default probabilities to a human-readable credit score. It is designed to reflect realistic lending constraints: class imbalance handling, probability calibration, cost-sensitive threshold optimisation, protected attribute removal, and per-decision explanations.

---

## Project Structure

```
Credit_Score_model.ipynb   # Main notebook (all sections below)
data/
  application_train.csv    # Main applicant table (307,511 rows × 122 cols)
  bureau.csv               # Credit history from external bureaus
  bureau_balance.csv       # Monthly repayment status per bureau loan
  installments_payments.csv
  credit_card_balance.csv
  previous_application.csv
  POS_CASH_balance.csv     # (optional)
```

---

## Dataset

| Table | Description |
|---|---|
| `application_train.csv` | Core applicant demographics and loan details. `TARGET=1` means 60+ consecutive days overdue. Class ratio ≈ 11:1. |
| `bureau.csv` | All prior loans reported to the credit bureau across all lenders |
| `bureau_balance.csv` | Month-by-month DPD status for each bureau loan |
| `installments_payments.csv` | Scheduled vs actual payments for prior Home Credit loans |
| `credit_card_balance.csv` | Monthly credit card statements — utilisation, DPD, drawings |
| `previous_application.csv` | All prior applications at Home Credit and their outcomes |
| `POS_CASH_balance.csv` | Monthly POS and cash loan snapshots (optional) |

---

## Pipeline

The notebook is structured as 16 sections:

1. **Configuration** — all file paths, model constants, and score range set in one cell
2. **Imports**
3. **Load Data** — shape summary, class imbalance visualisation
4. **Auxiliary Table Aggregation** — bureau, bureau balance, installments, credit card, previous applications, POS cash
5. **Feature Engineering**
   - Merge all auxiliary aggregations
   - Previous-application cross-features (credit escalation, recency)
   - Financial ratios and employment/age features
   - Alternative data composite scores (`payment_discipline_score`, `utilisation_health`, `bureau_depth_score`, `stability_score`, `alt_credit_score`)
   - High-signal binary flags and interaction terms
   - Trend cross-features (worsening × low external score)
   - Deep feature engineering (per-credit-type bureau splits, per-loan instalment aggregation)
6. **EDA** — density plots by target class, correlation heatmap
7. **Preprocessing & Feature Selection**
   - Protected attributes and high-missing columns dropped
   - Label encoding → stratified 80/20 split
   - K-fold smoothed target encoding for high-cardinality categoricals
   - Missing-indicator flags
   - Inf sanitisation → median imputation
   - Variance threshold + high-correlation (>0.90) filtering
   - 10% calibration holdout carved from train
   - SMOTE on model-training portion only
8. **Model Training** — four families (see below)
9. **Evaluation** — ROC/PR curves, KS statistic
10. **Probability Calibration** — Platt scaling on held-out calibration set
11. **Threshold Optimisation** — Youden J and cost-sensitive (FN cost = 5× FP)
12. **SHAP Explainability** — global bar, beeswarm, per-applicant waterfall
13. **Fairness Audit** — four-fifths rule across income quartiles and age bands
14. **Credit Score Function** — probability → 300–900 integer score with decision bands
15. **Sample Results** — score distributions by actual default outcome
16. **Final Summary**

---

## Models

| Model | Training Data | Imbalance Strategy |
|---|---|---|
| Logistic Regression | SMOTE-balanced | Via SMOTE + `class_weight="balanced"` |
| Random Forest | SMOTE-balanced | `class_weight="balanced_subsample"` |
| XGBoost | Raw (imbalanced) | `scale_pos_weight` |
| LightGBM | Raw (imbalanced) | `sample_weight` (balanced) |
| CatBoost (optional) | Raw (imbalanced) | `scale_pos_weight` |

XGBoost and LightGBM both support optional Optuna hyperparameter tuning (50 trials, TPE sampler with multivariate mode). Early stopping uses an internal 15% validation split; the best iteration is then used to retrain on 100% of the training data.

The final output is an **ensemble** of the best-performing models, evaluated across three blending strategies — OOF stacking (logistic regression meta-learner), rank averaging, and weighted averaging — with the highest-AUC strategy selected automatically.

---

## Evaluation

Primary metric is **AUC-ROC** (appropriate for imbalanced credit risk). Supplementary metrics reported for each model:

- Average Precision (AP)
- Accuracy, Precision, Recall, F1 at the chosen threshold
- **KS Statistic** (KS > 0.40 is considered strong for a credit scorecard)

---

## Credit Score Output

Default probability is mapped to a 300–900 integer score:

```
Score = 300 + (1 − default_probability) × 600
```

**Decision bands:**

| Score Range | Decision |
|---|---|
| ≥ 750 | **APPROVE** |
| 650–749 | **APPROVE WITH LOWER LIMIT** |
| < 650 | **DECLINE** |

Each applicant report also surfaces the top 5 SHAP contributors with direction (↑ risk / ↓ risk) and the raw feature value, intended for use by underwriters or for regulatory explainability requirements.

---

## Fairness & Explainability

**Protected attributes removed from the model:** gender (`CODE_GENDER`), family status (`NAME_FAMILY_STATUS`), and organisation type (`ORGANIZATION_TYPE`).

**Fairness audit** checks the four-fifths (80%) rule: the approval rate for any subgroup must be ≥ 80% of the highest-approval group. This is run across income quartiles and age bands (18–25, 25–35, 35–45, 45–55, 55–65, 65+). Groups that fail the threshold are highlighted in red.

**SHAP** (TreeExplainer) provides three views:
- Global feature importance (mean |SHAP|)
- Beeswarm showing direction and spread per feature
- Per-applicant waterfall for individual decisions

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn        # SMOTE
xgboost
lightgbm
shap
optuna                  # optional, for hyperparameter tuning
catboost                # optional
scipy
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap optuna catboost scipy
```

---

## Usage

1. Place all CSV files in the `./data/` directory (or update `DATA_DIR` in the configuration cell).
2. Open `Credit_Score_model.ipynb` in Jupyter and run all cells in order.
3. The notebook is designed so that only **Section 1 (Configuration)** needs to be edited to adapt it to a different environment.

To score a new applicant after training:

```python
raw_prob = final_model.predict_proba(applicant_features)[0][1]
calibrated_prob = float(expit(a * raw_prob + b))   # Platt params a, b from Section 10
score = compute_credit_score(calibrated_prob)
decision, _ = loan_decision(score, calibrated_prob)
```

---

## Configuration

All tunable constants live in the first notebook cell:

| Parameter | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data/` | Folder containing input CSVs |
| `RANDOM_STATE` | 42 | Global random seed |
| `TEST_SIZE` | 0.20 | Train/test split ratio |
| `CV_FOLDS` | 5 | Number of cross-validation folds |
| `N_OPTUNA_TRIALS` | 50 | Optuna trials per model |
| `SHAP_SAMPLE` | 500 | Rows sampled for SHAP (speed vs detail) |
| `SCORE_MIN` | 300 | Minimum credit score |
| `SCORE_MAX` | 900 | Maximum credit score |
| `APPROVE_THRESHOLD` | 750 | Score above which applications are approved |
| `REVIEW_THRESHOLD` | 650 | Score above which applications go to review |
