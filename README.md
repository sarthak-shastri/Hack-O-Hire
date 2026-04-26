# AI-Powered Credit Scoring Model

## Overview

This is an **end-to-end machine learning pipeline** for credit risk assessment that predicts loan default probability and assigns a credit score (300–900 range, matching Indian credit bureau standards). The model integrates **alternative data sources**, **ensemble learning**, **fairness auditing**, and **explainability via SHAP**.

---

## Dataset & Problem

**Source:** Home Credit Default Risk (Kaggle competition)  
**Training samples:** 307,511 applicants  
**Target:** Binary classification (default in 60+ consecutive days = 1, non-default = 0)  
**Class imbalance:** ~11:1 (non-default : default)

### Data Architecture

Six auxiliary tables enrich the main application dataset:

| Table | Rows | Signal | Key Features |
|-------|------|--------|-------------|
| `application_train.csv` | 307K | Core application + demographics | Income, age, employment, assets |
| `bureau.csv` | 1.7M | Credit bureau history | Loan counts, credit amounts, overdue days |
| `bureau_balance.csv` | 27M | Monthly repayment status by DPD | Delinquency trends, payment consistency |
| `installments_payments.csv` | 13M | Prior Home Credit loan records | On-time vs late payments, underpayments |
| `credit_card_balance.csv` | 3.8M | Credit card statements (monthly) | Utilisation, DPD, payment-to-minimum ratio |
| `previous_application.csv` | 1.7M | All prior Home Credit applications | Approval/refusal history, credit escalation |

---

## 🔧 Feature Engineering (240+ final features)

### Layer 1: Auxiliary Aggregation
Raw tables are grouped by applicant ID and aggregated into **loan-level, contract-level, and applicant-level** statistics:

- **Bureau:** active loan count, total debt, max overdue days, debt-to-credit ratio
- **Bureau Balance:** average/max DPD status, clean rate, recent vs historical trend
- **Installments:** late payment rate, underpayment frequency, payment consistency
- **Credit Cards:** average utilisation, DPD count, payment-to-minimum ratio, high-utilisation flag
- **Previous Apps:** approval rate, refusal rate, credit escalation pattern

### Layer 2: Domain Features
Traditional financial ratios and lifecycle indicators:

- **Debt burden:** annuity-to-income, credit-to-income, debt service ratio
- **Stability:** employment-to-age, registration tenure, phone change recency
- **Credit appetite:** bureau loan count divided by age (credit hunger)
- **External score composites:** mean, harmonic mean, missing indicators

### Layer 3: Composite Scores
Four domain-specific scores combined into a single `alt_credit_score`:

```
Payment Discipline Score  = 0.5 × (1 - inst_late_rate) + 0.5 × cc_payment_rate
Utilisation Health        = 1 - cc_avg_utilisation
Bureau Depth Score        = (time span of bureau credit) / 99th percentile
Stability Score           = 0.6 × employment_to_age + 0.4 × (age / 70)

alt_credit_score          = 0.35 × payment + 0.25 × util + 0.20 × depth + 0.20 × stability
```

### Layer 4: High-Signal Cross-Features
Interaction terms capturing specific risk patterns:

- **Debt stress:** bureau_total_overdue / income
- **Recent deterioration:** inst_recent_late_rate - inst_lifetime_late_rate (clipped at 0)
- **High-utilisation flag:** cc_avg_utilisation > 0.9
- **Total DPD risk:** weighted sum of all delinquency sources (bureau, installment, credit card)
- **Refusal × Low External Score:** prior loan refusal × (1 - external_score)

### Layer 5: Deep Segmentation
Bureau and installment tables are re-aggregated by **credit type** (consumer, car, mortgage, credit card) and **loan tenure** (recent 12m vs historical), capturing which product categories a borrower struggles with.

### Layer 6: Feature Selection
- **Variance threshold:** drop features with <0.1% variance
- **Correlation pruning:** drop one feature from pairs with >90% correlation (keeping higher predictive power)
- **Importances filter (LGB only):** drop bottom 15% by model importance

---

## Model Architecture

### Four Base Learners

| Model | Training | Imbalance Handle | Tuning |
|-------|----------|------------------|--------|
| **Logistic Regression** | SMOTE-balanced | Class weight | Manual C=0.1 |
| **Random Forest** | SMOTE-balanced | balanced_subsample | Fixed 300 trees |
| **XGBoost** | Raw (imbalanced) | scale_pos_weight | Optuna (50 trials) |
| **LightGBM** | Raw (imbalanced) | sample_weight | Optuna (40 trials) + DART boost |

### Training Pipeline

1. **Stratified 80/20 split** on application-level (preserves class ratio in train & test)
2. **Carve out 10% calibration set** from train (never seen by SMOTE or model training)
3. **SMOTE on model-training set only** (90% of train) → balanced training set
4. **Variance + correlation filtering** on the imputed feature matrix
5. **Early stopping:** tree models use 15% internal validation split
6. **Calibration:** Platt sigmoid fitted on held-out calibration set to map raw probabilities to true frequencies

### Ensemble Strategy

Three ensemble approaches are evaluated and the **best-performing** is selected:

```
1. Stacking:      Meta-learner (logistic regression) trained on OOF predictions
2. Rank Average:  Aggregate quantile-normalized predictions
3. Weighted:      Manual weights (XGB tuned 45%, LGB 35%, tuned LGB 15%, stack 5%)
```

**Final model:** Ensemble of calibrated base learner predictions

---

## Model Performance

### Benchmark Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **AUC-ROC** | 0.7850+ | Strong ranking ability; separates defaulters from non-defaulters |
| **Avg Precision** | 0.4500+ | Reasonable precision among flagged applicants |
| **F1 Score** | 0.5200+ | Balanced recall–precision tradeoff |
| **KS Statistic** | 0.4100+ | >0.40 is considered excellent for credit (max=1.0) |
| **Accuracy** | 0.7350+ | High due to class imbalance; not primary metric |

### Threshold Optimisation

Default 0.5 probability threshold is inappropriate for imbalanced datasets. Two thresholds are computed:

- **Youden J:** Maximises sensitivity + specificity simultaneously
- **Cost-sensitive:** Assumes false negative (approving a defaulter) costs 5× false positive

**Used threshold:** Cost-optimal (typically 0.38–0.42)

---

## Credit Score & Decision Logic

### Score Mapping

```
Raw probability p ∈ [0, 1]  →  Score = 300 + (1 - p) × 600
```

**Range:** 300 (highest default risk) → 900 (lowest default risk)

### Decision Bands

| Band | Threshold | Action | Risk Profile |
|------|-----------|--------|--------------|
| **Approve** | Score ≥ 750 | Auto-approve | Prime (very low risk) |
| **Review** | 650 ≤ Score < 750 | Manual underwriting | Sub-prime (moderate risk) |
| **Decline** | Score < 650 | Decline | High risk |

### Per-Applicant Report

Each decision includes:
- Credit score (300–900)
- Default probability (%)
- Decision category
- Top 5 SHAP contributors explaining the score

Example:
```
╔═══════════════════════════════╗
║  CREDIT DECISION REPORT       ║
║  Credit Score: 720 / 900      ║
║  Default Prob: 0.2667 (26.7%) ║
║  Decision: REVIEW             ║
║  Risk Band: SUB-PRIME         ║
╠═══════════════════════════════╣
║  Top drivers (↑ risk):        ║
║  • cc_avg_utilisation: 0.94   ║
║  • bbal_max_status: 4         ║
║  • inst_late_rate: 0.15       ║
║  • bureau_total_overdue: 5000 ║
║  • employment_to_age: 0.35    ║
╚═══════════════════════════════╝
```

---

## Fairness & Compliance

### Removed Protected Attributes

The model does **NOT** use:
- `CODE_GENDER` (gender)
- `NAME_FAMILY_STATUS` (marital status)
- `ORGANIZATION_TYPE` (employer category)

All features are financial / behavioural, not demographic.

### Four-Fifths Rule Audit

Approval rates are compared across **income quartiles** and **age bands**. Subgroup approval must be ≥80% of the highest-approval group.

Example:
```
Income Q1 (Low):    82% approval → 0.95 ratio (PASS)
Income Q2:          85% approval → 0.98 ratio (PASS)
Income Q3:          87% approval → 1.00 ratio (PASS)
Income Q4 (High):   86% approval → 0.99 ratio (PASS)

All groups PASS the four-fifths rule ✓
```

---

## Explainability

### Global Feature Importance

Model-agnostic via SHAP TreeExplainer:
- **Bar plot:** Mean absolute SHAP by feature (top 20)
- **Beeswarm:** Distribution of SHAP values, color-coded by feature value (red=high, blue=low)

Interpretation: Features pushed above the baseline (left) increase default risk; below baseline (right) decrease it.

### Per-Applicant Explanation

SHAP **Waterfall plot** for individual applicants:
1. Base value (population average default probability)
2. Each feature's SHAP value (contribution to this applicant's prediction)
3. Final predicted probability

Example waterfall for a denied applicant:
```
base_value                          = 0.0900
cc_avg_utilisation (0.95) ↑        = +0.0850  (red = increases risk)
bbal_max_status (5) ↑              = +0.0620
inst_late_rate (0.25) ↑            = +0.0410
employment_to_age (0.10) ↓         = -0.0180 (blue = decreases risk)
age_years (28) ↓                   = -0.0150
---
predicted_probability              = 0.3450 → DECLINE
```

---

## Preprocessing & Validation

### Missing Data Strategy

1. **Deletion:** Drop columns with >60% missing (signal too sparse)
2. **Missing indicators:** Flag key columns where absence carries signal
   - EXT_SOURCE_1/2/3 missing → often poor bureau data
   - bureau_loan_count missing → no credit history
3. **Median imputation:** Remaining features filled with training-set median
4. **Inf sanitisation:** Replace inf/-inf with NaN, clip extreme floats

### Target Encoding

High-cardinality categoricals (OCCUPATION_TYPE, INCOME_TYPE, etc.) are **K-fold smoothed target encoded**:

```
smoothed_mean = (count × class_mean + smoothing_weight × global_mean) 
                / (count + smoothing_weight)
```

Uses 5 folds + smoothing weight of 10 to prevent target leakage and overfitting.

### Class Imbalance

- **SMOTE:** Oversample minority class on model-training set only (90% of train)
- **No SMOTE on:** calibration set (10%), test set (untouched), XGB/LGB (use scale_pos_weight instead)
- **Reason:** SMOTE creates synthetic samples; test/calib must stay real

---

## Usage & Deployment

### Scoring an Applicant

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model, scaler, feature columns
model = load_model("final_credit_model.pkl")
scaler = load_scaler("scaler.pkl")
feature_cols = load_json("feature_columns.json")

# Prepare applicant data
applicant_raw = pd.read_csv("new_applicant.csv")
applicant = applicant_raw[feature_cols]

# Preprocess
applicant = handle_missing(applicant, strategy="median")
applicant = scaler.transform(applicant)

# Predict
prob = model.predict_proba(applicant)[0][1]  # default probability
score = 300 + (1 - prob) * 600  # credit score

# Decision
if score >= 750:
    decision = "APPROVE"
elif score >= 650:
    decision = "REVIEW"
else:
    decision = "DECLINE"

print(f"Credit Score: {score:.0f} | Decision: {decision}")
```

### Batch Scoring

```python
# Score 10k applicants in parallel
applicants_batch = pd.read_csv("batch_10k.csv")
probs = model.predict_proba(applicants_batch[feature_cols])[:, 1]
scores = 300 + (1 - probs) * 600
applicants_batch["credit_score"] = scores
applicants_batch["decision"] = applicants_batch["credit_score"].apply(
    lambda s: "APPROVE" if s >= 750 else "REVIEW" if s >= 650 else "DECLINE"
)
applicants_batch.to_csv("scored_batch_10k.csv", index=False)
```

---

## Dependencies

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
xgboost>=1.5
lightgbm>=3.3
shap>=0.41
optuna>=2.10
matplotlib>=3.4
seaborn>=0.11
scipy>=1.7
imbalanced-learn>=0.8  (for SMOTE)
catboost>=1.0  (optional)
```

### Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap optuna matplotlib seaborn scipy imbalanced-learn
```

---

##  File Structure

```
credit-score-model/
├── Credit_Score_model.ipynb          # Full pipeline notebook
├── README.md                           # This file
├── data/
│   ├── application_train.csv          # Main applicant data
│   ├── bureau.csv                     # Credit bureau history
│   ├── bureau_balance.csv             # Monthly bureau status
│   ├── installments_payments.csv      # Installment records
│   ├── credit_card_balance.csv        # Credit card statements
│   └── previous_application.csv       # Prior applications
├── models/
│   ├── final_xgb_model.pkl           # XGBoost (tuned)
│   ├── lgb_gbdt.pkl                   # LightGBM (base)
│   ├── lgb_tuned.pkl                  # LightGBM (tuned)
│   ├── final_ensemble.pkl             # Stacked ensemble
│   └── feature_columns.json           # Feature list
├── outputs/
│   ├── model_comparison.csv           # Benchmark metrics
│   ├── fairness_audit.csv             # Four-fifths rule check
│   ├── feature_importance.png         # SHAP bar plot
│   └── credit_score_distribution.png  # Score histograms
└── utils/
    ├── preprocessing.py               # Data pipeline
    ├── scoring.py                     # Score computation
    └── fairness.py                    # Audit functions
```

---

##  Key Takeaways

1. **Ensemble strength:** Combining XGB, LGB variants + stacking beats any single model
2. **Feature engineering dominates:** 240 engineered features from raw 122 + auxiliary tables
3. **Calibration matters:** Platt sigmoid improves alignment of probabilities to true defaults
4. **Fairness first:** Protected attributes removed; four-fifths rule audited across demographics
5. **Explainability essential:** SHAP provides actionable per-applicant insights
6. **Threshold tuning:** Cost-sensitive optimisation (FN cost 5× FP cost) reflects lending economics
7. **Imbalance handling:** SMOTE on train-only, scale_pos_weight on XGB/LGB prevents leakage

---

##  References

- **SHAP:** Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- **XGBoost:** Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- **LightGBM:** Ke et al. (2017). "LightGBM: A Fast, Distributed, High-Performance Gradient Boosting Framework"
- **Fairness:** Feldman et al. (2015). "Certifying and Removing Disparate Impact"
- **Imbalanced Learning:** Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"

---



**Last updated:** April 2026

---
