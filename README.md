# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **A full end-to-end binary classification project** — from raw data to a deployment-ready model — predicting which bank customers are at risk of churning, with actionable business recommendations.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Project Workflow](#project-workflow)
5. [Key Findings](#key-findings)
6. [Model Performance](#model-performance)
7. [Feature Importance](#feature-importance)
8. [Business Recommendations](#business-recommendations)
9. [Repository Structure](#repository-structure)
10. [How to Run](#how-to-run)
11. [Technologies Used](#technologies-used)
12. [Connect with Me](#connect-with-me)

---

## Project Overview

Customer churn — when a customer closes their account and moves to a competitor — is one of the costliest problems in retail banking. Acquiring a new customer costs **5 to 7 times more** than retaining an existing one, and the customers who leave are often the most financially valuable.

This project builds a **Gradient Boosting Classifier** that predicts whether a bank customer will churn based on their demographic, financial, and behavioural profile. The model achieves a **ROC-AUC of 0.87** and is saved as a deployment-ready artefact for integration into a Streamlit web application.

---

## Business Problem

**Objective:** Identify at-risk customers before they leave so the retention team can intervene early and efficiently.

**Target Variable — `churn`:**
- `1` → Customer has **exited** the bank
- `0` → Customer **remains** active

**Constraints:**
- Class imbalance: only **20.37%** of customers churned (3.91:1 ratio)
- Accuracy is a misleading metric — a model that predicts "No Churn" for everyone achieves 79.63% without learning anything
- Primary evaluation metric: **F1-Score** and **ROC-AUC**

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| **Records** | 10,000 customer records |
| **Features** | 14 columns (3 identifiers + 10 predictors + 1 target) |
| **Missing Values** | Zero — completely clean dataset |
| **Duplicates** | Zero |

**Feature Summary:**

| Feature | Type | Description |
|---------|------|-------------|
| `credit_score` | Numerical | Customer credit score (350–850) |
| `geography` | Categorical | Country: France / Germany / Spain |
| `gender` | Categorical | Male / Female |
| `age` | Numerical | Customer age in years |
| `tenure` | Numerical | Years as a bank customer (0–10) |
| `balance` | Numerical | Account balance in GBP |
| `num_of_products` | Categorical | Number of bank products held (1–4) |
| `has_cr_card` | Binary | Has credit card (1) or not (0) |
| `is_active_member` | Binary | Active member (1) or inactive (0) |
| `estimated_salary` | Numerical | Estimated annual salary in GBP |
| `churn` *(target)* | Binary | Exited (1) or Retained (0) |

---

## Project Workflow

```
Raw Data
   │
   ▼
Step 1: Basic Data Exploration
   ├── Data loading & inspection
   ├── Column standardisation (snake_case)
   ├── Missing value report → 0 missing
   ├── Duplicate detection → 0 duplicates
   ├── Skewness analysis (age: +1.01 — only highly skewed predictor)
   └── Outlier detection (IQR) → 0 rows removed

Step 2: Exploratory Data Analysis
   ├── Univariate: target distribution, numerical histograms, categorical counts
   ├── Bivariate: correlation heatmap, box plots vs churn, churn rates by category
   └── Multivariate: age group × churn, geography × gender × churn

Step 3: Feature Engineering & Preprocessing
   ├── One-Hot Encoding: geography (→ Germany/Spain dummies), gender (→ Male dummy)
   ├── Feature/target separation: X (11 features), y (churn)
   ├── Stratified train-test split: 80/20, stratify=y
   └── StandardScaler: fit on X_train only → zero leakage

Step 4: Model Development
   ├── Baseline: Logistic Regression
   ├── Advanced: Gradient Boosting Classifier (200 estimators)
   ├── Comparison: 5-metric evaluation on test set
   ├── Tuning: GridSearchCV (16 combos × 5 folds = 80 fits, F1 objective)
   ├── Validation: 5-fold cross-validation (Mean F1 = 0.5906 ± 0.0211)
   ├── Interpretability: Feature importance ranking
   └── Saving: joblib serialisation (model + scaler)

Step 5: Business Insights
   ├── Key findings from data
   ├── Model recommendation
   ├── 6 actionable business recommendations
   └── Limitations and next steps
```

---

## Key Findings

### 🔴 Who is churning?

| Risk Factor | Churn Rate | vs Overall (20.4%) |
|-------------|-----------|---------------------|
| Age 51–60 | **56.2%** | +35.8 pp |
| 4 products | **100.0%** | +79.6 pp |
| 3 products | **82.7%** | +62.3 pp |
| Germany | **32.4%** | +12.0 pp |
| German females | **37.6%** | +17.2 pp |
| Inactive members | **26.9%** | +6.5 pp |
| Female customers | **25.1%** | +4.7 pp |

### ✅ Who is staying?
| Segment | Churn Rate |
|---------|-----------|
| 2 products | **7.6%** |
| Active members | **14.3%** |
| Age 18–30 | **7.5%** |
| French males | **12.7%** |

### 💡 Most Counterintuitive Finding
Churned customers have a **25% higher average account balance** (£91,109 vs £72,745). The bank is losing its most financially valuable depositors — churn is a **revenue concentration risk**, not just a volume problem.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.808 | 0.589 | 0.187 | 0.284 | 0.775 |
| **Tuned Gradient Boosting** | **0.871** | **0.795** | **0.494** | **0.609** | **0.868** |

**Improvement over baseline:**
- F1-Score: **+0.325** (+114%)
- Precision: **+0.206** (+35%)
- Recall: **+0.307** (+164%)
- ROC-AUC: **+0.093** (+12%)

**Cross-Validation (5-fold F1):** `0.5856 | 0.6038 | 0.5520 | 0.5996 | 0.6119`
→ Mean: **0.5906** | Std: **±0.0211** — stable and generalisable

**Tuned Hyperparameters (GridSearchCV):**
```
learning_rate     : 0.05
max_depth         : 5
min_samples_split : 5
n_estimators      : 200
```

---

## Feature Importance

| Rank | Feature | Importance | Key EDA Signal |
|------|---------|-----------|----------------|
| 1 | `age` | **33.1%** | Churners are 7.4 years older on average |
| 2 | `num_of_products` | **23.3%** | U-shape: 28% → 8% → 83% → 100% |
| 3 | `balance` | **12.3%** | Churners hold 25% higher balances |
| 4 | `is_active_member` | **10.2%** | Inactive churn at 26.9% vs 14.3% active |
| 5 | `estimated_salary` | **6.7%** | Weak signal — near-uniform distribution |
| 6 | `geography_Germany` | **5.7%** | Germany: 32.4% vs France: 16.2% |
| 7 | `credit_score` | **5.0%** | Minimal separation (651 vs 645) |
| 8 | `tenure` | **1.7%** | Identical medians (5.0 yrs both groups) |
| 9 | `gender_Male` | **1.3%** | Female churn premium: +8.6 pp |
| 10 | `geography_Spain` | **0.5%** | Spain ≈ France — near-redundant |
| 11 | `has_cr_card` | **0.3%** | 0.63 pp difference — negligible |

---

## Business Recommendations

1. **Age-Segmented Retention Programme** — Target the 41–60 age band proactively. The 51–60 group churns at 56.2%; reducing this to 45% would retain ~£11M in deposits.
2. **Product Experience Audit** — Investigate why 3–4 product customers churn at 83–100%. Fix the multi-product experience; cross-sell 1-product holders to the 2-product "sweet spot" (7.6% churn).
3. **Germany Retention Task Force** — German customers churn at 2× the rate of France/Spain. Launch dedicated NPS tracking and competitive pricing review for the German market.
4. **Inactive Member Re-engagement** — 4,849 inactive customers churn at 26.9%. Automated 60-day inactivity trigger → personalised outreach could retain ~124 additional customers.
5. **High-Value Customer Priority** — Rank at-risk customers by CLV × churn probability, not probability alone. Customers with balance > £100k and probability > 50% get same-day senior RM contact.
6. **Monthly Batch Scoring** — Deploy the model as a scoring engine. Score all customers monthly → High/Medium/Low risk tiers → retention team focuses on the top ~500.

---

## Repository Structure

```
bank-customer-churn-prediction/
│
├── 📓 Bank_Customer_Churn_Prediction.ipynb   ← Main notebook (all outputs included)
│
├── 📁 data/
│   └── Churn_Modelling_Dataset.csv           ← Raw dataset (10,000 records)
│
├── 📁 models/
│   ├── best_gradient_boosting_churn_model.joblib  ← Trained model (deployment-ready)
│   └── standard_scaler.joblib                     ← Fitted scaler (must accompany model)
│
├── 📄 requirements.txt                       ← Python dependencies
└── 📄 README.md                              ← This file
```

---

## How to Run

### Option 1: Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/cssadewale/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

**2. Create and activate a virtual environment** *(recommended)*
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the notebook**
```bash
jupyter notebook Bank_Customer_Churn_Prediction.ipynb
```

### Option 2: Run on Google Colab
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> Upload `Churn_Modelling_Dataset.csv` to your Colab session when prompted.

### Option 3: Use the Saved Model Directly
```python
import joblib
import pandas as pd

# Load model and scaler
model  = joblib.load('models/best_gradient_boosting_churn_model.joblib')
scaler = joblib.load('models/standard_scaler.joblib')

# Prepare a new customer record
customer = pd.DataFrame([{
    'credit_score': 620, 'age': 52, 'tenure': 3,
    'balance': 115000, 'num_of_products': 1,
    'has_cr_card': 1, 'is_active_member': 0,
    'estimated_salary': 88000,
    'geography_Germany': 1, 'geography_Spain': 0, 'gender_Male': 0
}])

# Scale numerical features
num_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
customer[num_cols] = scaler.transform(customer[num_cols])

# Predict
probability = model.predict_proba(customer)[0][1]
prediction  = model.predict(customer)[0]

print(f"Churn Prediction : {'⚠️  CHURN RISK' if prediction == 1 else '✓  RETAINED'}")
print(f"Churn Probability: {probability:.1%}")
```

---

## Technologies Used

| Category | Library / Tool | Purpose |
|----------|---------------|---------|
| **Language** | Python 3.10 | Core programming language |
| **Data Manipulation** | Pandas 2.0, NumPy | Data loading, cleaning, feature engineering |
| **Visualisation** | Matplotlib, Seaborn | EDA plots, confusion matrices, ROC curves |
| **Machine Learning** | scikit-learn 1.3 | Models, preprocessing, evaluation, GridSearchCV |
| **Model Persistence** | joblib | Saving and loading trained models |
| **Environment** | Google Colab / Jupyter | Development and execution |
| **Version Control** | Git, GitHub | Code hosting and portfolio display |

---

## Connect with Me

**Adewale Samson Adeagbo**
*Data Scientist | ML Engineer | STEM Tutor*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/adewalesamsonadeagbo)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/cssadewale)

---

*If this project was useful or gave you ideas, feel free to ⭐ star the repository — it helps others find it.*
