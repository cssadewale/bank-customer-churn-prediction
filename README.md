# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **A full end-to-end machine learning project** — from raw data to a deployment-ready churn prediction model, built as part of my data science portfolio.

---

## 📌 Project Overview

A multinational retail bank is losing customers at an alarming rate. Every churned customer represents a direct revenue loss, and acquiring a new customer costs **5 to 7 times more** than retaining an existing one.

This project builds a **binary classification model** that predicts whether a bank customer will churn, using their demographic, financial, and behavioural profile. The final model is deployment-ready and designed to power a real-time retention scoring system.

**Target Variable — `churn`:**
- `1` → The customer **has churned** (exited the bank)
- `0` → The customer **has not churned** (is still active)

---

## 🎯 Project Objective

To develop a **robust, interpretable binary classification model** that:
- Accurately identifies customers at high risk of churning
- Quantifies the relative importance of each contributing risk factor
- Provides actionable intelligence for the bank's retention team to prioritise outreach

---

## 📁 Repository Structure

```
bank-customer-churn-prediction/
│
├── 📓 Churn_Prediction_Portfolio_PROFESSIONAL.ipynb   ← Main project notebook
├── 📄 README.md                                        ← You are here
├── 📄 requirements.txt                                 ← Python dependencies
├── 📄 .gitignore                                       ← Files excluded from Git
│
├── 📂 data/
│   └── churn_data.csv                                  ← Dataset (10,000 customer records)
│
├── 📂 models/
│   ├── best_gradient_boosting_churn_model.joblib       ← Saved trained model
│   └── standard_scaler.joblib                          ← Saved fitted scaler
│
└── 📂 outputs/
    ├── target_distribution.png
    ├── numerical_distributions.png
    ├── categorical_distributions.png
    ├── correlation_heatmap.png
    ├── numerical_vs_churn.png
    ├── categorical_vs_churn.png
    ├── multivariate_analysis.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    └── feature_importance_gb.png
```

---

## 🔄 Project Workflow

The notebook follows a structured, step-by-step data science methodology:

| Step | Phase | Description |
|:----:|-------|-------------|
| **1** | Basic Data Exploration | Data loading, inspection, cleaning, skewness and outlier analysis |
| **2** | Exploratory Data Analysis | Univariate, Bivariate, and Multivariate analysis |
| **3** | Feature Engineering & Preprocessing | Encoding, scaling, and train-test split |
| **4** | Model Development | Training, evaluation, tuning, interpretability, and saving |
| **5** | Business Insights | Key findings, recommendations, limitations, and project summary |

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle — Bank Customer Churn Dataset |
| **Records** | 10,000 customer records |
| **Features** | 10 predictive features + 1 binary target |
| **Target** | `churn` (0 = Retained, 1 = Churned) |
| **Class ratio** | 79.63% No Churn : 20.37% Churned (3.91:1 imbalance) |

**Feature descriptions:**

| Feature | Type | Description |
|---------|------|-------------|
| `credit_score` | Numerical | Customer credit score (350–850) |
| `geography` | Categorical | Country of residence (France, Germany, Spain) |
| `gender` | Categorical | Male / Female |
| `age` | Numerical | Customer age in years |
| `tenure` | Numerical | Number of years as a bank customer (0–10) |
| `balance` | Numerical | Account balance in £ |
| `num_of_products` | Categorical | Number of bank products held (1–4) |
| `has_cr_card` | Binary | Whether the customer holds a credit card (0/1) |
| `is_active_member` | Binary | Whether the customer is an active member (0/1) |
| `estimated_salary` | Numerical | Estimated annual salary in £ |

---

## 🔍 Key EDA Findings

These are the most important patterns discovered during Exploratory Data Analysis:

| Finding | Detail |
|---------|--------|
| 🔴 **Age is the #1 churn driver** | Churned customers are **7.43 years older** on average. The 51–60 band churns at **56.21%** — over 1 in 2 customers |
| 🔴 **Products follow a U-shaped risk** | 2-product customers churn at **7.58%** (lowest). 3-product: **82.71%**. 4-product: **100%** |
| 🔴 **Germany churns at double the rate** | Germany: **32.44%** vs France: **16.15%** and Spain: **16.67%** |
| 🔴 **High-balance customers are leaving** | Churned customers hold **£18,364 higher mean balance** — the bank is losing its most valuable depositors |
| ⚠️ **Inactivity is a leading signal** | Inactive members churn at **26.85%** vs **14.27%** for active members |
| ⚠️ **Female customers churn more** | Female: **25.07%** vs Male: **16.46%** — a consistent **8.61 pp gap** across all geographies |
| ✅ **Credit card ownership is irrelevant** | Churn rates: 20.81% (no card) vs 20.18% (has card) — only **0.63 pp difference** |

---

## 🤖 Model Development & Results

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression (baseline) | 0.8080 | 0.5891 | 0.1867 | 0.2836 | 0.7748 |
| Gradient Boosting (default) | 0.8670 | 0.7681 | 0.4963 | 0.6030 | 0.8693 |
| **Tuned Gradient Boosting ✅** | **0.8710** | **0.7945** | **0.4939** | **0.6091** | **0.8675** |

### Why Gradient Boosting Wins

Logistic Regression captures only **1 in 5 actual churners** (Recall = 0.1867) because its linear decision boundary cannot model the non-linear patterns in this dataset — particularly the `num_of_products` U-shape and the `age` acceleration above 40. Gradient Boosting's sequential tree-splitting captures these patterns directly, improving F1-Score by **+31.9 percentage points**.

### Best Model — Tuned Hyperparameters (GridSearchCV)

```python
GradientBoostingClassifier(
    learning_rate     = 0.05,
    max_depth         = 5,
    min_samples_split = 5,
    n_estimators      = 200,
    random_state      = 42
)
```

**Cross-validation (5-fold):** Mean F1 = **0.5906 ± 0.0211** — stable and generalisable.

### Feature Importances

| Rank | Feature | Importance | Key EDA Signal |
|:----:|---------|:----------:|----------------|
| 1 | `age` | **33.05%** | Strongest numerical separator — 7.43-year mean gap |
| 2 | `num_of_products` | **23.30%** | U-shaped churn pattern |
| 3 | `balance` | **12.28%** | High-value depositors churning at higher rates |
| 4 | `is_active_member` | **10.21%** | Inactive members churn at nearly 2× the rate |
| 5 | `estimated_salary` | **6.67%** | Weak signal — near-uniform distribution |
| 6 | `geography_Germany` | **5.71%** | Germany churns at double France/Spain |
| 7 | `credit_score` | **5.00%** | Very weak separator |
| 8 | `tenure` | **1.70%** | No meaningful separation |
| 9 | `gender_Male` | **1.31%** | Captures female churn premium |
| 10 | `geography_Spain` | **0.50%** | Near-redundant — Spain ≈ France |
| 11 | `has_cr_card` | **0.26%** | Effectively zero — confirmed irrelevant |

---

## ⚙️ How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/cssadewale/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Notebook

```bash
jupyter notebook Churn_Prediction_Portfolio_PROFESSIONAL.ipynb
```

> **Note:** The dataset downloads automatically from Google Drive via `gdown` when you run the first data loading cell. No manual download is needed.

### 5. Load the Saved Model (Optional)

If you want to run inference directly without re-training:

```python
import joblib
import pandas as pd

# Load the saved model and scaler
model  = joblib.load('models/best_gradient_boosting_churn_model.joblib')
scaler = joblib.load('models/standard_scaler.joblib')

# Prepare a sample customer (after encoding geography/gender)
# Feature order: credit_score, age, tenure, balance, num_of_products,
#                has_cr_card, is_active_member, estimated_salary,
#                geography_Germany, geography_Spain, gender_Male

sample = pd.DataFrame([[650, 45, 3, 95000, 1, 1, 0, 80000, 1, 0, 0]],
                       columns=['credit_score','age','tenure','balance',
                                'num_of_products','has_cr_card','is_active_member',
                                'estimated_salary','geography_Germany',
                                'geography_Spain','gender_Male'])

# Scale the continuous features
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
sample[numerical_cols] = scaler.transform(sample[numerical_cols])

# Predict
prediction   = model.predict(sample)[0]
probability  = model.predict_proba(sample)[0][1]

print(f"Prediction  : {'CHURN RISK ⚠️' if prediction == 1 else 'RETAINED ✓'}")
print(f"Probability : {probability:.1%}")
```

---

## 📈 Business Recommendations

Based on the model's findings, the following retention strategies are recommended:

1. **Age-Segmented Retention Programme** *(Priority: Critical)* — Target customers aged 41–60 with personalised outreach, dedicated relationship managers, and preferential rates. The 51–60 band churns at 56.21%.

2. **Product Holding Optimisation** *(Priority: High)* — Investigate the 3 and 4-product customer experience urgently (82.7% and 100% churn rates). Push 1-product customers to the safe "2-product sweet spot" (7.58% churn).

3. **Germany-Specific Retention Strategy** *(Priority: High)* — German customers churn at double the rate of France/Spain. German females churn at 37.55% — the highest-risk demographic intersection.

4. **Inactive Member Re-Engagement** *(Priority: High)* — Automate an early-warning system for customers inactive for 60+ days. 4,849 inactive customers churn at 26.85%.

5. **High-Value Customer Priority** *(Priority: Medium)* — Prioritise outreach by CLV × churn probability. High-balance customers at high churn risk should receive same-day personal outreach.

6. **Monthly Batch Scoring** *(Priority: Medium)* — Deploy the model as a monthly scoring engine, segmenting all customers into High / Medium / Low risk tiers.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib / Seaborn** | Data visualisation |
| **Scikit-Learn** | Machine learning — preprocessing, modelling, evaluation |
| **Joblib** | Model serialisation and saving |
| **gdown** | Automated dataset download from Google Drive |
| **Jupyter Notebook** | Interactive development environment |

---

## 🔮 Next Steps

- [ ] Build a **Streamlit web app** for real-time churn risk scoring
- [ ] Explore **XGBoost / LightGBM** for performance improvements
- [ ] Engineer **temporal delta features** (change in balance, login frequency over time)
- [ ] Build a **CLV-weighted churn severity model**
- [ ] Establish a **monthly automated retraining pipeline**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Adewale Samson Adeagbo**
Data Scientist | Data Analyst | Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?logo=github&logoColor=white)](https://github.com/cssadewale)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-adewalesamsonadeagbo-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adewalesamsonadeagbo)

---

*If you found this project useful or interesting, please consider giving it a ⭐ — it helps others discover the work!*
