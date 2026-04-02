# ─────────────────────────────────────────────────────────────────────────────
#  Bank Customer Churn Prediction — Streamlit App
#  Author  : Adewale Samson Adeagbo
#  GitHub  : https://github.com/cssadewale
#  LinkedIn: https://www.linkedin.com/in/adewalesamsonadeagbo
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Bank Churn Predictor",
    page_icon  = "🏦",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & SCALER  (cached so they are only loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load the saved Gradient Boosting model and StandardScaler from disk.

    We build the path dynamically from the location of app.py itself.
    This means the path is always correct — whether you run locally
    with 'streamlit run app.py' or on Streamlit Cloud.

    Expected folder structure:
        bank-customer-churn-prediction/
        ├── app.py                          ← this file
        └── models/
            ├── best_gradient_boosting_churn_model.joblib
            └── standard_scaler.joblib
    """
    # BASE_DIR = the folder that contains app.py (works on all machines)
    BASE_DIR    = Path(__file__).resolve().parent
    models_dir  = BASE_DIR / "models"

    model  = joblib.load(models_dir / "best_gradient_boosting_churn_model.joblib")
    scaler = joblib.load(models_dir / "standard_scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA
# These lists must stay in exactly the same order as the training pipeline.
# ─────────────────────────────────────────────────────────────────────────────

# Columns the scaler was fitted on (continuous numerical only)
COLS_TO_SCALE = ["credit_score", "age", "tenure", "balance", "estimated_salary"]

# Final feature column order — exactly as X_train was presented to the model
FEATURE_COLS = [
    "credit_score", "age", "tenure", "balance", "num_of_products",
    "has_cr_card", "is_active_member", "estimated_salary",
    "geography_Germany", "geography_Spain", "gender_Male"
]

# Feature importance values from the trained model (for the results chart)
FEATURE_IMPORTANCE = {
    "age"                : 0.3305,
    "num_of_products"    : 0.2330,
    "balance"            : 0.1228,
    "is_active_member"   : 0.1021,
    "estimated_salary"   : 0.0667,
    "geography_Germany"  : 0.0571,
    "credit_score"       : 0.0500,
    "tenure"             : 0.0170,
    "gender_Male"        : 0.0131,
    "geography_Spain"    : 0.0050,
    "has_cr_card"        : 0.0026,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_input_dataframe(
    credit_score, age, tenure, balance, num_of_products,
    has_cr_card, is_active_member, estimated_salary,
    geography, gender
):
    """
    Convert raw user inputs into the exact 11-column DataFrame
    the model expects — one-hot encoded and in the correct column order.
    """
    row = {
        "credit_score"      : credit_score,
        "age"               : age,
        "tenure"            : tenure,
        "balance"           : balance,
        "num_of_products"   : num_of_products,
        "has_cr_card"       : int(has_cr_card),
        "is_active_member"  : int(is_active_member),
        "estimated_salary"  : estimated_salary,
        # One-hot encoded geography (France = both 0; Germany = Germany 1; Spain = Spain 1)
        "geography_Germany" : int(geography == "Germany"),
        "geography_Spain"   : int(geography == "Spain"),
        # One-hot encoded gender (Female = 0; Male = 1)
        "gender_Male"       : int(gender == "Male"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


def run_prediction(input_df):
    """
    Scale the continuous columns and return:
      - prediction  : 0 (retained) or 1 (churned)
      - probability : float — probability the customer will churn
    """
    df_scaled = input_df.copy()
    df_scaled[COLS_TO_SCALE] = scaler.transform(df_scaled[COLS_TO_SCALE])
    prediction  = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    return int(prediction), float(probability)


def get_risk_tier(probability):
    """Classify the churn probability into a named risk tier."""
    if probability >= 0.60:
        return "🔴 HIGH RISK",   "#EF4444"
    elif probability >= 0.30:
        return "🟠 MEDIUM RISK", "#F97316"
    else:
        return "🟢 LOW RISK",    "#22C55E"


def get_top_risk_drivers(input_df):
    """
    Return the top 3 model features for this customer that are associated
    with elevated churn risk, based on the customer's own input values
    cross-referenced with known churn signal thresholds from EDA.
    """
    drivers = []
    row = input_df.iloc[0]

    if row["age"] >= 40:
        drivers.append(("Age", f"Age {int(row['age'])} is in a high-risk band (churn accelerates above 40)"))
    if row["num_of_products"] >= 3:
        drivers.append(("Products Held", f"{int(row['num_of_products'])} products — churn rate at this level is 82–100%"))
    if row["num_of_products"] == 1:
        drivers.append(("Products Held", "Only 1 product — single-product customers churn at 27.7%"))
    if row["is_active_member"] == 0:
        drivers.append(("Activity Status", "Inactive members churn at 26.9% vs 14.3% for active members"))
    if row["balance"] > 90000:
        drivers.append(("Account Balance", f"Balance £{row['balance']:,.0f} — high-balance customers churn at 24.1%"))
    if row["geography_Germany"] == 1:
        drivers.append(("Geography", "German customers churn at 32.4% — double the rate of France and Spain"))
    if row["gender_Male"] == 0:
        drivers.append(("Gender", "Female customers churn at 25.1% vs 16.5% for male customers"))

    # Return top 3 by FEATURE_IMPORTANCE weight of the triggering feature
    priority_map = {
        "Age": FEATURE_IMPORTANCE["age"],
        "Products Held": FEATURE_IMPORTANCE["num_of_products"],
        "Activity Status": FEATURE_IMPORTANCE["is_active_member"],
        "Account Balance": FEATURE_IMPORTANCE["balance"],
        "Geography": FEATURE_IMPORTANCE["geography_Germany"],
        "Gender": FEATURE_IMPORTANCE["gender_Male"],
    }
    drivers.sort(key=lambda x: priority_map.get(x[0], 0), reverse=True)
    return drivers[:3]


def plot_gauge(probability):
    """Draw a simple semicircular gauge chart for the churn probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    # Background arc (full semicircle)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#1E293B", linewidth=18, solid_capstyle="round")

    # Filled arc up to the probability value
    if probability > 0:
        fill_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
        colour = "#EF4444" if probability >= 0.60 else ("#F97316" if probability >= 0.30 else "#22C55E")
        ax.plot(np.cos(fill_theta), np.sin(fill_theta),
                color=colour, linewidth=18, solid_capstyle="round")

    # Central probability label
    ax.text(0, 0.10, f"{probability:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white")
    ax.text(0, -0.25, "Churn Probability", ha="center", va="center",
            fontsize=9, color="#94A3B8")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def plot_feature_importance():
    """Horizontal bar chart of all 11 feature importances."""
    features = list(FEATURE_IMPORTANCE.keys())
    values   = list(FEATURE_IMPORTANCE.values())
    colours  = ["#EF4444" if v >= 0.10 else "#3B82F6" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    bars = ax.barh(features[::-1], values[::-1], color=colours[::-1], height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8.5, color="white")

    ax.set_xlabel("Relative Importance", color="#94A3B8", fontsize=9)
    ax.set_title("Feature Importances — Gradient Boosting",
                 color="white", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="#CBD5E1", labelsize=8.5)
    ax.spines[:].set_color("#334155")
    ax.set_xlim(0, max(values) * 1.25)

    red_patch  = mpatches.Patch(color="#EF4444", label="High importance (≥10%)")
    blue_patch = mpatches.Patch(color="#3B82F6", label="Lower importance (<10%)")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8,
              facecolor="#1E293B", edgecolor="#334155", labelcolor="white",
              loc="lower right")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0F172A; color: #E2E8F0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E293B; }

    /* All input labels */
    label { color: #CBD5E1 !important; font-size: 0.88rem !important; }

    /* Section headers */
    .section-header {
        font-size: 1.05rem; font-weight: 700;
        color: #38BDF8; margin-top: 0.8rem; margin-bottom: 0.4rem;
        border-bottom: 1px solid #334155; padding-bottom: 4px;
    }

    /* Result cards */
    .result-card {
        background: #1E293B; border-radius: 10px;
        padding: 18px 20px; margin-bottom: 12px;
        border-left: 4px solid #38BDF8;
    }
    .risk-high   { border-left-color: #EF4444; }
    .risk-medium { border-left-color: #F97316; }
    .risk-low    { border-left-color: #22C55E; }

    /* Metric tiles */
    .metric-tile {
        background: #1E293B; border-radius: 8px;
        padding: 12px 16px; text-align: center;
        border: 1px solid #334155;
    }
    .metric-tile .value { font-size: 1.5rem; font-weight: 700; color: #38BDF8; }
    .metric-tile .label { font-size: 0.78rem; color: #94A3B8; margin-top: 2px; }

    /* Driver badges */
    .driver-badge {
        background: #1E293B; border-radius: 6px;
        padding: 8px 12px; margin-bottom: 6px;
        border-left: 3px solid #F97316; font-size: 0.85rem;
    }
    .driver-name  { color: #FBBF24; font-weight: 600; }
    .driver-desc  { color: #CBD5E1; }

    /* Footer */
    .footer {
        margin-top: 2rem; padding-top: 1rem;
        border-top: 1px solid #334155;
        text-align: center; color: #64748B; font-size: 0.80rem;
    }
    .footer a { color: #38BDF8; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Customer Profile")
    st.markdown("Fill in the customer's details below, then click **Predict**.")
    st.markdown("---")

    # ── Demographic Information ───────────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Demographic Information</div>',
                unsafe_allow_html=True)

    geography = st.selectbox(
        "Geography (Country)",
        options=["France", "Germany", "Spain"],
        help="The country where the customer is based."
    )
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        help="The customer's gender."
    )
    age = st.slider(
        "Age (years)",
        min_value=18, max_value=92, value=38, step=1,
        help="Customer age. Risk accelerates above 40."
    )

    # ── Account Information ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏛️ Account Information</div>',
                unsafe_allow_html=True)

    tenure = st.slider(
        "Tenure (years with the bank)",
        min_value=0, max_value=10, value=5, step=1,
        help="How many years the customer has been with the bank (0–10)."
    )
    num_of_products = st.selectbox(
        "Number of Products Held",
        options=[1, 2, 3, 4],
        index=0,
        help="Number of bank products the customer holds. 2 = safest. 3 or 4 = extreme churn risk."
    )
    has_cr_card = st.checkbox(
        "Has Credit Card",
        value=True,
        help="Whether the customer holds a credit card with the bank. (Low predictive importance)"
    )
    is_active_member = st.checkbox(
        "Is Active Member",
        value=True,
        help="Whether the customer is currently active. Inactive members churn at nearly 2× the rate."
    )

    # ── Financial Information ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Financial Information</div>',
                unsafe_allow_html=True)

    credit_score = st.slider(
        "Credit Score",
        min_value=350, max_value=850, value=650, step=1,
        help="Customer credit score (350 = very poor, 850 = excellent)."
    )
    balance = st.number_input(
        "Account Balance (£)",
        min_value=0.0, max_value=260000.0, value=76486.0, step=500.0,
        format="%.2f",
        help="Current account balance in £. Leave as 0.00 if the account is dormant."
    )
    estimated_salary = st.number_input(
        "Estimated Annual Salary (£)",
        min_value=0.0, max_value=200000.0, value=100090.0, step=1000.0,
        format="%.2f",
        help="Customer's estimated annual salary in £."
    )

    st.markdown("---")

    predict_btn = st.button("🔍  Predict Churn Risk", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Bank Customer Churn Prediction")
st.markdown(
    "An end-to-end machine learning application that predicts whether a bank customer "
    "will churn, using a **Tuned Gradient Boosting Classifier** trained on 10,000 customer records."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE METRICS (always visible at the top)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Model Performance — Tuned Gradient Boosting")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown('<div class="metric-tile"><div class="value">87.1%</div><div class="label">Accuracy</div></div>',
                unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-tile"><div class="value">79.5%</div><div class="label">Precision</div></div>',
                unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-tile"><div class="value">49.4%</div><div class="label">Recall</div></div>',
                unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-tile"><div class="value">0.609</div><div class="label">F1-Score</div></div>',
                unsafe_allow_html=True)
with m5:
    st.markdown('<div class="metric-tile"><div class="value">0.868</div><div class="label">ROC-AUC</div></div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:

    # Build input and run prediction
    input_df    = build_input_dataframe(
        credit_score, age, tenure, balance, num_of_products,
        has_cr_card, is_active_member, estimated_salary,
        geography, gender
    )
    prediction, probability = run_prediction(input_df)
    risk_label, risk_colour = get_risk_tier(probability)
    top_drivers             = get_top_risk_drivers(input_df)

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    # ── Result summary ────────────────────────────────────────────────────────
    col_gauge, col_verdict = st.columns([1, 2], gap="large")

    with col_gauge:
        st.pyplot(plot_gauge(probability), use_container_width=True)

    with col_verdict:
        if prediction == 1:
            st.markdown(
                f'<div class="result-card risk-high">'
                f'<div style="font-size:2rem; font-weight:800; color:#EF4444;">⚠️ CHURN RISK</div>'
                f'<div style="font-size:1rem; color:#CBD5E1; margin-top:6px;">'
                f'This customer is predicted to <strong>leave the bank</strong>.</div>'
                f'<div style="margin-top:10px; font-size:0.9rem; color:#94A3B8;">'
                f'Risk Tier: <strong style="color:#EF4444;">{risk_label}</strong> &nbsp;|&nbsp; '
                f'Probability: <strong style="color:#EF4444;">{probability:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card risk-low">'
                f'<div style="font-size:2rem; font-weight:800; color:#22C55E;">✅ RETAINED</div>'
                f'<div style="font-size:1rem; color:#CBD5E1; margin-top:6px;">'
                f'This customer is predicted to <strong>stay with the bank</strong>.</div>'
                f'<div style="margin-top:10px; font-size:0.9rem; color:#94A3B8;">'
                f'Risk Tier: <strong style="color:#22C55E;">{risk_label}</strong> &nbsp;|&nbsp; '
                f'Probability: <strong style="color:#22C55E;">{probability:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: drivers + customer summary ─────────────────────────
    col_drivers, col_summary = st.columns(2, gap="large")

    with col_drivers:
        st.markdown("#### 🔍 Top Risk Drivers for This Customer")
        st.caption("Based on EDA findings and this customer's specific input values.")
        if top_drivers:
            for name, desc in top_drivers:
                st.markdown(
                    f'<div class="driver-badge">'
                    f'<span class="driver-name">{name}</span><br>'
                    f'<span class="driver-desc">{desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="driver-badge">'
                '<span class="driver-name">No elevated risk signals detected</span><br>'
                '<span class="driver-desc">This customer\'s profile does not trigger any known high-risk patterns.</span>'
                '</div>',
                unsafe_allow_html=True
            )

    with col_summary:
        st.markdown("#### 📋 Customer Profile Summary")
        summary_data = {
            "Feature"  : ["Geography", "Gender", "Age", "Credit Score",
                           "Tenure", "Balance", "Products", "Credit Card",
                           "Active Member", "Salary"],
            "Value"    : [
                geography, gender, f"{age} years", str(credit_score),
                f"{tenure} years", f"£{balance:,.2f}", str(num_of_products),
                "Yes" if has_cr_card else "No",
                "Yes" if is_active_member else "No",
                f"£{estimated_salary:,.2f}"
            ]
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )

    # ── Retention recommendation ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Recommended Retention Action")

    if probability >= 0.60:
        st.error(
            "**Immediate Action Required.** This customer is at critical churn risk. "
            "Assign a dedicated relationship manager for a same-day personal call. "
            "If the balance exceeds £90,000, escalate to a senior manager."
        )
    elif probability >= 0.30:
        st.warning(
            "**Proactive Outreach Recommended.** This customer shows elevated churn signals. "
            "Trigger a personalised re-engagement campaign within the next 7 days — "
            "consider a product upgrade offer, loyalty reward, or account health review."
        )
    else:
        st.success(
            "**Standard Monitoring.** This customer's churn risk is low. "
            "Continue regular engagement touchpoints and monitor for any changes in "
            "activity status or product holdings."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE SECTION (always visible below the fold)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 What Drives Churn? — Feature Importance Analysis")
st.caption(
    "These are the 11 features the model uses to make predictions, "
    "ranked by how much each one reduces the model's prediction error across all 200 boosting trees."
)

col_chart, col_insight = st.columns([1.4, 1], gap="large")

with col_chart:
    st.pyplot(plot_feature_importance(), use_container_width=True)

with col_insight:
    st.markdown("**Key Insights from EDA & Model:**")
    insights = [
        ("🔴", "Age (33%)",          "Churn accelerates sharply above 40. The 51–60 band churns at 56%."),
        ("🔴", "Products (23%)",     "U-shaped: 2 products = 7.6% churn. 3 products = 83%. 4 products = 100%."),
        ("🔴", "Balance (12%)",      "Counterintuitive — high-balance customers are the ones leaving."),
        ("🟠", "Activity (10%)",     "Inactive members churn at nearly 2× the rate of active ones."),
        ("🟡", "Germany (6%)",       "German customers churn at 32% — double France and Spain."),
        ("🟢", "Credit Card (0.3%)", "Effectively zero — card ownership has no link to churn."),
    ]
    for icon, label, desc in insights:
        st.markdown(
            f'<div style="background:#1E293B; border-radius:6px; padding:8px 12px; '
            f'margin-bottom:6px; border-left:3px solid #334155;">'
            f'<span style="color:#FBBF24; font-weight:600;">{icon} {label}</span><br>'
            f'<span style="color:#CBD5E1; font-size:0.82rem;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT THIS PROJECT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧠 About This Project")

col_about, col_pipeline = st.columns(2, gap="large")

with col_about:
    st.markdown("""
**Dataset:** 10,000 bank customer records  
**Target:** Binary — Will this customer churn? (Yes / No)  
**Class Imbalance:** 79.6% No Churn : 20.4% Churned (3.91:1)  
**Best Model:** Tuned Gradient Boosting Classifier  

**Model Hyperparameters (from GridSearchCV):**
- `learning_rate = 0.05`
- `max_depth = 5`
- `min_samples_split = 5`
- `n_estimators = 200`

**Cross-Validation (5-Fold):** Mean F1 = 0.5906 ± 0.0211
    """)

with col_pipeline:
    st.markdown("""
**Project Pipeline:**

`1. Data Exploration` → Shape, types, descriptive stats  
`2. Data Cleaning` → Drop identifiers, snake_case, missing values  
`3. EDA` → Univariate, Bivariate, Multivariate analysis  
`4. Preprocessing` → One-Hot Encoding, train-test split, StandardScaler  
`5. Modelling` → Logistic Regression vs Gradient Boosting  
`6. Tuning` → GridSearchCV (16 combos × 5 folds = 80 fits)  
`7. Evaluation` → Confusion matrix, ROC, Precision-Recall curves  
`8. Deployment` → Saved model + scaler → this Streamlit app  
    """)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Built by <strong>Adewale Samson Adeagbo</strong> &nbsp;|&nbsp;
        Data Scientist · Data Analyst · ML Engineer · Lagos, Nigeria
        <br><br>
        <a href="https://github.com/cssadewale/bank-customer-churn-prediction" target="_blank">
            🐙 GitHub Repository
        </a>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <a href="https://www.linkedin.com/in/adewalesamsonadeagbo" target="_blank">
            💼 LinkedIn Profile
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
