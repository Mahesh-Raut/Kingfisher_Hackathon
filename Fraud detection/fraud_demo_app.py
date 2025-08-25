import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

# -----------------
# Streamlit Config
# -----------------
st.set_page_config(page_title="Fraud Detection Demo (India)", layout="wide")

st.title("🇮🇳 Fraud Detection Demo — Kingfisher Synthetic Data")
st.markdown("Train a model on your CSVs, visualize patterns, and test live predictions.")

# -----------------
# Sidebar Inputs
# -----------------
st.sidebar.header("1) Load Data")
default_legit = "indian_legit_transactions.csv"
default_fraud = "indian_fraud_transactions.csv"
legit_path = st.sidebar.text_input("Legit CSV path", value=default_legit)
fraud_path = st.sidebar.text_input("Fraud CSV path", value=default_fraud)

# -----------------
# Data Loader
# -----------------
@st.cache_data
def load_and_prepare(legit_path, fraud_path):
    legit_df = pd.read_csv(legit_path)
    fraud_df = pd.read_csv(fraud_path)

    if "is_fraud" not in legit_df.columns:
        legit_df["is_fraud"] = 0
    if "is_fraud" not in fraud_df.columns:
        fraud_df["is_fraud"] = 1

    legit_df.columns = [c.strip().lower() for c in legit_df.columns]
    fraud_df.columns = [c.strip().lower() for c in fraud_df.columns]

    df = pd.concat([legit_df, fraud_df], ignore_index=True)

    # Timestamp feature engineering
    ts_col = None
    for candidate in ["timestamp", "transaction_datetime", "time", "date_time"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"] = df[ts_col].dt.hour
        df["dayofweek"] = df[ts_col].dt.dayofweek
    else:
        df["hour"] = np.nan
        df["dayofweek"] = np.nan

    # Amount column detection
    amount_col = None
    for candidate in ["transaction_amount", "amount", "amt", "value"]:
        if candidate in df.columns:
            amount_col = candidate
            break

    feature_cols = []
    if amount_col: feature_cols.append(amount_col)
    for col in ["merchant_category", "payment_method", "location", "hour", "dayofweek"]:
        if col in df.columns:
            feature_cols.append(col)

    df = df.dropna(subset=[c for c in [amount_col, "merchant_category", "payment_method", "location"] if c in df.columns])

    if "is_fraud" not in df.columns:
        st.error("No 'is_fraud' column found. Please check your CSVs.")
        st.stop()

    if amount_col:
        upper = df[amount_col].quantile(0.999)
        df[amount_col] = np.clip(df[amount_col], 0, upper)

    return df, feature_cols, amount_col

try:
    df, feature_cols, amount_col = load_and_prepare(legit_path, fraud_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.success(f"✅ Loaded dataset with shape: {df.shape}")
st.dataframe(df.head())

# -----------------
# Model Settings
# -----------------
st.sidebar.header("2) Model Settings")
test_size = st.sidebar.slider("Test size fraction", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("RandomForest trees (n_estimators)", 50, 500, 100, 50)
max_depth = st.sidebar.slider("RandomForest max_depth", 5, 50, 20)
threshold = st.sidebar.slider("Fraud threshold (≥)", 0.3, 0.9, 0.5, 0.05)

# -----------------
# Model Training
# -----------------
X, y = df[feature_cols], df["is_fraud"].astype(int)
categorical_cols = [c for c in feature_cols if c not in [amount_col, "hour", "dayofweek"]]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)
model = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
)
pipe = Pipeline([("prep", preprocess), ("clf", model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
pipe.fit(X_train, y_train)

y_probs = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

# -----------------
# Tabs for UI
# -----------------
tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🔍 Feature Insights", "🧪 Live Prediction"])

# -----------------
# Tab 1: Model Performance
# -----------------
with tab1:
    auc = roc_auc_score(y_test, y_probs)
    st.subheader("Model Performance")
    st.metric("ROC-AUC", f"{auc:.3f}")

    cr = classification_report(y_test, y_pred, output_dict=True)
    st.json(cr)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

# -----------------
# Tab 2: Feature Insights
# -----------------
with tab2:
    st.subheader("Feature Insights")
    if amount_col:
        fig3, ax3 = plt.subplots()
        sns.histplot(df, x=amount_col, hue="is_fraud", bins=40, kde=True, ax=ax3)
        ax3.set_title("Transaction Amount Distribution")
        st.pyplot(fig3)

    if "merchant_category" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        top_cats = df["merchant_category"].value_counts().nlargest(10).index
        grp = df[df["merchant_category"].isin(top_cats)].groupby(["is_fraud","merchant_category"]).size().unstack(fill_value=0)
        grp.plot(kind="bar", stacked=True, ax=ax4)
        ax4.set_title("Top Merchant Categories by Class")
        st.pyplot(fig4)

    if "location" in df.columns:
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        top_loc = df["location"].value_counts().nlargest(10).index
        grp2 = df[df["location"].isin(top_loc)].groupby(["is_fraud","location"]).size().unstack(fill_value=0)
        grp2.plot(kind="bar", stacked=True, ax=ax5)
        ax5.set_title("Top Locations by Class")
        st.pyplot(fig5)

# -----------------
# Tab 3: Live Prediction
# -----------------
with tab3:
    st.subheader("Try a Live Prediction")

    with st.form("predict_form"):
        amt_in = st.number_input("Transaction Amount (₹)", min_value=0.0, value=float(df[amount_col].median()) if amount_col else 1000.0, step=100.0)
        cat_in = st.selectbox("Merchant Category", sorted(df["merchant_category"].dropna().unique())) if "merchant_category" in df.columns else None
        pay_in = st.selectbox("Payment Method", sorted(df["payment_method"].dropna().unique())) if "payment_method" in df.columns else None
        loc_in = st.selectbox("Location", sorted(df["location"].dropna().unique())) if "location" in df.columns else None
        hour_in = st.slider("Hour of Day", 0, 23, int(df["hour"].median()) if "hour" in df.columns else 12)
        day_in = st.slider("Day of Week (0=Mon ... 6=Sun)", 0, 6, int(df["dayofweek"].median()) if "dayofweek" in df.columns else 3)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {}
        if amount_col: input_dict[amount_col] = [amt_in]
        if "merchant_category" in df.columns: input_dict["merchant_category"] = [cat_in]
        if "payment_method" in df.columns: input_dict["payment_method"] = [pay_in]
        if "location" in df.columns: input_dict["location"] = [loc_in]
        if "hour" in df.columns: input_dict["hour"] = [hour_in]
        if "dayofweek" in df.columns: input_dict["dayofweek"] = [day_in]

        X_new = pd.DataFrame(input_dict)
        prob = pipe.predict_proba(X_new)[:, 1][0]
        pred = int(prob >= threshold)

        st.metric("Fraud Probability", f"{prob:.2f}")
        if pred == 1:
            st.error("🚨 Prediction: FRAUD")
        else:
            st.success("✅ Prediction: LEGIT")

st.caption("⚡ Tip: Adjust the decision threshold in the sidebar to balance false positives and false negatives.")

