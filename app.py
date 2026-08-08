"""
Streamlit app for ML Assignment 2 
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Breast Cancer Classification Demo", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

SCALED_MODELS = {"Logistic Regression", "kNN"}


@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


st.title("Breast Cancer Classification — Model Demo")
st.markdown(
    """
This app demonstrates 5 classification models trained on the
**Breast Cancer Wisconsin (Diagnostic)** dataset (UCI ML Repository).
Upload the provided `test_data.csv` (or a similarly formatted CSV with a
`target` column) to see predictions and evaluation metrics.
"""
)

# ---------------------------------------------------------------
# a. Dataset upload option (CSV)
# ---------------------------------------------------------------
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with shape {test_df.shape}")
    st.dataframe(test_df.head())

    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column with true labels.")
        st.stop()

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # ---------------------------------------------------------------
    # b. Model selection dropdown
    # ---------------------------------------------------------------
    st.header("2. Select a Model")
    model_choice = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))

    model = load_model(model_choice)

    if model_choice in SCALED_MODELS:
        scaler = load_scaler()
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    try:
        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]
    except Exception as e:
        st.error(f"Prediction failed — check that the uploaded CSV has the same "
                  f"30 feature columns as the training data. Error: {e}")
        st.stop()

    # ---------------------------------------------------------------
    # c. Display of evaluation metrics
    # ---------------------------------------------------------------
    st.header("3. Evaluation Metrics")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred),
    }
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].round(4)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(metrics_df, hide_index=True)
    with col2:
        st.bar_chart(metrics_df.set_index("Metric"))

    # ---------------------------------------------------------------
    # d. Confusion matrix / classification report
    # ---------------------------------------------------------------
    st.header("4. Confusion Matrix & Classification Report")
    cm_col, report_col = st.columns(2)

    with cm_col:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Malignant (0)", "Benign (1)"],
                    yticklabels=["Malignant (0)", "Benign (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_choice}")
        st.pyplot(fig)

    with report_col:
        report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"])
        st.text("Classification Report")
        st.code(report)

    # ---------------------------------------------------------------
    # Optional: compare all models on the uploaded data
    # ---------------------------------------------------------------
    st.header("5. Compare All Models on Uploaded Data")
    if st.checkbox("Show comparison across all 5 models"):
        rows = []
        for name in MODEL_FILES:
            m = load_model(name)
            if name in SCALED_MODELS:
                s = load_scaler()
                Xi = s.transform(X_test)
            else:
                Xi = X_test
            yp = m.predict(Xi)
            ypr = m.predict_proba(Xi)[:, 1]
            rows.append({
                "ML Model Name": name,
                "Accuracy": round(accuracy_score(y_test, yp), 4),
                "AUC": round(roc_auc_score(y_test, ypr), 4),
                "Precision": round(precision_score(y_test, yp), 4),
                "Recall": round(recall_score(y_test, yp), 4),
                "F1": round(f1_score(y_test, yp), 4),
                "MCC": round(matthews_corrcoef(y_test, yp), 4),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

else:
    st.info("Upload the `test_data.csv` file (included in this repository) to get started.")
