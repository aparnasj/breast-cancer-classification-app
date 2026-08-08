"""
train_models.py
----------------
Loads the Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository),
trains 5 classification models, evaluates them on a held-out test split,
and saves:
  - trained models (model/*.pkl)
  - the fitted StandardScaler (model/scaler.pkl)
  - the held-out test data as CSV (test_data.csv) for the Streamlit app
  - a metrics comparison table (model/metrics_comparison.csv)

Dataset source: UCI Machine Learning Repository / scikit-learn built-in loader
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 569 instances, 30 numeric features, binary target (malignant / benign)
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame  # 569 rows x 30 features + target
X = df.drop(columns=["target"])
y = df["target"]  # 0 = malignant, 1 = benign

print(f"Dataset shape: {df.shape}")
print(f"Feature count: {X.shape[1]} (min required: 12)")
print(f"Instance count: {X.shape[0]} (min required: 500)")
print(f"Class balance:\n{y.value_counts()}")

# ---------------------------------------------------------------
# Step 2: Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the held-out test set as the "test_data.csv" required by the assignment
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with shape {test_df.shape}")

