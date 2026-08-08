# ML Assignment 2 — Breast Cancer Classification

---

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple
classification models on a real-world dataset, and to expose the results
through an interactive Streamlit web application. The task is a **binary
classification** problem: given a set of diagnostic measurements computed
from a digitized image of a breast mass, predict whether the mass is
**malignant** or **benign**.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also available via
  `sklearn.datasets.load_breast_cancer`, which mirrors the same UCI data)
- **Instances:** 569 (≥ 500 required)
- **Features:** 30 numeric features (≥ 12 required) — mean, standard error,
  and "worst" (largest) values of 10 real-valued measurements computed for
  each cell nucleus (e.g., radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class distribution:** 212 malignant, 357 benign
- **Train/test split:** 80/20, stratified by class, `random_state=42`

The held-out 20% test split (114 rows) is saved as `test_data.csv` and is
used both for offline evaluation and as the sample upload file for the
Streamlit app.

## c. GitHub Repository Link

`https://github.com/aparnasj/breast-cancer-classification-app`

Repository contains: `app.py`, `requirements.txt`, `README.md`,
`test_data.csv`, and the `model/` folder with training code and saved
model files.

## d. Models Used

All 5 models were trained on the same 80% training split and evaluated on
the same 20% held-out test split described above. Logistic Regression and
kNN were trained on standardized features (`StandardScaler`); the
tree-based and Naive Bayes models were trained on the raw feature values.

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree               | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                          | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                 | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble)    | 0.9561   | 0.9931 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name             | Observation about model performance |
|----------------------------|--------------------------------------|
| Logistic Regression        | Best performer across almost every metric on this dataset. The classes are near-linearly separable after standardization, which suits a linear decision boundary well, and the high MCC (0.9623) shows the result isn't just from class imbalance. |
| Decision Tree               | Weakest of the five models. A single unpruned tree overfits the training data and generalizes worse than the ensemble/linear alternatives, reflected in the lowest accuracy, AUC, and MCC. |
| kNN                          | Solid performance once features are standardized (distance-based methods are sensitive to feature scale). Performs identically to Random Forest on Accuracy/Precision/Recall/F1 here, but with a lower AUC, indicating its predicted probabilities are less well-calibrated/ranked than the ensemble's. |
| Naive Bayes                 | Reasonable performance despite the (unrealistic) feature-independence assumption, given many of the 30 features are highly correlated (mean/error/worst versions of the same measurement). Its AUC (0.9878) is surprisingly strong even though Accuracy/F1 trail the top models. |
| Random Forest (Ensemble)    | Second-best overall. The ensemble corrects most of the single Decision Tree's overfitting and achieves the highest AUC among the tree-based/probabilistic models, though it doesn't surpass plain Logistic Regression on this particular dataset. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy, AUC, Precision, Recall, F1, and MCC. This dataset's classes are well-separated by a (roughly) linear combination of the standardized features, which favors a linear model over more complex alternatives. |

---

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models, scaler, and test_data.csv
streamlit run app.py
```

## Streamlit App Features

1. **Dataset upload** — upload `test_data.csv` (CSV with the 30 feature
   columns + `target`).
2. **Model selection dropdown** — choose between Logistic Regression,
   Decision Tree, kNN, Naive Bayes, and Random Forest.
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1,
   MCC for the selected model on the uploaded data.
4. **Confusion matrix & classification report** — visual confusion matrix
   heatmap and a full `sklearn` classification report.
5. A checkbox to compare all 5 models side-by-side on the
   uploaded data.
