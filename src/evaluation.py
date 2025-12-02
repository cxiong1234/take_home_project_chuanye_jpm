'''
In this code, we will evaluate the three models: (1) logistic regression; (2) Random Forest; (3) XGBoost
Read from the previous saved .pkl file and calculate the (a) weighted accuracy (b) Precision (c) Recall (d) F1-score (e) AUC value
Also, we will plot the ROC curve of the three model.
'''

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# Load Raw Data
DATA_FILE = "../data/raw/census-bureau.data"
COLUMN_FILE = "../data/raw/census-bureau.columns"

columns = [c.strip() for c in open(COLUMN_FILE)]
df = pd.read_csv(DATA_FILE, header=None, names=columns)

df = df.replace("?", np.nan)

#Fix Dtypes
df["detailed industry recode"] = df["detailed industry recode"].astype("object")
df["detailed occupation recode"] = df["detailed occupation recode"].astype("object")

categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("label")

numeric_cols = [
    'age', 'wage per hour', 'capital gains', 'capital losses',
    'dividends from stocks', 'weight', 'num persons worked for employer',
    'own business or self employed', 'veterans benefits',
    'weeks worked in year', 'year'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# handle Missing Values

df[categorical_cols] = df[categorical_cols].fillna("Unknown")
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())


# Convert Label

df["label"] = df["label"].apply(lambda x: 1 if str(x).strip() == "50000+." else 0)

y = df["label"]
weights = df["weight"]
X = df.drop(columns=["label"])


# Load Encoder & Scaler .pkl file saved before

encoder = pickle.load(open("../models/ordinal_encoder.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

X[categorical_cols] = encoder.transform(X[categorical_cols])
X[numeric_cols] = scaler.transform(X[numeric_cols])


#  Load Models

log_clf = pickle.load(open("../models/logistic_reg.pkl", "rb"))
rf = pickle.load(open("../models/random_forest.pkl", "rb"))
xgb_clf = pickle.load(open("../models/xgboost.pkl", "rb"))


#  Predict
models = {
    "Logistic Regression": log_clf,
    "Random Forest": rf,
    "XGBoost": xgb_clf
}

results = []

plt.figure(figsize=(8, 6))

for name, model in models.items():

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, y_pred, sample_weight=weights)
    prec = precision_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)
    auc  = roc_auc_score(y, y_proba, sample_weight=weights)

    results.append([name, acc, prec, rec, f1, auc])

    # draw ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

## 
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate", size=16)
plt.ylabel("True Positive Rate", size=16)
plt.title("ROC Curves for 3 Classification Models", size=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("../models/roc_curve.png", dpi=300)
plt.close()

print("ROC curve saved: ../models/roc_curve.png")


# Summary Table
df_results = pd.DataFrame(
    results,
    columns=["Model", "Weighted Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
)

print("\n=== MODEL PERFORMANCE SUMMARY ===\n")
print(df_results.to_string(index=False))

df_results.to_csv("../models/model_evaluation_summary.csv", index=False)
print("\nSaved summary to: ../models/model_evaluation_summary.csv")
