'''
This code will predict the new results based on the previous save .pkl model file
it will use the demo file ../data/raw/demo_predict.data
'''
import pandas as pd
import numpy as np
import pickle


## load models
encoder = pickle.load(open("../models/ordinal_encoder.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

log_clf = pickle.load(open("../models/logistic_reg.pkl", "rb"))
rf = pickle.load(open("../models/random_forest.pkl", "rb"))
xgb_clf = pickle.load(open("../models/xgboost.pkl", "rb"))



predict_file = "../data/raw/demo_predict.data"
column_file = "../data/raw/census-bureau.columns"

columns = [c.strip() for c in open(column_file).readlines()]
df_new = pd.read_csv(predict_file, header=None, names=columns)

# Replace '?' with NaN
df_new = df_new.replace("?", np.nan)

# Fix types
df_new["detailed industry recode"] = df_new["detailed industry recode"].astype("object")
df_new["detailed occupation recode"] = df_new["detailed occupation recode"].astype("object")

categorical_cols = df_new.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("label")   # may not exist in prediction file

numeric_cols = [
    'age', 'wage per hour', 'capital gains', 'capital losses',
    'dividends from stocks', 'weight', 'num persons worked for employer',
    'own business or self employed', 'veterans benefits',
    'weeks worked in year', 'year'
]

# Convert numeric types
for col in numeric_cols:
    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

# Fill missing values
df_new[categorical_cols] = df_new[categorical_cols].fillna("Unknown")
for col in numeric_cols:
    df_new[col] = df_new[col].fillna(df_new[col].median())

# Drop label if exists
if "label" in df_new.columns:
    df_new = df_new.drop(columns=["label"])



df_new[categorical_cols] = encoder.transform(df_new[categorical_cols])
df_new[numeric_cols] = scaler.transform(df_new[numeric_cols])


# Predict with 3 models


pred_log = log_clf.predict(df_new)
pred_rf  = rf.predict(df_new)
pred_xgb = xgb_clf.predict(df_new)

proba_log = log_clf.predict_proba(df_new)[:, 1]
proba_rf  = rf.predict_proba(df_new)[:, 1]
proba_xgb = xgb_clf.predict_proba(df_new)[:, 1]


# Save output

df_out = pd.DataFrame({
    "logistic_pred": pred_log,
    "rf_pred": pred_rf,
    "xgb_pred": pred_xgb,
    "logistic_proba": proba_log,
    "rf_proba": proba_rf,
    "xgb_proba": proba_xgb
})

df_out.to_csv("../data/raw/prediction_output.csv", index=False)
print("Prediction saved to prediction_output.csv!")
