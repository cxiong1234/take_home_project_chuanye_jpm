import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#Read Raw Data


data_file = "../data/raw/census-bureau.data"
column_file = "../data/raw/census-bureau.columns"

# Load columns
columns = [c.strip() for c in open(column_file).readlines()]

# Load data
df = pd.read_csv(data_file, header=None, names=columns)

# Replace '?' with NaN
df = df.replace("?", np.nan)



# Fix dtypes


# These should be categorical even though they look numeric, its the recode value
df["detailed industry recode"] = df["detailed industry recode"].astype("object")
df["detailed occupation recode"] = df["detailed occupation recode"].astype("object")

# Identify types
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("label")    # label is not a feature

numeric_cols = [
    'age', 'wage per hour', 'capital gains', 'capital losses',
    'dividends from stocks', 'weight', 'num persons worked for employer',
    'own business or self employed', 'veterans benefits',
    'weeks worked in year', 'year'
]

# Convert numerics
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')



# Handle missing values


df[categorical_cols] = df[categorical_cols].fillna("Unknown")

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())



# Convert label to 0 / 1


df["label"] = df["label"].apply(
    lambda x: 1 if str(x).strip() == "50000+." else 0
)

y = df["label"]
weights = df["weight"]
X = df.drop(columns=["label"])



#  Encoding and Scaling

# Ordinal Encode categorical columns
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save encoder and scaler
pickle.dump(encoder, open("../models/ordinal_encoder.pkl", "wb"))
pickle.dump(scaler, open("../models/scaler.pkl", "wb"))



# Train-Test Split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Train Logistic Regression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train, y_train, sample_weight=w_train)
pickle.dump(log_clf, open("../models/logistic_reg.pkl", "wb"))



#  Train Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train, sample_weight=w_train)
pickle.dump(rf, open("../models/random_forest.pkl", "wb"))



# Train XGBoost
xgb_clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
xgb_clf.fit(X_train, y_train, sample_weight=w_train)

pickle.dump(xgb_clf, open("../models/xgboost.pkl", "wb"))

print("Training complete! All models and encoders saved.")
