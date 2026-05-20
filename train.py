import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("final_professional_loan_dataset.csv")

# -------------------------
# Remove Dependents Column
# -------------------------

if "no_of_dependents" in df.columns:
    df = df.drop("no_of_dependents", axis=1)

# -------------------------
# Feature Engineering
# -------------------------

df["loan_to_income_ratio"] = (
    df["loan_amount"] / (df["income_annum"] + 1)
)

df["asset_to_loan_ratio"] = (
    df["total_assets"] / (df["loan_amount"] + 1)
)

# -------------------------
# Encode Categorical Columns
# -------------------------

label_encoders = {}

categorical_cols = [
    "gender",
    "education",
    "self_employed",
    "loan_status"
]

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

# -------------------------
# Features and Target
# -------------------------

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

# -------------------------
# Split Data
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Accuracy
# -------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "loan_model.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel Saved Successfully!")