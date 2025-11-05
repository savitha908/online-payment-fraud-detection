import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ======================
# 1. Generate synthetic dataset
# ======================
np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "amount": np.random.randint(100, 10000, n),
    "transaction_type": np.random.choice(["Online", "Offline"], n, p=[0.8, 0.2]),
    "account_age": np.random.randint(0, 20, n),
    "customer_location": np.random.choice(["Hyderabad", "Bengaluru", "Chennai", "Pune"], n),
    "merchant_category": np.random.choice(
        ["Electronics", "Groceries", "Luxury", "Travel", "Utilities", "Clothing", "Food"], n
    ),
})

# Fraud probability (simple pattern)
fraud_prob = (
    (data["transaction_type"] == "Online").astype(int) * 0.2
    + (data["amount"] > 8000).astype(int) * 0.3
    + (data["merchant_category"].isin(["Luxury", "Electronics"])).astype(int) * 0.2
    + np.random.rand(n) * 0.1
)
data["is_fraud"] = (fraud_prob > 0.4).astype(int)

# ======================
# 2. Encode categorical columns
# ======================
cat_cols = ["transaction_type", "customer_location", "merchant_category"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ======================
# 3. Train model
# ======================
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# ======================
# 4. Save model & encoders
# ======================
joblib.dump(model, "fraud_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("✅ Model and encoders saved successfully!")
print(" - fraud_model.pkl")
print(" - label_encoders.pkl")