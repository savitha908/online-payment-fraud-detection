import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="🛡️ Online Payment Fraud Detection", page_icon="🛡️")

st.title("🛡️ Online Payment Fraud Detection App")
st.write("Enter transaction details below to estimate the probability of fraud.")

# ✅ Paths
MODEL_PATH = "fraud_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

# ✅ Check model existence
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error("❌ Could not load model. Please run `train_model.py` first.")
    st.stop()

# ✅ Load model & encoders
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ✅ Input form
st.header("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("💰 Transaction Amount (₹)", min_value=1, value=1000)
    transaction_type = st.selectbox("📱 Transaction Type", ["Online", "Offline"])
    account_age = st.number_input("👤 Account Age (years)", min_value=0, max_value=30, value=2)

with col2:
    customer_location = st.selectbox("📍 Customer Location", ["Hyderabad", "Bengaluru", "Chennai", "Pune"])
    merchant_category = st.selectbox(
        "🏪 Merchant Category",
        ["Electronics", "Groceries", "Luxury", "Travel", "Utilities", "Clothing", "Food"]
    )

# ✅ Prepare input
input_df = pd.DataFrame({
    "amount": [amount],
    "transaction_type": [transaction_type],
    "account_age": [account_age],
    "customer_location": [customer_location],
    "merchant_category": [merchant_category],
})

# ✅ Encode categorical columns
for col in input_df.select_dtypes(include=["object"]).columns:
    if col in encoders:
        le = encoders[col]
        input_df[col] = input_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    else:
        st.warning(f"⚠️ Missing encoder for {col}.")
        input_df[col] = -1

# ✅ Predict
if st.button("🔍 Predict Fraud Probability"):
    try:
        proba = model.predict_proba(input_df)[0]
        fraud_prob = proba[1] * 100
        legit_prob = proba[0] * 100

        st.success("✅ Prediction complete!")
        st.metric("🚨 Fraud Probability (%)", f"{fraud_prob:.2f}%")
        st.metric("💸 Legitimate Probability (%)", f"{legit_prob:.2f}%")

        if fraud_prob > 50:
            st.error("⚠️ High risk of fraudulent transaction!")
        else:
            st.success("✅ Transaction seems legitimate.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")