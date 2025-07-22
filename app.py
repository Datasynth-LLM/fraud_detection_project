# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
from io import BytesIO

# Load model
model_path = "models/fraud_rf_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Please run main.py first.")
    st.stop()

model = joblib.load(model_path)

# App title and clock
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Transaction Prediction App")

# ⏰ Real-time clock
clock_placeholder = st.empty()
def update_clock():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clock_placeholder.markdown(f"🕒 **Current Time:** {current_time}")

update_clock()

st.write("Predict if a transaction is **fraudulent** or **legitimate** based on input details.")

# Function to extract features
def extract_features(df):
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["hour"] = df["TX_DATETIME"].dt.hour
    df["day"] = df["TX_DATETIME"].dt.day
    df["month"] = df["TX_DATETIME"].dt.month
    df["weekday"] = df["TX_DATETIME"].dt.weekday
    df["tx_amount"] = df["TX_AMOUNT"]
    return df[["tx_amount", "hour", "day", "month", "weekday"]]

# Sidebar for CSV upload
st.sidebar.header("📂 Upload Transaction CSV (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Batch prediction section
if uploaded_file is not None:
    st.subheader("📊 Batch Transaction Predictions")
    df = pd.read_csv(uploaded_file)

    if 'TX_DATETIME' not in df.columns or 'TX_AMOUNT' not in df.columns:
        st.error("❌ CSV must contain 'TX_DATETIME' and 'TX_AMOUNT' columns.")
    else:
        features = extract_features(df)
        df["Fraud_Probability"] = model.predict_proba(features)[:, 1]
        df["Prediction"] = model.predict(features)

        # Show dataframe
        st.dataframe(df[["TX_DATETIME", "TX_AMOUNT", "Prediction", "Fraud_Probability"]])

        # Plot
        st.write("📈 Fraud Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="Fraud_Probability", bins=20, kde=True, ax=ax, color="crimson")
        ax.set_title("Fraud Probability Histogram")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Transaction Count")
        st.pyplot(fig)

        # 🧾 Download results
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_download,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

# Single prediction section
st.subheader("🔎 Single Transaction Prediction")
form = st.form("prediction_form", clear_on_submit=True)
with form:
    customer_id = st.text_input("Customer ID", value="C001")
    terminal_id = st.text_input("Terminal ID", value="T001")
    tx_amount = st.number_input("Transaction Amount (₹)", min_value=1.0, step=1.0)
    tx_date = st.date_input("Transaction Date")
    tx_time = st.time_input("Transaction Time")
    submit = st.form_submit_button("Predict")

if submit:
    tx_datetime = datetime.datetime.combine(tx_date, tx_time)

    df_input = pd.DataFrame({
        "CUSTOMER_ID": [customer_id],
        "TERMINAL_ID": [terminal_id],
        "TX_AMOUNT": [tx_amount],
        "TX_DATETIME": [pd.to_datetime(tx_datetime)]
    })

    features = extract_features(df_input)
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.markdown("---")

    # 🚨 Alert
    if prob > 0.9:
        st.error("🚨 HIGH RISK: Fraudulent Transaction Detected!")
        st.markdown(f"<h3 style='color:red;'>⚠️ ALERT: Fraud Probability is {prob:.2%}</h3>", unsafe_allow_html=True)
    else:
        st.success(f"✅ Transaction is likely Legitimate. Fraud Probability: {prob:.2%}")

    # Transaction summary
    st.markdown("📝 **Transaction Details:**")
    st.json({
        "Customer ID": customer_id,
        "Terminal ID": terminal_id,
        "Amount (₹)": tx_amount,
        "Timestamp": str(tx_datetime),
        "Prediction": "Fraud" if prediction == 1 else "Legit",
        "Fraud Probability": f"{prob:.2%}"
    })

# Update clock again
time.sleep(1)
update_clock()
