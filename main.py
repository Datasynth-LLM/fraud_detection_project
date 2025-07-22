# main.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Import from scripts
from scripts.load_data import load_transaction_data

# Step 1: Load the dataset
data_path = r"C:\Users\avish\Documents\fraud_detection_project\data"
df = load_transaction_data(data_path)
print("✅ Loaded data:", df.shape)

# Step 2: Feature Engineering — use only features used in app.py
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
df["hour"] = df["TX_DATETIME"].dt.hour
df["day"] = df["TX_DATETIME"].dt.day
df["month"] = df["TX_DATETIME"].dt.month
df["weekday"] = df["TX_DATETIME"].dt.weekday
df["tx_amount"] = df["TX_AMOUNT"]

# Step 3: Save a fraud vs non-fraud count plot
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(6, 4))
sns.countplot(x="TX_FRAUD", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Is Fraud?")
plt.ylabel("Count")
plt.savefig("outputs/fraud_vs_nonfraud.png")
plt.close()

# Step 4: Prepare features and target
feature_cols = ["tx_amount", "hour", "day", "month", "weekday"]
target_col = "TX_FRAUD"

X = df[feature_cols]
y = df[target_col]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("✅ Model trained.")

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Step 9: Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_rf_model.pkl")
print("✅ Model saved to models/fraud_rf_model.pkl")
