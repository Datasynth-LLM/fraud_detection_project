import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Dummy model
model = RandomForestClassifier()
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_rf_model.pkl")
print("✅ Dummy model saved.")
