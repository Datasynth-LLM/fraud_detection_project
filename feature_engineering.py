# scripts/feature_engineering.py

import pandas as pd

def engineer_features(df):
    print("🧪 Starting feature engineering...")

    # Convert columns to lowercase
    df.columns = df.columns.str.lower()

    df["tx_datetime"] = pd.to_datetime(df["tx_datetime"])
    df = df.sort_values(by=["customer_id", "tx_datetime"])

    # Set tx_datetime as index temporarily for time-based rolling
    df.set_index("tx_datetime", inplace=True)

    # Step 1: Rolling mean (7 days) per customer
    print("🔄 Calculating rolling mean (7 days) per customer...")
    df["tx_amount_rolling_mean_7d"] = (
        df.groupby("customer_id")["tx_amount"]
        .apply(lambda x: x.rolling("7D").mean())
    )

    df.reset_index(inplace=True)  # Restore tx_datetime as column

    # Step 2: Difference from rolling mean
    print("➗ Calculating amount difference from rolling mean...")
    df["amount_diff_from_mean"] = df["tx_amount"] - df["tx_amount_rolling_mean_7d"]

    # Step 3: Flag for amount over ₹220
    print("📌 Creating flag for tx_amount > 220...")
    df["amount_over_220"] = (df["tx_amount"] > 220).astype(int)

    # Step 4: Count past frauds in last 100 transactions per terminal
    print("📊 Counting past frauds in last 100 transactions per terminal...")
    if "tx_fraud" in df.columns:
        df["fraud_count_last_100"] = (
            df.groupby("terminal_id")["tx_fraud"]
            .transform(lambda x: x.shift().rolling(100, min_periods=1).sum())
        )
    else:
        df["fraud_count_last_100"] = 0  # Default when no labels available

    # Step 5: Final fill
    print("🧼 Final fillna cleanup...")
    df.fillna(0, inplace=True)

    print("✅ Feature engineering complete.")
    return df
