import pandas as pd

def engineer_features(df):
    """
    Generate features aligned with 3 fraud scenarios:
    1. Amount over threshold
    2. Terminal-based rolling fraud activity
    3. Customer spending behavior
    """

    # 1. Scenario 1 — Amount over threshold
    df["AMOUNT_OVER_220"] = (df["TX_AMOUNT"] > 220).astype(int)

    # 2. Time-based columns for grouping and rolling
    df["TX_DATE"] = df["TX_DATETIME"].dt.date
    df["TX_DAY"] = df["TX_DATETIME"].dt.floor("D")

    # 3. Scenario 3 — Customer behavior
    df = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).reset_index(drop=True)

    # Rolling mean of TX_AMOUNT for each customer (last 100 transactions)
    df["TX_AMOUNT_ROLLING_MEAN_7D"] = (
        df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .rolling(window=100, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )

    # Difference from rolling mean
    df["AMOUNT_DIFF_FROM_MEAN"] = df["TX_AMOUNT"] - df["TX_AMOUNT_ROLLING_MEAN_7D"]

    # 4. Scenario 2 — Terminal-based fraud activity
    df = df.sort_values(["TERMINAL_ID", "TX_DATETIME"]).reset_index(drop=True)

    # Rolling count of frauds per terminal over last 100 transactions
    df["FRAUD_COUNT_LAST_100"] = (
        df.groupby("TERMINAL_ID")["TX_FRAUD"]
        .rolling(window=100, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )

    return df
