import seaborn as sns
import matplotlib.pyplot as plt

def plot_fraud_counts(df):
    sns.countplot(data=df, x="TX_FRAUD")
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xticks([0, 1], ["Not Fraud (0)", "Fraud (1)"])
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.show()

def plot_amount_distribution(df):
    sns.histplot(df["TX_AMOUNT"], bins=100, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.xlim(0, 500)
    plt.show()

def plot_fraud_over_time(df):
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    daily_fraud = df.groupby("TX_DATE")["TX_FRAUD"].sum()
    daily_fraud.plot(figsize=(14, 6))
    plt.title("Daily Fraud Transactions Over Time")
    plt.ylabel("Fraud Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fraud_by_scenario(df):
    sns.countplot(data=df[df["TX_FRAUD"] == 1], x="TX_FRAUD_SCENARIO")
    plt.title("Fraud by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Fraud Count")
    plt.show()
