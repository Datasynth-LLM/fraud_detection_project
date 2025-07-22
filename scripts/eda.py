import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_fraud_counts(df, save=False):
    """
    Plot count of fraudulent vs non-fraudulent transactions.
    """
    sns.countplot(data=df, x="TX_FRAUD")
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xticks([0, 1], ["Not Fraud (0)", "Fraud (1)"])
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    if save:
        plt.savefig("outputs/fraud_vs_nonfraud.png")
    plt.show()


def plot_amount_distribution(df, save=False):
    """
    Plot the distribution of transaction amounts.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df["TX_AMOUNT"], bins=100, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.xlim(0, 500)  # Zoom to focus on typical range
    if save:
        plt.savefig("outputs/amount_distribution.png")
    plt.show()


def plot_fraud_over_time(df, save=False):
    """
    Plot number of frauds per day over time.
    """
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    daily_fraud = df.groupby("TX_DATE")["TX_FRAUD"].sum()

    plt.figure(figsize=(14, 6))
    daily_fraud.plot()
    plt.title("Daily Fraud Transactions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Fraudulent Transactions")
    plt.grid(True)
    if save:
        plt.savefig("outputs/fraud_over_time.png")
    plt.tight_layout()
    plt.show()


def plot_fraud_by_scenario(df, save=False):
    """
    Plot fraud counts by scenario label (if TX_FRAUD_SCENARIO column exists).
    """
    if "TX_FRAUD_SCENARIO" not in df.columns:
        print("❌ 'TX_FRAUD_SCENARIO' column not found.")
        return

    sns.countplot(data=df[df["TX_FRAUD"] == 1], x="TX_FRAUD_SCENARIO")
    plt.title("Fraud by Scenario Type")
    plt.xlabel("Fraud Scenario")
    plt.ylabel("Count")
    if save:
        plt.savefig("outputs/fraud_by_scenario.png")
    plt.show()
