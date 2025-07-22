import os
import pandas as pd

def load_transaction_data(folder_path):
    """
    Loads all .pkl transaction files and returns a combined, sorted DataFrame.
    """
    pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])
    all_dfs = []

    for file in pkl_files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_pickle(full_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

    if not all_dfs:
        raise ValueError("No valid .pkl files found or loaded.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["TX_DATETIME"] = pd.to_datetime(df_all["TX_DATETIME"])
    df_all = df_all.sort_values("TX_DATETIME").reset_index(drop=True)

    return df_all
