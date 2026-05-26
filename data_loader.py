"""Data loading and preparation module."""
import pandas as pd


def load_and_prepare_data(csv_path="./data/train.csv"):
    """Load CSV and aggregate sales by date and store."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Tổng sales theo store và ngày
    df = (
        df.groupby(["date", "store"])["sales"]
        .sum()
        .reset_index()
    )

    # Sort dữ liệu
    df = df.sort_values(
        ["date", "store"]
    ).reset_index(drop=True)

    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    return df
