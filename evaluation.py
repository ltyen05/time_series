"""Model evaluation module."""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from tabulate import tabulate


def calculate_metrics(actual, pred):
    """Calculate all evaluation metrics."""
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, pred) * 100
    r2 = r2_score(actual, pred)
    
    return mae, mse, rmse, mape, r2


def evaluate_overall(y_actual, y_pred, dataset_name=""):
    """Evaluate overall metrics."""
    mae, mse, rmse, mape, r2 = calculate_metrics(y_actual, y_pred)
    
    results = [[
        f"{mae:.2f}",
        f"{mse:.2f}",
        f"{rmse:.2f}",
        f"{mape:.2f}%",
        f"{r2:.4f}"
    ]]

    print(f"\nOVERALL {dataset_name} METRICS")
    print("=" * 80)
    print(
        tabulate(
            results,
            headers=["MAE", "MSE", "RMSE", "MAPE", "R2"],
            tablefmt="grid"
        )
    )


def evaluate_by_store(df, dataset_name=""):
    """Evaluate metrics by store."""
    results = []

    for store_id in sorted(df["store"].unique()):
        store_df = df[df["store"] == store_id]
        actual = store_df["sales"]
        pred = store_df["predicted_sales"]

        mae, mse, rmse, mape, r2 = calculate_metrics(actual, pred)

        results.append([
            f"Store {store_id}",
            f"{mae:.2f}",
            f"{mse:.2f}",
            f"{rmse:.2f}",
            f"{mape:.2f}%",
            f"{r2:.4f}"
        ])

    print(f"\nSTORE-WISE {dataset_name} METRICS")
    print("=" * 80)
    print(
        tabulate(
            results,
            headers=["Store", "MAE", "MSE", "RMSE", "MAPE", "R2"],
            tablefmt="grid"
        )
    )
