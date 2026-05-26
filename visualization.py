"""Visualization module."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, features):
    """Plot feature importance."""
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(by="Importance", ascending=False)
    )

    print("\nFEATURE IMPORTANCE")
    print("=" * 80)
    print(importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.grid(True)
    plt.show()


def plot_forecast(test_data):
    """Plot forecast visualization for all stores."""
    print("\nFORECAST VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(20, 25)
    )

    axes = axes.flatten()

    for idx, store_id in enumerate(
        sorted(test_data["store"].unique())
    ):
        ax = axes[idx]
        store_df = test_data[
            test_data["store"] == store_id
        ].copy()

        ax.plot(
            store_df["date"],
            store_df["sales"],
            label="Actual Sales"
        )

        ax.plot(
            store_df["date"],
            store_df["predicted_sales"],
            label="Predicted Sales"
        )

        ax.set_title(f"Store {store_id} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
