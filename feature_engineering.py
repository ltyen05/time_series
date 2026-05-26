"""Feature engineering module."""
import pandas as pd


def create_features(df):
    """Create all features for the model."""
    df = df.copy()
    
    # TIME FEATURES
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    # LAG FEATURES
    df["sales_lag_1"] = (
        df.groupby("store")["sales"]
        .shift(1)
    )

    df["sales_lag_7"] = (
        df.groupby("store")["sales"]
        .shift(7)
    )

    df["sales_lag_365"] = (
        df.groupby("store")["sales"]
        .shift(365)
    )

    # ROLLING MEAN FEATURES
    df["rolling_mean_7"] = (
        df.groupby("store")["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_mean_14"] = (
        df.groupby("store")["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(14).mean()
        )
    )

    df["rolling_mean_30"] = (
        df.groupby("store")["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(30).mean()
        )
    )

    # DAY-OF-WEEK CYCLICAL FEATURE
    df["rolling_mean_day_of_week_4"] = (
        df.groupby(
            ["store", "day_of_week"]
        )["sales"]
        .transform(
            lambda x:
            x.shift(1).rolling(4).mean()
        )
    )

    # REMOVE MISSING VALUES
    df = df.dropna().reset_index(drop=True)

    print("\nFinal Dataset Shape:", df.shape)
    print(df.head())
    
    return df


def get_feature_list():
    """Return list of features for model."""
    return [
        # Store
        "store",

        # Time Features
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",

        # Lag Features
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_365",

        # Rolling Mean Features
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",

        # Cyclical Weekly Feature
        "rolling_mean_day_of_week_4"
    ]
