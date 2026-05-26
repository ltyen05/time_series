"""Data preprocessing and splitting module."""
import numpy as np


def split_data(df):
    """Split data into train, validation, and test sets."""
    train_data = df[
        df["date"] < "2017-01-01"
    ].copy()

    test_data = df[
        df["date"] >= "2017-01-01"
    ].copy()

    print("Train Shape:", train_data.shape)
    print("Test Shape :", test_data.shape)
    
    # Validation split
    train_final = train_data[
        train_data["date"] < "2016-10-01"
    ].copy()

    valid_final = train_data[
        train_data["date"] >= "2016-10-01"
    ].copy()

    print("Train Final Shape :", train_final.shape)
    print("Validation Shape  :", valid_final.shape)
    
    return train_final, valid_final, train_data, test_data


def prepare_xy(data, features):
    """Prepare X and y from data."""
    X = data[features]
    y = np.log1p(data["sales"])
    return X, y
