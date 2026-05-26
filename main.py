"""Main script for time series forecasting."""
import numpy as np
from data_loader import load_and_prepare_data
from feature_engineering import create_features, get_feature_list
from preprocessing import split_data, prepare_xy
from model_training import train_model
from evaluation import evaluate_overall, evaluate_by_store
from visualization import plot_feature_importance, plot_forecast

# Set random seed
np.random.seed(42)

print("=" * 80)
print("TIME SERIES SALES FORECASTING WITH XGBOOST")
print("=" * 80)

# Load data
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
df = load_and_prepare_data()

# Feature engineering
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)
df = create_features(df)

# Split data
print("\n" + "=" * 80)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)
train_final, valid_final, train_data, test_data = split_data(df)

# Get features
features = get_feature_list()

# Prepare data
print("\n" + "=" * 80)
print("PREPARE DATA")
print("=" * 80)
X_train, y_train = prepare_xy(train_data, features)
X_train_final, y_train_final = prepare_xy(train_final, features)
X_valid_final, y_valid_final = prepare_xy(valid_final, features)
X_test, y_test = prepare_xy(test_data, features)
print("Data prepared successfully!")

# Train model
print("\n" + "=" * 80)
print("GRID SEARCH & MODEL TRAINING")
print("=" * 80)
model = train_model(X_train, y_train)

# Train evaluation
print("\n" + "=" * 80)
print("TRAIN METRICS")
print("=" * 80)
train_pred_log = model.predict(X_train_final)
train_pred = np.expm1(train_pred_log)
train_actual = np.expm1(y_train_final)
train_final["predicted_sales"] = train_pred

evaluate_overall(train_actual, train_pred, "TRAIN")
evaluate_by_store(train_final, "TRAIN")

# Validation evaluation
print("\n" + "=" * 80)
print("VALIDATION METRICS")
print("=" * 80)
valid_pred_log = model.predict(X_valid_final)
valid_pred = np.expm1(valid_pred_log)
valid_actual = np.expm1(y_valid_final)
valid_final["predicted_sales"] = valid_pred

evaluate_overall(valid_actual, valid_pred, "VALIDATION")
evaluate_by_store(valid_final, "VALIDATION")

# Test evaluation
print("\n" + "=" * 80)
print("TEST METRICS")
print("=" * 80)
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_data["predicted_sales"] = test_pred

evaluate_overall(y_test, test_pred, "TEST")
evaluate_by_store(test_data, "TEST")

# Visualization
plot_feature_importance(model, features)
plot_forecast(test_data)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
