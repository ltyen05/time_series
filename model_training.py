"""Model training module."""
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def train_model(X_train, y_train):
    """Train XGBoost model with grid search."""
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=-1
    )

    tss = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tss,
        verbose=2,
        n_jobs=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_
