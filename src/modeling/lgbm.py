from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np

def train_predict_walkforward(X, Y, lookback_days = 756):
    """
    Walk-forward training: for each date, train on past lookback_days and predict.
    Args:
        X: Features DataFrame with MultiIndex (date, ticker)
        Y: Labels Series with MultiIndex (date, ticker)
        lookback_days: Training window size (about 3 years = 756 training days)
    Returns:
        Series of predictions aligned with Y
    """

    # Get unique dates sorted
    dates = X.index.get_level_values(0).unique().sort_values()
    predictions = []

    for t in dates:
        # Define training window: [t - lookback_days, t)
        train_start = t - pd.Timedelta(days = lookback_days)
        train_mask = (X.index.get_level_values(0) >= train_start) & \
                     (X.index.get_level_values(0) < t)
        test_mask = X.index.get_level_values(0) == t

        # Extract train and test data
        X_train = X[train_mask]
        Y_train = Y[train_mask]
        X_test = X[test_mask]

        # Skip if insufficient training data or no test data
        if len(X_train) < 100 or len(X_test) == 0:
            continue

        # Remove any NaN values from training (from rolling windows at start of data)
        valid_train = ~(X_train.isna().any(axis=1) | Y_train.isna())
        X_train = X_train[valid_train]
        Y_train = Y_train[valid_train]

        if len(X_train) < 50:
            continue

        # Train LightGBM model
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1  # Suppress training logs
        )
        model.fit(X_train, Y_train)

        # Predict on test date (handle NaN in test features)
        X_test_clean = X_test.fillna(0) # Fill NaN with 0 for prediction
        pred = model.predict(X_test_clean)

        # Store predictions with original index
        predictions.append(pd.Series(pred, index = X_test.index))

    # Concatenate all predictions
    return pd.concat(predictions) if predictions else pd.Series(dtype = float)