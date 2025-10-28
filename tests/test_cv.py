import numpy as np
import pandas as pd

from src.modeling.cv import PurgedKFold

def make_fake_data(num_days=30, tickers=("AAPL", "MSFT", "GOOGL")):
    """
    Build a fake (date, ticker)-indexed DataFrame to simulate features.
    Each (date, ticker) row is one sample.
    """
    dates = pd.date_range("2024-01-01", periods=num_days, freq="B")  # Business days
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    # Create some dummy feature values just so X is not empty
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(len(idx)),
            "feature2": np.random.randn(len(idx)),
        },
        index=idx,
    )

    return X

def test_purged_kfold_basic():
    """
    - Splits data into folds with embargo.
    - Checks:
        1. We get the right number of folds.
        2. Train and val sets don't overlap.
        3. Embargo is being applied (no near-future leakage).
    """
    # Make fake panel data
    X = make_fake_data(num_days=30)

    # Set up CV
    cv = PurgedKFold(n_splits=3, embargo_days=2)

    all_folds = list(cv.split(X))

    # 1. Number of folds
    assert len(all_folds) == 3, f"Expected 3 folds, got {len(all_folds)}"

    for fold_idx, (train_idx, val_idx) in enumerate(all_folds):
        # Grab the actual dates from the row indices
        train_dates = X.index.get_level_values("date")[train_idx]
        val_dates = X.index.get_level_values("date")[val_idx]

        # 2. No overlap in rows
        assert len(set(train_idx).intersection(set(val_idx))) == 0, \
            f"Fold {fold_idx}: train/val overlap in rows!"

        # 3. Embargo check:
        # The last validation date
        last_val_date = val_dates.max()

        # No training dates should be within embargo_days AFTER last_val_date
        too_close_mask = (
            (train_dates > last_val_date) &
            (train_dates <= last_val_date + pd.Timedelta(days=cv.embargo_days))
        )
        assert not too_close_mask.any(), \
            f"Fold {fold_idx}: training data violates embargo after {last_val_date}"

    print("PurgedKFold basic behavior OK.")

if __name__ == "__main__":
    test_purged_kfold_basic()
    print("All tests passed.")
