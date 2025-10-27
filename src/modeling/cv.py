from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold CV with embargo for time-series data.
    Prevents leakage from overlapping label windows.
    """

    def __init__(self, n_splits = 5, embargo_days = 5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, X, y = None, groups = None):
        """
        Generate train/val indices for each fold.
        X must have DatetimeIndex at level 0 (date).
        """
        # Get unique dates and sort them
        unique_dates = X.index.get_level_values(0).unique().sort_values()
        fold_size = len(unique_dates) // self.n_splits

        for i in range(self.n_splits):
            # Define validation period for this fold
            val_start = i * fold_size
            val_end = (i + 1) *fold_size if i < self.n_splits - 1 else len(unique_dates)
            val_dates = unique_dates[val_start : val_end]

            # Embargo: exclude train dates within embargo_days after last val date
            embargo_cutoff = val_dates[-1] + pd.Timedelta(days = self.embargo_days)
            train_dates = unique_dates[(unique_dates < val_dates[0]) | 
                                       (unique_dates > embargo_cutoff)]
            
            # Convert dates to row indices
            train_idx = np.where(X.index.get_level_values(0).isin(train_dates))[0]
            val_idx = np.where(X.index.get_level_values(0).isin(val_dates))[0]

            yield train_idx, val_idx

    def get_n_splits(self, X = None, y = None, groups = None):
        return self.n_splits