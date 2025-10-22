import pandas as pd
import numpy as np

def build_calendar_features(earnings_df, daily_index, tickers):
    """
    Build calendar-based features: days since/until earnings, month-end indicator.
    """

    # Initialize empty feature DataFrame
    idx = pd.MultiIndex.from_product([daily_index, tickers], names = ['date', 'ticker'])
    features = pd.DataFrame(index = idx)

    # Month-end indicator (1 if last 3 trading days of month, else 0)
    features['month_end'] = features.index.get_level_values('date').isin(
        daily_index.to_series().groupby(daily_index.to_period('M')).tail(3).index
        ).astype(int)
    
    # If no earnings data, return just month_end feature
    if earnings_df is None or earnings_df.empty:
        return features
    
    # Calculate days since/until earnings (placeholder - requires proper earnings dates)
    # This would need earnings_df with columns: ticker, last_earn_date, next_earn_date
    # For now, just return month_end feature
    return features