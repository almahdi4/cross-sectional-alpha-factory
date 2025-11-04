import pandas as pd
import numpy as np

def build_calendar_features(earnings_df, daily_index, tickers):
    """
    Build calendar-based features: days since/until earnings, month-end indicator.
    """
    # Initialize empty feature DataFrame with proper MultiIndex
    idx = pd.MultiIndex.from_product([daily_index, tickers], names=['date', 'ticker'])
    features = pd.DataFrame(index=idx)
    
    # Month-end indicator (1 if last 3 trading days of month, else 0)
    month_end_dates = daily_index.to_series().groupby(
        daily_index.to_period('M')
    ).apply(lambda x: x.tail(3).index)
    
    # Flatten the month-end dates
    month_end_dates_flat = []
    for dates in month_end_dates:
        month_end_dates_flat.extend(dates)
    
    features['month_end'] = features.index.get_level_values('date').isin(month_end_dates_flat).astype(int)
    
    # If no earnings data, return just month_end feature
    if earnings_df is None or earnings_df.empty:
        return features
    
    # Placeholder for earnings proximity features when you have data
    return features
