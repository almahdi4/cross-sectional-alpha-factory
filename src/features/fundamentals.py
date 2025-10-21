import pandas as pd
from pathlib import Path

def load_fundamentals_csv(path):
    """
    Load fundamentals CSV if it exists. Returns empty DataFrame if not.
    Expeceted columns: data, ticker, gm, om, rev_ttm_yoy, eps_ttm_yoy, market_cap
    """

    if not path.exists():
        return pd.DataFrame()   # Return empty if no fundamentals file
    
    df = pd.read_csv(path, parse_dates = ['date'])
    return df[['date', 'ticker', 'gm', 'om', 'rev_ttm_yoy', 'eps_ttm_yoy', 'market_cap']]

def align_fundamentals_to_prices(fund_df, daily_index, tickers):
    """
    Forward-fill fundamentals onto daily price grid.
    Fundamentals are reported quarterly but we need them daily.
    """

    if fund_df.empty:
        # No fundamentals: return empty DataFrame with correct structure
        return pd.DataFrame(index=pd.MultiIndex.from_product(
            [daily_index, tickers], names = ['date', 'ticker']))
        
    # Set MultiIndex (date, ticker) and sort
    fund_df = fund_df.set_index(['date', 'ticker']).sort_index()

    # Forward-fill by ticker: each ticker's fundamentals stay constant until next report
    fund_df = fund_df.groupby(level = 'ticker').ffill()

    #Reindex to match daily price grid (will have NaNs for dates before first report)
    return fund_df.reindex(
        pd.MultiIndex.from_product([daily_index, tickers], names = ['date', 'ticker']))


def build_fundamental_features(fund_df):
    """
    Normalize fundamental features using same cross-sectional z-score approach.
    """

    if fund_df.empty:
        return fund_df
    
    # Select numeric columns and apply per-date winsorization + z-score
    numeric_cols = ['gm', 'om', 'rev_ttm_yoy', 'eps_ttm_yoy', 'market_cap']
    for col in numeric_cols:
        if col in fund_df.columns:
            # Per-date winsorization
            lower = fund_df.groupby(level = 'date')[col].transform(lambda x: x.quantile(0.01))
            upper = fund_df.groupby(level = 'date')[col].transform(lambda x: x.quantile(0.99))
            fund_df[col] = fund_df[col].clip(lower = lower, upper = upper)

            # Per-date z-score
            mean = fund_df.groupby(level = 'date')[col].transform('mean')
            std = fund_df.groupby(level = 'date')[col].transform('std')
            fund_df[col] = (fund_df[col] - mean) / (std + 1e-9)

    return fund_df
