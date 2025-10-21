import pandas as pd
import yfinance as yf
from pathlib import Path
# v0.1: finished basic yfinance loaders
def load_universe(base_path):
    """
    Load stock universe (tickers + sectors).
    Returns: DataFrame with columns [ticker, sector]
    """

    universe_file = base_path / "data" / "universe.csv"

    if universe_file.exists():
        return pd.read_csv(universe_file)
    
def load_prices_volume(tickers, start_date, end_date):
    """"
    Download daily adjusted close prices and volume for given tickers.
    Returns: (prices_df, volume_df) - both with dates as index, tickers as columns
    """

    # Download data from Yahoo Finance for all tickers at once
    data = yf.download(tickers, start = start_date, end = end_date, auto_adjust = True, progress = False)

    # Extract closing prices and volume into separate DataFrames
    prices = data['Close']
    volume = data['Volume']

    return prices, volume
