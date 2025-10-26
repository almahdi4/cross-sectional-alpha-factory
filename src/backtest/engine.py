import pandas as pd
import numpy as np

def weekly_rebalance_dates(index, weekday = 4):
    """
    Extract rebalance dates (Fridays) from DatetimeIndex.
    weekday = 4 means Friday (0 = Monday, 4 = Friday)
    """

    return index[index.weekday == weekday]

def build_sector_neutral_weights(scores, sector_map, top_bottom_pct = 0.15, cap = 0.02):
    """
    Convert model scores to sector-neutral long/short weights.

    Args:
    scores: Serties with MultiIndex (date, ticker) of model predictions
    sector_map: DataFrame with columns [ticker, sector]
    top_bottom_pct: Keep top/bottom 15% of stocks (0.15)
    cap: Maximum weight per position (0.02 = 2%)
    """

    # Rank scores to [-1. 1] per date (cross-sectional ranking)
    ranks = scores.groupby(level = 0).rank(pct=True) * 2 - 1

    # Keep only extreme tails (top 15% and bottom 15%)
    top_threshold = 1 - 2 * top_bottom_pct # 0.70 for top 15%
    bottom_threshold = -1 + 2 * top_bottom_pct # -0.70 for bottom 15%

    # Zero out middle 70% (only keep extreme scores)
    weights = ranks.copy()
    weights[(weights > bottom_threshold) & (weights < top_threshold)] = 0

    # Convert to DataFrame and add sectore information
    weights_df = weights.reset_index()
    weights_df = weights_df.merge(sector_map, on = 'ticker', how = 'left')
    weights_df = weights_df.set_index(['date', 'ticker'])

    # Sector-demean: subtract sector average from each stock's weight
    sector_means = weights_df.groupby(['date', 'sector'])[0].transform('mean')
    weights_df[0] = weights_df[0] - sector_means

    # Cap individual positions at +/- 2%
    weights_df[0] = weights_df[0].clip(-cap, cap)

    # Normalize so total gross exposure = 1 (sum of absolute values = 1)
    gross_exposure = weights_df.groupby('date')[0].transform(lambda x: x.abs().sum())
    weights_df[0] = weights_df[0] / gross_exposure

    return weights_df[0]

def backtest_portfolio(prices, weights, costs_bps = 5):
    """
    Simulate portfolio returns with transaction costs.

    Args:
    prices: Daily prices DataFrame (dates x tickers)
    weights: Series with MultiIndex (date, ticker) of portfolio weights
    costs_bps: One-way transaction cost in basis points (5 bps = 0.05%)

    Returns:
    Dictionary with equity curve, returns, turnover
    """

    # Unstack weights to DataFrame format (dates x tickers)
    weights_df = weights.unstack(fill_value=0)

    # Align weights with prices (reindex to price dates)
    weights_df = weights_df.reindex(prices.index, method = 'ffill').fillna(0)

    # Calculate daily returns for each position
    daily_returns = prices.pct_change()

    # Portfolio returns = weighted sum of stock returns
    portfolio_returns = (weights_df.shift(1) * daily_returns).sum(axis=1)

    # Calculate turnover (sum of absolute weight changes)
    weight_changes = weights_df.diff().abs().sum(axis=1)

    # Apply transaction costs (costs_bps / 10000 because 5 bps = 0.0005)
    transaction_costs = weight_changes * (costs_bps / 10000)

    # Net returns after costs
    net_returns = portfolio_returns - transaction_costs

    # Equity curve (cumulative returns starting at 1.0)
    equity_curve = (1 + net_returns).cumprod()

    return {
        'equity': equity_curve,
        'returns': net_returns,
        'turnover': weight_changes
    }

  