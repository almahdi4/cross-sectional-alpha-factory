import pandas as pd
import numpy as np

def weekly_rebalance_dates(index, weekday = 4):
    """
    Extract rebalance dates (Fridays) from DatetimeIndex.
    weekday = 4 means Friday (0 = Monday, 4 = Friday)
    """

    return index[index.weekday == weekday]

import pandas as pd
import numpy as np

def build_sector_neutral_weights(scores, sector_map, top_bottom_pct=0.15, cap=0.02):
    """
    Convert model scores to sector-neutral long/short weights.

    Args:
        scores: Series with MultiIndex (date, ticker) of model predictions
        sector_map: DataFrame with columns ['ticker', 'sector']
        top_bottom_pct: Keep only top/bottom % of names (e.g. 0.15 = top 15%, bottom 15%)
        cap: Max absolute weight per stock (0.02 = 2%)

    Returns:
        Series with MultiIndex (date, ticker) of portfolio weights
    """

    # 1. Rank scores cross-sectionally each day to [-1, 1]
    #    rank(pct=True) -> [0,1], then *2-1 -> [-1,1]
    ranks = scores.groupby(level=0).rank(pct=True) * 2 - 1

    # 2. Keep only the extreme tails
    #    Example: top_bottom_pct = 0.15 keeps top 15% and bottom 15%
    top_threshold = 1 - 2 * top_bottom_pct      # e.g. 1 - 0.30 = 0.70
    bottom_threshold = -1 + 2 * top_bottom_pct  # e.g. -1 + 0.30 = -0.70

    weights = ranks.copy()
    # zero out the middle ~70%
    middle_mask = (weights > bottom_threshold) & (weights < top_threshold)
    weights[middle_mask] = 0

    # 3. Turn it into a DataFrame and attach sector info
    weights_df = (
        weights
        .rename("weight")        # <-- give the column a real name
        .reset_index()           # columns: ['date', 'ticker', 'weight']
        .merge(sector_map, on="ticker", how="left")  # add 'sector'
        .set_index(["date", "ticker"])
    )
    # now weights_df has columns: ['weight', 'sector']

    # 4. Sector neutralization:
    #    subtract each (date, sector)'s average weight from individual stock weights
    sector_means = (
        weights_df
        .groupby(["date", "sector"])["weight"]
        .transform("mean")
    )
    weights_df["weight"] = weights_df["weight"] - sector_means

    # 5. Cap individual positions at +/- cap (e.g. +/-2%)
    weights_df["weight"] = weights_df["weight"].clip(-cap, cap)

    # 6. Normalize so total gross exposure per date = 1
    #    gross exposure = sum(|weights|) for that date
    gross_exposure = (
        weights_df
        .groupby("date")["weight"]
        .transform(lambda x: x.abs().sum())
    )
    weights_df["weight"] = weights_df["weight"] / (gross_exposure.replace(0, np.nan))

    # 7. Return a clean Series with MultiIndex (date, ticker)
    final_weights = weights_df["weight"]
    return final_weights


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

  