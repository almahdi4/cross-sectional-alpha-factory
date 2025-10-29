import numpy as np
import pandas as pd
from pathlib import Path

from src.backtest.engine import (
    weekly_rebalance_dates,
    build_sector_neutral_weights,
    backtest_portfolio,
)
from src.backtest.metrics import (
    summarize_performance,
    plot_equity_curve,
    plot_deciles,
    plot_ic_timeseries,
)

def make_fake_prices(num_days=60, tickers=("AAPL", "MSFT", "GOOGL", "XOM", "JPM")):
    """
    Create a fake daily price history DataFrame (dates x tickers).
    We'll simulate some random walk prices just for testing.
    """
    dates = pd.date_range("2024-01-01", periods=num_days, freq="B")  # business days
    rng = np.random.default_rng(42)

    # start each stock around 100 and simulate daily drift
    prices_data = {}
    for t in tickers:
        steps = rng.normal(loc=0.0005, scale=0.02, size=num_days)  # daily returns ~N(mu, sigma)
        prices_series = 100 * np.cumprod(1 + steps)
        prices_data[t] = prices_series

    prices = pd.DataFrame(prices_data, index=dates)
    return prices


def make_fake_scores(prices):
    """
    Make a fake model score per (date, ticker).
    We'll pretend the model 'likes' tech tickers more.
    Output: Series with MultiIndex (date, ticker).
    """
    # We'll generate some deterministic-looking scores that vary by ticker
    tickers = prices.columns
    dates = prices.index

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    # Simple rule: tech tickers get higher scores, others lower, plus some noise
    sector_hint = {
        "AAPL": 0.8,
        "MSFT": 0.7,
        "GOOGL": 0.75,
        "XOM": -0.4,
        "JPM": 0.1,
    }

    rng = np.random.default_rng(0)
    base_scores = []
    for d in dates:
        for t in tickers:
            score = sector_hint.get(t, 0.0) + rng.normal(0, 0.1)
            base_scores.append(score)

    scores = pd.Series(base_scores, index=idx, name="score")
    return scores


def make_fake_sector_map(prices):
    """
    Build a sector map DataFrame with columns ['ticker', 'sector'].
    This is needed for sector neutral weighting.
    """
    mapping = {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOGL": "Tech",
        "XOM": "Energy",
        "JPM": "Financials",
    }

    sector_map = pd.DataFrame({
        "ticker": list(prices.columns),
        "sector": [mapping[t] for t in prices.columns]
    })
    return sector_map


def test_backtest_pipeline():
    """
    End-to-end smoke test:
    1. Create fake prices.
    2. Create fake model scores (predicted alpha).
    3. Convert scores -> sector-neutral long/short weights.
    4. Simulate portfolio PnL with transaction costs.
    5. Summarize performance & generate plots.
    """

    # 1. Fake historical prices
    prices = make_fake_prices()
    print("prices shape:", prices.shape)

    # 2. Fake model scores per (date, ticker)
    scores = make_fake_scores(prices)
    print("scores shape:", scores.shape)

    # 3. Sector map (ticker -> sector)
    sector_map = make_fake_sector_map(prices)

    # 4. Build portfolio weights from model scores
    weights = build_sector_neutral_weights(
        scores=scores,
        sector_map=sector_map,
        top_bottom_pct=0.15,
        cap=0.02,
    )
    print("weights shape:", weights.shape)
    print("weights sample:\n", weights.head())

    # 5. Run backtest
    result = backtest_portfolio(
        prices=prices,
        weights=weights,
        costs_bps=5,  # include trading costs
    )

    print("backtest keys:", list(result.keys()))
    print("equity curve length:", len(result["equity"]))
    print("final equity:", result["equity"].iloc[-1])

    # 6. Summarize performance
    perf_summary = summarize_performance(result)
    print("\n performance summary:")
    for k, v in perf_summary.items():
        print(f"  {k}: {v}")

    # 7. Generate diagnostic plots to /tmp (or local plots/ dir)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plot_equity_curve(result, plots_dir / "equity_curve.png")

    # For deciles + IC we need realized next-day returns aligned with predictions.
    # We'll just fake a realized return series using next-day % change for each (date, ticker).
    # This simulates "did our score actually line up with next-day winners?"

    daily_returns = prices.pct_change().shift(-1)  # shift(-1) = forward return
    realized_returns = daily_returns.stack()

    plot_deciles(
        predictions=scores,
        realized_returns=realized_returns,
        output_path=plots_dir / "deciles.png",
    )

    plot_ic_timeseries(
        predictions=scores,
        realized_returns=realized_returns,
        output_path=plots_dir / "ic_timeseries.png",
    )

    print("\n Plots saved in 'plots/' directory.")
    print("Backtest smoke test completed.")


if __name__ == "__main__":
    test_backtest_pipeline()
