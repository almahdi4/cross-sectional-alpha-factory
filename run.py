import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.loaders import load_universe, load_prices_volume
from src.features.price import build_price_features, build_labels, xsection_normalize
from src.features.fundamentals import download_fundamentals, align_fundamentals_to_prices, build_fundamental_features
from src.features.calendar_features import build_calendar_features
from src.modeling.lgbm import train_predict_walkforward
from src.backtest.engine import weekly_rebalance_dates, build_sector_neutral_weights, backtest_portfolio
from src.backtest.metrics import summarize_performance, plot_equity_curve, plot_deciles, plot_ic_timeseries

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)  # Create reports directory if it doesn't exist
CONFIG = BASE / "configs" / "base.json"

def main():
    """Main pipeline: data → features → model → backtest → report"""
    print("=" * 60)
    print("ALPHA FACTORY - Cross-Sectional ML Pipeline")
    print("=" * 60)

    # Load configuration
    cfg = json.loads(CONFIG.read_text())
    start = cfg["start_date"]
    end = cfg["end_date"] or datetime.today().strftime("%Y-%m-%d")
    
    print(f"\n[1/7] Loading data from {start} to {end}...")
    
    # Load universe (tickers + sectors)
    univ_df = load_universe(BASE)
    tickers = univ_df['ticker'].tolist()
    print(f"Universe: {len(tickers)} stocks")
    
    # Load prices and volume
    prices, volume = load_prices_volume(tickers, start, end)
    print(f"Prices: {len(prices)} days × {len(tickers)} stocks")
    
    print("\n[2/7] Building price/volume features...")
    feat = build_price_features(prices, volume)
    print(f"Features shape: {feat.shape}")
    
    print("\n[3/7] Building labels (5-day forward excess returns)...")
    labels = build_labels(prices)
    print(f"Labels shape: {labels.shape}")
    
    # Get rebalance dates (Fridays)
    rebal_dates = weekly_rebalance_dates(prices.index, weekday=cfg["rebalance_weekday"])

    print(f"Rebalance dates: {len(rebal_dates)} Fridays")
    
    # Download and integrate fundamental features
    print("\n[4/7] Downloading fundamental features...")
    from src.features.fundamentals import download_fundamentals, align_fundamentals_to_prices, build_fundamental_features
    
    fund = download_fundamentals(tickers, start, end)
    
    if not fund.empty:
        fund = align_fundamentals_to_prices(fund, prices.index, tickers)
        fund = build_fundamental_features(fund)
        
        # Merge with existing features
        feat = pd.concat([feat, fund], axis=1)
        print(f"Added {len(fund.columns)} fundamental features: {list(fund.columns)}")
    else:
        print(f"No fundamental data available, continuing with price features only")

    
    # Calendar features
    print("\n[5/7] Building calendar features...")
    cal = build_calendar_features(None, prices.index, tickers)
    feat = pd.concat([feat, cal], axis=1)  # CHANGED: use concat
    print(f"Added calendar features (month-end)")
    
    # Cross-sectional normalization (winsorize + z-score)
    print("\n[6/7] Normalizing features (cross-sectional z-score)...")
    feat = xsection_normalize(feat)
    
    # Align features and labels to rebalance dates only
    print(f"\n Aligning to rebalance dates...")
    feat_rebal = feat.loc[feat.index.get_level_values(0).isin(rebal_dates)].copy()
    labels_rebal = labels.loc[labels.index.get_level_values(0).isin(rebal_dates)].copy()

    # Make sure both have the same index (inner join on index)
    common_index = feat_rebal.index.intersection(labels_rebal.index)
    feat_rebal = feat_rebal.loc[common_index]
    labels_rebal = labels_rebal.loc[common_index]

    print(f"Features aligned to {len(feat_rebal.index.get_level_values(0).unique())} rebalance dates")

    print("\n[7/7] Training walk-forward model and generating predictions...")
    # Remove rows with NaN in features or labels
    valid = ~(feat_rebal.isna().any(axis=1) | labels_rebal.isna())
    X = feat_rebal[valid]
    Y = labels_rebal[valid]

    print(f"Clean data: {len(X)} observations")

    
    predictions = train_predict_walkforward(X, Y, lookback_days=cfg["lookback_days"])
    print(f"Generated {len(predictions)} predictions")

        
    print("\n[Backtest] Building sector-neutral portfolio...")
    weights = build_sector_neutral_weights(
        predictions, 
        univ_df[['ticker', 'sector']], 
        top_bottom_pct=cfg["top_bottom_pct"],
        cap=cfg["weight_cap"]
    )
    print(f"Portfolio weights: top/bottom {cfg['top_bottom_pct']*100}%, capped at {cfg['weight_cap']*100}%")

    print("\n[Backtest] Simulating trades with transaction costs...")
    backtest_result = backtest_portfolio(prices, weights, costs_bps=cfg["one_way_cost_bps"])
    print(f"Backtest complete")
    
    print("\n[Metrics] Calculating performance statistics...")
    stats = summarize_performance(backtest_result)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Annual Return:    {stats['ann_return']*100:>8.2f}%")


    print(f"  Annual Volatility: {stats['ann_vol']*100:>8.2f}%")
    print(f"  Sharpe Ratio:     {stats['sharpe']:>8.2f}")
    print(f"  Max Drawdown:     {stats['max_drawdown']*100:>8.2f}%")
    print(f"  Avg Turnover:     {stats['avg_turnover']*100:>8.2f}%")
    print(f"  Total Return:     {stats['total_return']*100:>8.2f}%")
    print("=" * 60)
    
    # Save metrics to JSON
    with open(REPORTS / "summary.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved summary.json")
    
    # Generate plots
    print(f"\n[Plots] Generating visualizations...")
    plot_equity_curve(backtest_result, REPORTS / "equity_curve.png")
    print(f"equity_curve.png")
    
    plot_deciles(predictions, Y, REPORTS / "deciles.png")
    print(f"deciles.png")
    
    plot_ic_timeseries(predictions, Y, REPORTS / "ic_timeseries.png")
    print(f"ic_timeseries.png")
    
    # SHAP Explainability Analysis
    print("\n[Explain] Generating SHAP feature importance...")
    from src.explain.shap_analysis import generate_shap_summary, generate_feature_importance_chart
    from src.explain.ablation import run_ablation_study
    
    # Use last successful model from walk-forward (train on final window)
    print("  Training final model on most recent data for SHAP...")
    final_train_size = min(5000, len(X))
    final_model = LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    final_model.fit(X.iloc[-final_train_size:], Y.iloc[-final_train_size:])
    
    # Generate SHAP plots (use last 1000 samples)
    sample_size = min(1000, len(X))
    X_sample = X.iloc[-sample_size:]
    
    generate_shap_summary(
        final_model, X_sample, X.columns.tolist(),
        REPORTS / "shap_summary.png"
    )
    
    generate_feature_importance_chart(
        final_model, X.columns.tolist(),
        REPORTS / "feature_importance.png"
    )
    
    # Run ablation study
    base_metrics = summarize_performance(backtest_result)
    ablation_results = run_ablation_study(
        X.iloc[-2000:],  # Use last 2000 samples for speed
        Y.iloc[-2000:],
        prices,
        univ_df[['ticker', 'sector']],
        base_metrics['sharpe'],
        REPORTS / "ablation_results.csv"
    )
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETE - Check reports/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()





