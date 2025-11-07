import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from src.backtest.metrics import summarize_performance
from src.backtest.engine import build_sector_neutral_weights, backtest_portfolio

def run_ablation_study(X, Y, prices, sector_map, base_sharpe, output_path):
    """
    Remove each feature one at a time and measure impact on Sharpe ratio.
    Shows which features are critical vs redundant.
    
    Args:
        X: Full feature DataFrame
        Y: Labels
        prices: Daily prices for backtesting
        sector_map: Sector information for portfolio construction
        base_sharpe: Sharpe ratio with all features (baseline)
        output_path: Where to save results
    """
    print(f"\n[Ablation] Running ablation study on {len(X.columns)} features.")
    
    results = []
    feature_names = X.columns.tolist()
    
    # Test each feature removal
    for i, feature in enumerate(feature_names):
        print(f"  [{i+1}/{len(feature_names)}] Testing without '{feature}'...")
        
        # Remove this feature
        X_ablated = X.drop(columns=[feature])
        
        # Train model with remaining features
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        
        # Use last 1000 samples for speed (not full walk-forward)
        train_size = min(1000, len(X_ablated) - 100)
        model.fit(X_ablated.iloc[-train_size:], Y.iloc[-train_size:])
        
        # Predict on last 100 samples
        predictions = pd.Series(
            model.predict(X_ablated.iloc[-100:]),
            index=Y.iloc[-100:].index
        )
        
        # Build portfolio and backtest
        try:
            weights = build_sector_neutral_weights(predictions, sector_map)
            # Get prices for last 100 dates
            test_dates = predictions.index.get_level_values(0).unique()
            prices_subset = prices.loc[test_dates]
            
            backtest = backtest_portfolio(prices_subset, weights)
            metrics = summarize_performance(backtest)
            sharpe = metrics['sharpe']
        except:
            sharpe = 0.0  # If backtest fails, mark as 0
        
        # Calculate impact
        impact = base_sharpe - sharpe  # Positive = feature was helpful
        
        results.append({
            'feature': feature,
            'sharpe_without': sharpe,
            'impact': impact,
            'critical': impact > 0.05  # Feature is "critical" if removing it drops Sharpe >0.05
        })
        
        print(f"Sharpe without {feature}: {sharpe:.3f} (impact: {impact:+.3f})")
    
    # Save results
    results_df = pd.DataFrame(results).sort_values('impact', ascending=False)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n Ablation study complete: {output_path}")
    print(f"\n  Critical features (impact > 0.05):")
    for _, row in results_df[results_df['critical']].iterrows():
        print(f"    - {row['feature']}: impact {row['impact']:+.3f}")
    
    return results_df
