import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize_performance(backtest_result):
    """
    Calculate key performance metrics from backtest results.
    Returns: Dictionary of metrics (Sharpe, return, vol, drawdown, turnover)
    """
    returns = backtest_result['returns']
    equity = backtest_result['equity']
    turnover = backtest_result['turnover']

    # Annualized return and volatility (assuming 252 trading days/year)
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Maximum drawdown: worst peak-to-trough decline
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Average weekly turnover
    avg_turnover = turnover.mean()

    return {
        'ann_return': ann_return, 
        'ann_vol' : sharpe,
        'max_drawdown' : max_drawdown,
        'avg_turnover' : avg_turnover,
        'num_periods' : len(returns),
        'total_return' : equity.iloc[-1] - 1 if len(equity) > 0 else 0
    }

def plot_equity_curve(backtest_result, output_path):
    """
    Plot equity curve and drawdown chart.
    """
    equity = backtest_result['equity']

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    # Create figure with 2 subplots (equity on top, drawdown below)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8), sharex = True)

    # Plot equity curve
    ax1.plot(equity.index, equity.values, linewidth = 2, colour = '#2E86AB')
    ax1.set_ylabel('Equity ($)', fontsize = 12)
    ax1.set_title('Portfolio Equity Curve', fontsize = 14, fontweight = 'bold')
    ax1.grid(True, alpha = 0.3)

    # Plot drawdown
    ax2.fill_between(drawdown.index, 0, drawdown.values, colour = '#A23B72', alpha = 0.7)
    ax2.set_ylabel('Drawdown (%)', fontsize = 12)
    ax2.set_xlabel('Date', fontsize = 12)
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_deciles(predictions, realized_returns, output_path):
    """
    Plot average realized returns by prediction decile.
    Tests if model predictions are monotonic (higher score -> higher returns).
    """
    # Combine predictions and realized returns
    df = pd.DataFrame({
        'pred' : predictions,
        'ret' : realized_returns
    }).dropna()

    # Assign deciles based on predictions (1 = worst, 10 = best)
    df['decile'] = pd.qcut(df['pred'], 10, labels = False, duplicates = 'drop') + 1

    # Calculate average realized return per decile
    decile_returns = df.groupby('decile')['ret'].mean()

    # Plot bar chart
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.bar(decile_returns.index, decile_returns.values, colour = '#06A77D', alpha = 0.8)
    ax.set_xlabel('Prediction Decile', fontsize = 12)
    ax.set_ylabel('Average Realized Return', fontsize = 12)
    ax.set_title('Return by Prediction Decile (Monotonicity Check)', fontsize = 14, fontweight = 'bold')
    ax.grid(True, alpha = 0.3, axis = 'y')

    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
    plt.close()

def plot_ic_timeseries(predictions, realized_returns, output_path):
    """
    Plot rolling Information Coefficient (IC) over time.
    IC = correlation between predictions and realized returns per date.
    """

    # Combine and group by date
    df = pd.DataFrame({
        'pred' : predictions,
        'ret' : realized_returns
    })

    # Calculate IC per date (Spearman correlation)
    ic_by_date = df.groupby(level = 0).apply(
        lambda x: x['pred'].corr(x['ret'], method = 'spearman')
    )

    # Plot IC timeseries
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(ic_by_date.index, ic_by_date.values, linewidth = 1.5, colour = '#F18f01', alpha = 0.7)
    ax.axhline(0, color = 'black', linestyle = '--', linewidth = 1)
    ax.set_ylabel('Information Coefficient', fontsize = 12)
    ax.set_xlabel('Date', fontsize = 12)
    ax.set_title('Rolling Information Coefficient (Prediciton vs. Realized Returns)', fontsize = 14, fontweight = 'bold')
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
    plt.close()