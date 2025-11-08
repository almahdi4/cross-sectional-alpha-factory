import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path so 'src' module can be found
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_prices_volume
from src.features.price import build_price_features
from src.features.fundamentals import (
    download_fundamentals,
    align_fundamentals_to_prices,
    build_fundamental_features
)
from src.features.calendar_features import build_calendar_features

tickers = ["AAPL", "MSFT", "GOOGL"]
start_date = "2023-01-01"
end_date = "2024-12-31"


prices, volume = load_prices_volume(tickers, start_date, end_date)
print("Loaded prices and volume:", prices.shape, volume.shape)


price_features = build_price_features(prices, volume)
print("Price features built:", price_features.shape)


tickers = ['AAPL', 'MSFT']  # Test with just 2 tickers for speed
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

fund_df = download_fundamentals(tickers, start_date, end_date)


aligned_fund = align_fundamentals_to_prices(fund_df, prices.index, prices.columns)
print("Aligned fundamentals:", aligned_fund.shape)


fund_features = build_fundamental_features(aligned_fund)
print("Normalized fundamentals:", fund_features.shape)


calendar_features = build_calendar_features(None, prices.index, prices.columns)
print("Calendar features built:", calendar_features.shape)


price_features.index = price_features.index.set_names(['date', 'ticker']) 
fund_features.index = fund_features.index.set_names(['date', 'ticker'])
calendar_features.index = calendar_features.index.set_names(['date', 'ticker'])


combined = (
    price_features
    .join(fund_features, how="left")
    .join(calendar_features, how="left")
)

print("Combined dataset shape:", combined.shape)

print("\nSample combined data:\n", combined.head(10))

def test_fundamental_features_dtype():
    """Test that fundamental features are numeric (not object)"""
    from src.features.fundamentals import download_fundamentals
    import pandas as pd
    
    # Download fundamentals for a few tickers (quick test)
    tickers = ['AAPL', 'MSFT']
    fund = download_fundamentals(tickers, '2024-01-01', '2024-12-31')
    
    if not fund.empty:
        # Check all columns are numeric
        numeric_cols = ['profit_margin', 'revenue_growth_yoy', 'pe_ratio']
        for col in numeric_cols:
            assert fund[col].dtype in ['float64', 'int64'], f"{col} must be numeric, got {fund[col].dtype}"
        
        # Check no infinite values
        assert not fund[numeric_cols].isin([float('inf'), -float('inf')]).any().any(), \
            "Fundamental features contain infinite values"


def test_forward_fill_no_lookahead():
    """Test that forward-fill doesn't introduce look-ahead bias"""
    from src.features.fundamentals import align_fundamentals_to_prices
    import pandas as pd
    
    # Create mock quarterly data
    fundamentals = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-04-01']),
        'ticker': ['AAPL', 'AAPL'],
        'profit_margin': [0.25, 0.27]
    })
    
    # Create daily index
    daily_index = pd.date_range('2024-01-01', '2024-06-01', freq='D')
    
    # Align to daily
    aligned = align_fundamentals_to_prices(fundamentals, daily_index, ['AAPL'])
    
    # Check: Jan 2 should use Jan 1 data (forward fill)
    jan2_margin = aligned.loc[('2024-01-02', 'AAPL'), 'profit_margin']
    assert jan2_margin == 0.25, "Forward fill should use previous quarter's value"
    
    # Check: April 2 should use April 1 NEW data
    apr2_margin = aligned.loc[('2024-04-02', 'AAPL'), 'profit_margin']
    assert apr2_margin == 0.27, "Should update to new quarter immediately"

def test_features_handle_missing_data():
    """Test that feature engineering handles missing data gracefully"""
    from src.features.price import build_price_features
    import pandas as pd
    import numpy as np
    
    # Create price data with some NaNs
    dates = pd.date_range('2024-01-01', '2024-01-30', freq='D')
    prices = pd.DataFrame({
        'AAPL': np.random.randn(len(dates)) + 100,
        'MSFT': np.random.randn(len(dates)) + 100
    }, index=dates)
    
    # Introduce some NaNs (simulate missing data)
    prices.iloc[5:8, 0] = np.nan  # AAPL missing for 3 days
    
    volume = pd.DataFrame({
        'AAPL': np.random.randint(1000, 10000, len(dates)),
        'MSFT': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Build features
    features = build_price_features(prices, volume)
    
    # Features should handle NaNs gracefully
    # (either by dropping those rows or forward-filling)
    # At minimum, no infinite values should appear
    assert not features.isin([float('inf'), -float('inf')]).any().any(), \
        "Features should not contain infinite values"
    
    # Check that features are calculated for non-NaN periods
    assert len(features) > 0, "Should produce some valid features"


