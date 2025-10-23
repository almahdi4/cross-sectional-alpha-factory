from datetime import datetime
from pathlib import Path

from src.data.loaders import load_prices_volume
from src.features.price import build_price_features
from src.features.fundamentals import (
    load_fundamentals_csv,
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


fund_path = Path("data/fundamentals.csv")
fund_df = load_fundamentals_csv(fund_path)


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
