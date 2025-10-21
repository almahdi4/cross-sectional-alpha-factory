import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from datetime import date, timedelta


from src.data.loaders import load_prices_volume
from src.features.price import build_price_features

tickers = ["AAPL", "MSFT"]
start = "2023-01-01"
end = "2024-12-31"

prices, volume = load_prices_volume(tickers, start, end)
features = build_price_features(prices, volume)
print(features.head())       # should show ret_5, ret_10, ret_21, ret_63 with (date, ticker) index
