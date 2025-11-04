import pandas as pd
import numpy as np

def build_price_features(prices, volume):
    """
    Build cross-sectional features from price and volume data.
    Returns: DataFrame with MultiIndex (date, ticker) and feature columns
    """

    # Calculate returns over multiple horizons (5, 10, 21, 63 trading days)
    ret_5 = prices.pct_change(5)    # 1 week momentum
    ret_10 = prices.pct_change(10)  # 2 week momentum
    ret_21 = prices.pct_change(21)  # 1 month momentum
    ret_63 = prices.pct_change(63)  # 3 month momentum

    # Stack into long format: (date, ticker) rows
    features = pd.DataFrame({
        'ret_5' : ret_5.stack(),
        'ret_10' : ret_10.stack(),
        'ret_21' : ret_21.stack(),
        'ret_63' : ret_63.stack()
    })

    # Volatility features: how "choppy" has the stock been?
    vol_5 = prices.pct_change().rolling(5).std() # 1 week volatility
    vol_21 = prices.pct_change().rolling(21).std() # 1 month volatility

    # RSI (Relative Strength Index): overbought/oversold indicator
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100/ (1 + rs))

    # Volume z-score: is volume unusually high/low vs recent avergage?
    vol_z_60 = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()

    # Add all volatility/volume features to the DataFrame
    features['vol_5'] = vol_5.stack()
    features['vol_21'] = vol_21.stack()
    features['rsi_14'] = rsi_14.stack()
    features['vol_z_60'] = vol_z_60.stack()

    #Beta: how much does this stock move with the overall market?
    #Idio vol: how much does it move independently of the market? 
    market_ret = prices.pct_change().mean(axis = 1) # Equal-weight market return

    beta_60 = pd.DataFrame(index = prices.index, columns = prices.columns)
    idio_vol_60 = pd.DataFrame(index = prices.index, columns = prices.columns)

    for ticker in prices.columns:
        stock_ret = prices[ticker].pct_change()
        # Rolling 60-day correlation and standard deviations
        cov = stock_ret.rolling(60).cov(market_ret) # Covariance
        var_market = market_ret.rolling(60).var()
        beta_60[ticker] = cov / var_market

        stock_vol = stock_ret.rolling(60).std()
        idio_vol_60[ticker] = np.sqrt((stock_vol**2) - (beta_60[ticker]**2 * market_ret.rolling(60).std()**2))
    
    features['beta_60'] = beta_60.stack()
    features['idio_vol_60'] = idio_vol_60.stack()

    return features

def xsection_normalize(df):
    """
    Winsorize and z-score features per date (cross-sectional normalization).
    This makes features comparable across different dates.
    """
    # Make a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Get unique dates
    dates = df.index.get_level_values(0).unique()
    
    # Normalize each column (feature) separately
    for col in df.columns:
        for date in dates:
            # Get mask for this date
            mask = df.index.get_level_values(0) == date
            
            # Get data for this date and column
            col_data = df.loc[mask, col].copy()
            
            # Skip if all NaN or insufficient data
            if col_data.isna().all() or len(col_data.dropna()) < 2:
                continue
            
            # Winsorize at 1st and 99th percentile
            lower = col_data.quantile(0.01)
            upper = col_data.quantile(0.99)
            col_data = col_data.clip(lower=lower, upper=upper)
            
            # Z-score normalization
            mean = col_data.mean()
            std = col_data.std()
            
            if std > 1e-9:  # Only normalize if std is not zero
                col_data = (col_data - mean) / std
            
            # Assign back to result using .loc
            result.loc[mask, col] = col_data.values
    
    return result

def build_labels(prices):
    """
    Build 5-day forward excess return (label to predict).
    Excesss return = stock return minus equal-weight market return
    """

    # 5-day forward return for each stock
    fwd_ret_5 = prices.pct_change(5).shift(-5)

    # Equal-weight market return (average across all stocks)
    market_fwd_ret = fwd_ret_5.mean(axis = 1)

    # Excess return = stock return - market return
    excess_ret = fwd_ret_5.subtract(market_fwd_ret, axis = 0)

    return excess_ret.stack()