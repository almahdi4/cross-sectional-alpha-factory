import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def download_fundamentals(tickers, start_date, end_date):
    """
    Download quarterly fundamental data from Yahoo Finance.
    Returns: DataFrame with P/E, profit margin, revenue growth per ticker per quarter
    """
    print(f"Downloading fundamentals for {len(tickers)} tickers...")

    all_fundamentals = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(tickers)} tickers...")
        
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            
            # Get quarterly financials (income statement and balance sheet)
            quarterly_financials = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet
            
            if quarterly_financials is None or quarterly_balance.empty:
                continue
            
            # Get quarterly info (has trailing P/E)
            info = stock.info
            
            # Extract key metrics for each available quarter
            for date in quarterly_financials.columns:
                # Get revenue and net income
                revenue = quarterly_financials.loc['Total Revenue', date] if 'Total Revenue' in quarterly_financials.index else np.nan
                net_income = quarterly_financials.loc['Net Income', date] if 'Net Income' in quarterly_financials.index else np.nan
                
                # Calculate profit margin = net income / revenue
                profit_margin = (net_income / revenue) if (revenue and not pd.isna(revenue) and revenue != 0) else np.nan

                # For revenue growth, we need to compare to same quarter last year (YoY)
                # Find the date 4 quarters ago (1 year)
                try:
                    dates_list = list(quarterly_financials.columns)
                    current_idx = dates_list.index(date)
                    
                    # Get revenue from 4 quarters ago (year-over-year comparison)
                    if current_idx + 4 < len(dates_list):
                        prev_year_date = dates_list[current_idx + 4]
                        prev_revenue = quarterly_financials.loc['Total Revenue', prev_year_date] if 'Total Revenue' in quarterly_financials.index else np.nan
                        
                        # Calculate YoY revenue growth rate
                        revenue_growth = (revenue - prev_revenue) / prev_revenue if (prev_revenue and not pd.isna(prev_revenue) and prev_revenue != 0) else np.nan
                    else:
                        revenue_growth = np.nan
                except:
                    revenue_growth = np.nan
                
                # Get P/E ratio from stock info (trailing 12 months)
                pe_ratio = info.get('trailingPE', np.nan) if info else np.nan

                # Store the data point
                all_fundamentals.append({
                    'date': pd.to_datetime(date),
                    'ticker': ticker,
                    'profit_margin': profit_margin,
                    'revenue_growth_yoy': revenue_growth,
                    'pe_ratio': pe_ratio
                })
        
        except Exception as e:
            # If download fails for this ticker, skip it
            print(f"Could not download {ticker}: {str(e)[:50]}")
            continue
    
    # Convert to DataFrame
    fundamentals_df = pd.DataFrame(all_fundamentals)
    
    if fundamentals_df.empty:
        print("No fundamental data downloaded")
        return pd.DataFrame()
    
    # Force convert all numeric columns to float (fixes "object" dtype issue)
    numeric_cols = ['profit_margin', 'revenue_growth_yoy', 'pe_ratio']
    for col in numeric_cols:
        fundamentals_df[col] = pd.to_numeric(fundamentals_df[col], errors='coerce')
    
    # Drop rows where ALL fundamentals are NaN
    fundamentals_df = fundamentals_df.dropna(subset=numeric_cols, how='all')
    
    print(f"Downloaded {len(fundamentals_df)} fundamental data points")
    return fundamentals_df


def align_fundamentals_to_prices(fundamentals_df, daily_index, tickers):
    """
    Forward-fill quarterly fundamentals to daily frequency.
    Each day uses the most recent available fundamental data.
    """
    if fundamentals_df.empty:
        return pd.DataFrame()
    
    # Create MultiIndex for daily data
    idx = pd.MultiIndex.from_product([daily_index, tickers], names=['date', 'ticker'])
    
    # Initialize empty DataFrame
    aligned = pd.DataFrame(index=idx, columns=['profit_margin', 'revenue_growth_yoy', 'pe_ratio'], dtype=float)
    
    # For each ticker, forward-fill from quarterly to daily
    for ticker in tickers:
        ticker_data = fundamentals_df[fundamentals_df['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            continue
        
        # Set date as index and sort
        ticker_data = ticker_data.set_index('date').sort_index()
        
        # Reindex to daily frequency with forward fill
        daily_data = ticker_data.reindex(daily_index, method='ffill')
        
        # Assign to aligned DataFrame
        for col in ['profit_margin', 'revenue_growth_yoy', 'pe_ratio']:
            aligned.loc[(slice(None), ticker), col] = daily_data[col].values
    
    # Ensure all columns are float type
    aligned = aligned.astype(float)
    
    return aligned


def build_fundamental_features(fundamentals_df):
    """
    Cross-sectional normalization of fundamental features.
    Same approach as price features: winsorize + z-score per date.
    """
    if fundamentals_df.empty:
        return pd.DataFrame()
    
    # Use the same normalization function from price features
    from src.features.price import xsection_normalize
    
    normalized = xsection_normalize(fundamentals_df)
    
    return normalized





