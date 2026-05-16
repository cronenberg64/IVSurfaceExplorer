import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_amd_options() -> pd.DataFrame:
    """
    Fetch AMD options chain from yfinance and filter/format for IV surface construction.
    """
    ticker = yf.Ticker('AMD')
    spot = ticker.fast_info['lastPrice']
    expiries = ticker.options
    
    all_options = []
    today = datetime.now()
    
    for expiry in expiries:
        opt_chain = ticker.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        calls['option_type'] = 'C'
        puts['option_type'] = 'P'
        
        df_expiry = pd.concat([calls, puts])
        df_expiry['expiry_date'] = pd.to_datetime(expiry)
        all_options.append(df_expiry)
        
    df = pd.concat(all_options)
    
    # Rename and select columns
    df = df.rename(columns={
        'lastTradeDate': 'last_trade_date',
        'impliedVolatility': 'yf_iv'
    })
    
    # Calculate mid price
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    # Time to expiry in years
    df['time_to_expiry'] = (df['expiry_date'] - today).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Add spot
    df['spot'] = spot
    
    # Filter
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]
    df = df.dropna(subset=['mid'])
    df = df[(df['time_to_expiry'] >= 5/365.25) & (df['time_to_expiry'] <= 1.0)]
    
    return df[['expiry_date', 'strike', 'option_type', 'bid', 'ask', 'mid', 
               'volume', 'openInterest', 'time_to_expiry', 'spot', 'yf_iv']]
