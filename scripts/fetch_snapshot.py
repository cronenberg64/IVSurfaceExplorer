import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import fetch_amd_options
from src.bs import implied_vol
from src.surface import build_surface

def main():
    print("Fetching AMD options data...")
    df = fetch_amd_options()
    
    r = 0.045
    print(f"Calculating implied volatilities (r={r})...")
    
    # Calculate our IV row-wise
    df['our_iv'] = df.apply(
        lambda row: implied_vol(
            row['mid'], row['spot'], row['strike'], row['time_to_expiry'], r, row['option_type']
        ), axis=1
    )
    
    # Save snapshot
    today_str = datetime.now().strftime('%Y-%m-%d')
    snapshot_path = f'outputs/snapshot_amd_{today_str}.parquet'
    df.to_parquet(snapshot_path)
    
    print("Building IV surface...")
    surface_dict = build_surface(df)
    surface_path = f'outputs/surface_amd_{today_str}.npz'
    np.savez(surface_path, **surface_dict)
    
    # Summary statistics
    n_expiries = df['expiry_date'].nunique()
    strike_min = df['strike'].min()
    strike_max = df['strike'].max()
    mean_iv = df['our_iv'].mean()
    
    # Sanity check: RMS error vs yf_iv
    mask = df['our_iv'].notna() & df['yf_iv'].notna()
    rmse = np.sqrt(((df.loc[mask, 'our_iv'] - df.loc[mask, 'yf_iv'])**2).mean())
    
    print("\n--- Summary ---")
    print(f"Number of expiries: {n_expiries}")
    print(f"Strike range: {strike_min:.2f} - {strike_max:.2f}")
    print(f"Mean our_iv: {mean_iv:.4f}")
    print(f"RMS error vs yfinance IV: {rmse:.4f}")
    print(f"Snapshot saved to: {snapshot_path}")
    
    print("\n--- Surface Info ---")
    print(f"Spot: {surface_dict['spot']:.2f}")
    print(f"k_grid range: {surface_dict['k_grid'].min():.2f} to {surface_dict['k_grid'].max():.2f}")
    print(f"T_grid range: {surface_dict['T_grid'].min():.4f} to {surface_dict['T_grid'].max():.4f}")
    print(f"IV min/max/mean: {surface_dict['iv_surface'].min():.4f} / {surface_dict['iv_surface'].max():.4f} / {surface_dict['iv_surface'].mean():.4f}")
    print(f"Surface saved to: {surface_path}")

if __name__ == "__main__":
    main()
