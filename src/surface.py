import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator

def build_surface(df, n_k=50, n_T=30) -> dict:
    """
    Construct a regular IV surface grid from option observations.
    """
    # 1. Compute log-moneyness k = log(strike / spot)
    df = df.copy()
    df['k'] = np.log(df['strike'] / df['spot'])
    
    # 2. Drop NaN IVs
    df = df.dropna(subset=['our_iv'])
    
    # 3. OTM convention: calls for k >= 0, puts for k < 0
    df_otm = df[((df['option_type'] == 'C') & (df['k'] >= 0)) | 
                ((df['option_type'] == 'P') & (df['k'] < 0))]
    
    if df_otm.empty:
        # Fallback to all if OTM is empty (shouldn't happen with liquid AMD)
        df_otm = df
    
    k_obs = df_otm['k'].values
    T_obs = df_otm['time_to_expiry'].values
    iv_obs = df_otm['our_iv'].values
    
    # 4. Grid setup
    k_grid = np.linspace(-0.30, 0.30, n_k)
    T_min = df_otm['time_to_expiry'].min()
    T_max = min(df_otm['time_to_expiry'].max(), 1.0)
    T_grid = np.linspace(T_min, T_max, n_T)
    
    KK, TT = np.meshgrid(k_grid, T_grid)
    
    # 5. Interpolation
    # Cubic interpolation
    iv_surface = griddata((k_obs, T_obs), iv_obs, (KK, TT), method='cubic')
    
    # Linear fallback for NaNs
    nan_mask = np.isnan(iv_surface)
    if nan_mask.any():
        iv_linear = griddata((k_obs, T_obs), iv_obs, (KK, TT), method='linear')
        iv_surface[nan_mask] = iv_linear[nan_mask]
        
    # Nearest fallback for remaining NaNs (edges)
    nan_mask = np.isnan(iv_surface)
    if nan_mask.any():
        iv_nearest = griddata((k_obs, T_obs), iv_obs, (KK, TT), method='nearest')
        iv_surface[nan_mask] = iv_nearest[nan_mask]
        
    return {
        'k_grid': k_grid,
        'T_grid': T_grid,
        'iv_surface': iv_surface,
        'spot': df['spot'].iloc[0]
    }

def deform_surface(surface_dict, spot_shift=0.0, iv_shift=0.0, skew_shift=0.0, term_shift=0.0) -> np.ndarray:
    """
    Apply parametric deformations to the base IV surface.
    """
    k_grid = surface_dict['k_grid']
    T_grid = surface_dict['T_grid']
    iv_base = surface_dict['iv_surface']
    
    # spot_shift affects log-moneyness: k_new = k_old - log(1 + spot_shift)
    # We need to evaluate iv_base(k + log(1 + spot_shift), T)
    k_shift = np.log(1 + spot_shift)
    
    # Setup interpolator for the base surface to handle k-axis shift
    # RegularGridInterpolator(points, values) where points is (T_grid, k_grid)
    # because iv_base is (n_T, n_k)
    interp = RegularGridInterpolator(
        (T_grid, k_grid), 
        iv_base, 
        bounds_error=False, 
        fill_value=None # Extrapolate nearest
    )
    
    KK, TT = np.meshgrid(k_grid, T_grid)
    
    # Shifted coordinates
    points = np.stack([TT.ravel(), KK.ravel() + k_shift], axis=-1)
    iv_shifted = interp(points).reshape(len(T_grid), len(k_grid))
    
    # Apply other deformations:
    # iv_new = iv_shifted + iv_shift + skew_shift*k + term_shift*(T - 30/365)
    
    # Term structure tilt reference: 30 days
    T_ref = 30 / 365.25
    
    # Broadcast k and T to surface shape
    # KK is (n_T, n_k), TT is (n_T, n_k)
    
    iv_final = iv_shifted + iv_shift + skew_shift * KK + term_shift * (TT - T_ref)
    
    # Clip to [0.01, 5.0]
    return np.clip(iv_final, 0.01, 5.0)
