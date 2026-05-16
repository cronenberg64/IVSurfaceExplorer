import numpy as np
import pandas as pd
from scipy.optimize import minimize

def svi_w(k, a, b, rho, m, sigma_param):
    """
    Gatheral SVI formula for total variance: w(k) = sigma^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma_param**2))

def fit_svi_slice(k_obs, iv_obs, T):
    """
    Fit SVI parameters to a single maturity slice.
    """
    # Convert IV to total variance
    w_obs = iv_obs**2 * T
    
    # Objective: MSE
    def objective(params):
        a, b, rho, m, sigma_param = params
        w_fit = svi_w(k_obs, a, b, rho, m, sigma_param)
        return np.mean((w_fit - w_obs)**2)
    
    # Initial guess
    # a: level, b: slope, rho: skew, m: shift, sigma: curvature
    initial_guess = [np.median(w_obs), 0.1, -0.3, 0.0, 0.1]
    
    # Bounds
    bounds = [
        (1e-6, 2.0),   # a > 0
        (1e-6, 1.0),   # b > 0
        (-0.99, 0.99), # -1 < rho < 1
        (-1.0, 1.0),   # m
        (1e-6, 1.0)    # sigma_param > 0
    ]
    
    res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if not res.success:
        return None
        
    a, b, rho, m, sigma_param = res.x
    
    # Lee's bound (soft no-arbitrage check)
    # b(1 + |rho|) <= 4
    no_arb_ok = b * (1 + abs(rho)) <= 4.0
    
    return {
        'params': {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma_param': sigma_param},
        'rmse': np.sqrt(res.fun),
        'no_arb_ok': no_arb_ok
    }

def fit_svi_surface(df):
    """
    Fit SVI parameters for each expiry in the dataset.
    """
    df = df.copy()
    df['k'] = np.log(df['strike'] / df['spot'])
    df = df.dropna(subset=['our_iv'])
    
    results = []
    expiries = sorted(df['expiry_date'].unique())
    
    for expiry in expiries:
        slice_df = df[df['expiry_date'] == expiry]
        if len(slice_df) < 5:
            continue
            
        T = slice_df['time_to_expiry'].iloc[0]
        fit = fit_svi_slice(slice_df['k'].values, slice_df['our_iv'].values, T)
        
        if fit:
            row = fit['params'].copy()
            row['expiry'] = expiry
            row['T'] = T
            row['rmse'] = fit['rmse']
            row['no_arb_ok'] = fit['no_arb_ok']
            results.append(row)
            
    return pd.DataFrame(results)
