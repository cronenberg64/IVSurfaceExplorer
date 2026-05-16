import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, sigma, option_type='C') -> float:
    """
    Calculate Black-Scholes price for European call or put.
    """
    if T <= 0:
        if option_type == 'C':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, option_type='C') -> float:
    """
    Invert Black-Scholes formula to find implied volatility using brentq.
    """
    # Objective function: BS price - observed price
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
        
    try:
        # Bracket [0.01, 5.0] as specified
        return brentq(objective, 0.01, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan
