# IVSurfaceExplorer

**IVSurfaceExplorer** is an interactive parametric implied volatility surface explorer for AMD options. It fetches live data from `yfinance`, inverts mid-prices to Black-Scholes implied volatilities, and fits a smoothed 3D surface that can be deformed in real-time via sliders.

### [🚀 View Live Demo](https://cronenberg64.github.io/IVSurfaceExplorer/)

![Static Surface](docs/assets/surface_static.png)

## Features

- **Live Data Pipeline**: Pulls the full AMD options chain across all expiries, filtering for liquidity and OTM consistency.
- **IV Inversion**: High-precision Black-Scholes inversion using `scipy.optimize.brentq`.
- **Interpolated Surface**: 3D surface construction using cubic spline interpolation with linear and nearest-neighbor fallbacks for edge regions.
- **Real-Time Deformations**: Four sliders allow you to manipulate the surface across canonical risk dimensions:
    - **Spot Shift**: Re-evaluates the surface relative to log-moneyness shifts.
    - **ATM IV Shift**: Parallel vertical shift of the entire surface.
    - **Skew Shift**: Linear tilt of the smile (vol-skew exposure).
    - **Term Shift**: Linear tilt of the term structure relative to a 30-day reference.
- **SVI Fitting (Stretch)**: Parametric fit using Gatheral's SVI (Stochastic Volatility Inspired) model per maturity slice, with parameter trajectory tracking.

## Methodology

### 1. Black-Scholes Inversion
We solve for $\sigma$ such that:
$$C_{\text{obs}} = S N(d_1) - K e^{-rT} N(d_2)$$
Using a risk-free rate $r=0.045$ and a bracketed solver ($[0.01, 5.0]$), we find the unique implied volatility for each liquid option.

### 2. Surface Construction
The surface is built on a log-moneyness grid $k = \ln(K/S)$ and time-to-expiry $T$. We use an OTM (Out-Of-The-Money) convention—using only calls for $k \ge 0$ and puts for $k < 0$—to minimize American-exercise bias and liquidity noise.

### 3. Parametric Deformations
The dashboard implements the following combined deformation formula:
$$\text{IV}(k, T) = \max\big(0.01,\ \text{IV}_0(k + \ln(1+\Delta s),\, T) + \Delta\sigma + \Delta\beta \cdot k + \Delta\gamma \cdot (T - 30/365)\big)$$
Where $\Delta s$ is spot shift, $\Delta\sigma$ is IV shift, $\Delta\beta$ is skew shift, and $\Delta\gamma$ is term shift.

## Repository Structure

```
IVSurfaceExplorer/
    ├── docs/                          # GitHub Pages source (index.html)
    │   └── assets/                    # Public images (static & GIF)
    ├── src/
│   ├── data.py                    # yfinance options fetching
│   ├── bs.py                      # Black-Scholes pricing + IV inversion
│   ├── surface.py                 # Surface fitting + deformations
│   ├── viz.py                     # Plotly figure + HTML template
│   └── svi.py                     # SVI parametric fitting
├── scripts/
│   ├── fetch_snapshot.py          # CLI: fetch options, compute IV, build surface
│   └── build_dashboard.py         # CLI: render the GitHub Pages HTML
└── outputs/
    ├── snapshot_amd_<date>.parquet
    ├── surface_amd_<date>.npz
    ├── svi_params.png             # SVI parameter trajectories (tracked)
    └── surface_static.png         # Dashboard screenshot (tracked)
```

## Setup & Reproducibility

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Fetch data and build surface**:
   ```bash
   python scripts/fetch_snapshot.py
   ```

3. **Generate dashboard**:
   ```bash
   python scripts/build_dashboard.py
   ```

## License
Apache 2.0