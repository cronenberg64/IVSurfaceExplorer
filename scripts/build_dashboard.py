import sys
import os
import numpy as np
import glob

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.viz import build_base_figure, render_dashboard_html
from src.svi import fit_svi_surface
import plotly.express as px
import pandas as pd

def main():
    # Find latest surface npz
    surface_files = glob.glob('outputs/surface_amd_*.npz')
    if not surface_files:
        print("No surface files found in outputs/. Run fetch_snapshot.py first.")
        return
        
    latest_surface = max(surface_files)
    print(f"Loading latest surface: {latest_surface}")
    
    with np.load(latest_surface, allow_pickle=True) as data:
        surface = {key: data[key] for key in data.files}
    
    print("Building Plotly figure...")
    fig = build_base_figure(surface)
    
    print("Rendering HTML dashboard...")
    os.makedirs('docs', exist_ok=True)
    render_dashboard_html(surface, fig, out_path='docs/index.html')
    
    print("Fitting SVI parameters...")
    # Need original snapshot for SVI fitting (observed points, not just interpolated surface)
    snapshot_files = glob.glob('outputs/snapshot_amd_*.parquet')
    if snapshot_files:
        latest_snapshot = max(snapshot_files)
        df = pd.read_parquet(latest_snapshot)
        svi_results = fit_svi_surface(df)
        
        if not svi_results.empty:
            print(f"SVI fitting complete for {len(svi_results)} expiries.")
            
            # Plot SVI parameters vs T
            params = ['a', 'b', 'rho', 'm', 'sigma_param']
            fig_svi = px.line(
                svi_results, x='T', y=params,
                title='SVI Parametric Trajectories across Maturity',
                template='plotly_dark',
                labels={'value': 'Parameter Value', 'T': 'Time to Expiry (Years)'}
            )
            fig_svi.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117')
            
            # Save SVI plot (static)
            svi_plot_path = 'docs/assets/svi_params.png'
            try:
                # Requires kaleido
                fig_svi.write_image(svi_plot_path)
                print(f"SVI parameter plot saved to: {svi_plot_path}")
            except Exception as e:
                print(f"Could not save SVI image (missing kaleido?): {e}")
                # Save as HTML as backup
                fig_svi.write_html('outputs/svi_params.html')
    
    print(f"Dashboard generated at: docs/index.html")

if __name__ == "__main__":
    main()
