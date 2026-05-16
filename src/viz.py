import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def build_base_figure(surface):
    k_grid = surface['k_grid']
    T_grid = surface['T_grid']
    iv_surface = surface['iv_surface']
    
    # Create 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'heatmap'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        subplot_titles=(
            '3D Implied Volatility Surface', '2D Volatility Heatmap',
            'Smile Slice (T ≈ 30 Days)', 'Term Structure (ATM)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. 3D Surface
    fig.add_trace(
        go.Surface(
            z=iv_surface, x=k_grid, y=T_grid,
            colorscale='Plasma', showscale=False,
            name='Surface'
        ),
        row=1, col=1
    )
    
    # 2. 2D Heatmap
    fig.add_trace(
        go.Heatmap(
            z=iv_surface, x=k_grid, y=T_grid,
            colorscale='Plasma', showscale=True,
            colorbar=dict(title='IV', x=1.02, y=0.75, len=0.45),
            name='Heatmap'
        ),
        row=1, col=2
    )
    
    # 3. Smile Slice (T closest to 30 days)
    i_T30 = np.argmin(np.abs(T_grid - 30/365.25))
    fig.add_trace(
        go.Scatter(
            x=k_grid, y=iv_surface[i_T30, :],
            mode='lines', line=dict(color='#ff7f0e', width=3),
            name='Smile'
        ),
        row=2, col=1
    )
    
    # 4. Term Structure (k closest to 0)
    i_k0 = np.argmin(np.abs(k_grid))
    fig.add_trace(
        go.Scatter(
            x=T_grid, y=iv_surface[:, i_k0],
            mode='lines', line=dict(color='#1f77b4', width=3),
            name='Term'
        ),
        row=2, col=2
    )
    
    # Layout styling
    fig.update_layout(
        template='plotly_dark',
        title={
            'text': 'IVSurfaceExplorer — AMD Implied Volatility Surface',
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#d8dde4', family='Inter, ui-sans-serif'),
        height=850,
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=False
    )
    
    # Axis labels
    fig.update_scenes(
        xaxis_title='Log-Moneyness (k)',
        yaxis_title='Time to Expiry (T)',
        zaxis_title='Implied Vol (σ)'
    )
    fig.update_xaxes(title_text='Log-Moneyness (k)', row=1, col=2)
    fig.update_yaxes(title_text='Time to Expiry (T)', row=1, col=2)
    fig.update_xaxes(title_text='Log-Moneyness (k)', row=2, col=1)
    fig.update_yaxes(title_text='Implied Vol (σ)', row=2, col=1)
    fig.update_xaxes(title_text='Time to Expiry (T)', row=2, col=2)
    fig.update_yaxes(title_text='Implied Vol (σ)', row=2, col=2)
    
    return fig

def render_dashboard_html(surface, fig, out_path='docs/index.html'):
    fig_json = fig.to_json()
    
    # Extract surface data for JS
    surface_data = {
        'k_grid': surface['k_grid'].tolist(),
        'T_grid': surface['T_grid'].tolist(),
        'iv_surface': surface['iv_surface'].tolist(),
        'spot': float(surface['spot'])
    }
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IVSurfaceExplorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        body {{
            background-color: #0d1117;
            color: #d8dde4;
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            background: #161b22;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #30363d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            max-width: 1200px;
            width: 100%;
            justify-content: space-around;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 200px;
        }}
        .control-group label {{
            font-weight: 700;
            font-size: 0.9rem;
            color: #8b949e;
        }}
        .control-group input {{
            width: 100%;
            accent-color: #58a6ff;
        }}
        .value-display {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #58a6ff;
        }}
        #plot {{
            width: 100%;
            max-width: 1400px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; font-size: 1.8rem; }}
        .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>IVSurfaceExplorer</h1>
        <p class="subtitle">AMD Parametric Implied Volatility Surface</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Spot Shift: <span id="val-spot" class="value-display">0.000</span></label>
            <input type="range" id="spot_shift" min="-0.10" max="0.10" step="0.005" value="0">
        </div>
        <div class="control-group">
            <label>ATM IV Shift: <span id="val-iv" class="value-display">0.000</span></label>
            <input type="range" id="iv_shift" min="-0.10" max="0.10" step="0.005" value="0">
        </div>
        <div class="control-group">
            <label>Skew Shift: <span id="val-skew" class="value-display">0.000</span></label>
            <input type="range" id="skew_shift" min="-0.5" max="0.5" step="0.025" value="0">
        </div>
        <div class="control-group">
            <label>Term Shift: <span id="val-term" class="value-display">0.000</span></label>
            <input type="range" id="term_shift" min="-0.3" max="0.3" step="0.025" value="0">
        </div>
    </div>

    <div id="plot"></div>

    <script>
        const baseSurface = {json.dumps(surface_data)};
        const fig = {fig_json};
        
        const kGrid = baseSurface.k_grid;
        const TGrid = baseSurface.T_grid;
        const ivBase = baseSurface.iv_surface;
        
        const iT30 = findClosestIndex(TGrid, 30/365.25);
        const ik0 = findClosestIndex(kGrid, 0);

        function findClosestIndex(arr, val) {{
            return arr.reduce((bestIdx, curr, idx) => 
                Math.abs(curr - val) < Math.abs(arr[bestIdx] - val) ? idx : bestIdx, 0);
        }}

        // Linear interpolation helper for JS deformation
        function interpolateBase(k, TIdx) {{
            // We only shift along k-axis, so T index is fixed
            const row = ivBase[TIdx];
            if (k <= kGrid[0]) return row[0];
            if (k >= kGrid[kGrid.length - 1]) return row[row.length - 1];
            
            // Binary search for interval
            let low = 0, high = kGrid.length - 1;
            while (high - low > 1) {{
                let mid = (low + high) >> 1;
                if (kGrid[mid] > k) high = mid;
                else low = mid;
            }}
            
            const t = (k - kGrid[low]) / (kGrid[high] - kGrid[low]);
            return row[low] * (1 - t) + row[high] * t;
        }}

        function deform() {{
            const spotShift = parseFloat(document.getElementById('spot_shift').value);
            const ivShift = parseFloat(document.getElementById('iv_shift').value);
            const skewShift = parseFloat(document.getElementById('skew_shift').value);
            const termShift = parseFloat(document.getElementById('term_shift').value);

            document.getElementById('val-spot').innerText = spotShift.toFixed(3);
            document.getElementById('val-iv').innerText = ivShift.toFixed(3);
            document.getElementById('val-skew').innerText = skewShift.toFixed(3);
            document.getElementById('val-term').innerText = termShift.toFixed(3);

            const kShift = Math.log(1 + spotShift);
            const TRef = 30 / 365.25;

            const nT = TGrid.length;
            const nk = kGrid.length;
            
            let newZ = [];
            for (let i = 0; i < nT; i++) {{
                let row = [];
                for (let j = 0; j < nk; j++) {{
                    const k = kGrid[j];
                    const T = TGrid[i];
                    
                    // Formula: iv_new(k, T) = iv_base(k + kShift, T) + ivShift + skewShift*k + termShift*(T - TRef)
                    let val = interpolateBase(k + kShift, i) + ivShift + skewShift * k + termShift * (T - TRef);
                    row.push(Math.max(0.01, Math.min(5.0, val)));
                }}
                newZ.push(row);
            }}

            // Update traces
            // 0: Surface, 1: Heatmap, 2: Smile, 3: Term
            
            // Plotly.react is efficient for updates
            Plotly.react('plot', [
                {{...fig.data[0], z: newZ}},
                {{...fig.data[1], z: newZ}},
                {{...fig.data[2], y: newZ[iT30]}},
                {{...fig.data[3], y: newZ.map(r => r[ik0])}}
            ], fig.layout);
        }}

        // Initialize plot
        Plotly.newPlot('plot', fig.data, fig.layout);

        // Attach listeners
        ['spot_shift', 'iv_shift', 'skew_shift', 'term_shift'].forEach(id => {{
            document.getElementById(id).addEventListener('input', deform);
        }});
    </script>
</body>
</html>
"""
    with open(out_path, 'w') as f:
        f.write(html_template)
