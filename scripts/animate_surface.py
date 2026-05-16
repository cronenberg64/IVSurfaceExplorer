import sys
import os
import numpy as np
import glob
import imageio
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.surface import deform_surface
from src.viz import build_base_figure

def main():
    # 1. Load latest surface
    surface_files = glob.glob('outputs/surface_amd_*.npz')
    if not surface_files:
        print("No surface files found. Run fetch_snapshot.py first.")
        return
    
    latest_surface = max(surface_files)
    print(f"Animating from base surface: {latest_surface}")
    
    with np.load(latest_surface, allow_pickle=True) as data:
        base_surface = {key: data[key] for key in data.files}
    
    # 2. Define animation sweep
    # We'll sweep spot shift from -0.10 to 0.10 and back
    spot_shifts = np.linspace(-0.10, 0.10, 15)
    spot_shifts = np.concatenate([spot_shifts, spot_shifts[::-1]])
    
    frames_dir = 'outputs/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_files = []
    print(f"Generating {len(spot_shifts)} frames...")
    
    for i, shift in enumerate(spot_shifts):
        # Apply deformation
        deformed_iv = deform_surface(base_surface, spot_shift=shift)
        
        # Prepare surface dict for plotting
        temp_surface = base_surface.copy()
        temp_surface['iv_surface'] = deformed_iv
        
        # Build figure
        fig = build_base_figure(temp_surface)
        fig.update_layout(title_text=f"IVSurfaceExplorer — Spot Shift: {shift:+.3f}")
        
        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        fig.write_image(frame_path, width=1200, height=800)
        frame_files.append(frame_path)
        print(f"Frame {i+1}/{len(spot_shifts)} done.")
        
    # 3. Stitch into GIF
    output_gif = 'docs/assets/slider_sweep.gif'
    print(f"Stitching {len(frame_files)} frames into {output_gif}...")
    
    images = []
    for filename in frame_files:
        images.append(imageio.imread(filename))
        
    # duration is seconds per frame
    imageio.mimsave(output_gif, images, duration=0.1, loop=0)
    
    # 4. Clean up frames
    print("Cleaning up temporary frames...")
    for f in frame_files:
        os.remove(f)
    os.rmdir(frames_dir)
    
    print(f"Animation complete! Saved to: {output_gif}")

if __name__ == "__main__":
    main()
