import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_noise_calibrated():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'datasets'
    
    # 1. Load Data
    try:
        mat_d1 = sio.loadmat(data_dir / 'D1.mat')
        d1 = mat_d1['d'].flatten()
        d1_idx = mat_d1['Index'].flatten()
        d1_cls = mat_d1['Class'].flatten()
        
        mat_d6 = sio.loadmat(data_dir / 'D6.mat')
        d6 = mat_d6['d'].flatten()
    except FileNotFoundError:
        print("Error: Ensure D1.mat and D6.mat are in the ../datasets/ folder.")
        return

    # 2. Setup Window (1000 samples for better context)
    center_spike_idx = int(d1_idx[10]) 
    window_radius = 500  
    start, end = center_spike_idx - window_radius, center_spike_idx + window_radius
    
    d1_chunk = d1[start:end]
    time_axis = np.arange(start, end)
    
    # Identify Ground Truth for markers
    mask = (d1_idx >= start) & (d1_idx < end)
    local_spikes = d1_idx[mask]
    local_classes = d1_cls[mask]

    # 3. Define Sigma levels for comparison
    noise_levels = [2.5, 4.5, 5.8]
    fig, axes = plt.subplots(len(noise_levels) + 2, 1, figsize=(14, 18), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # FIX: Increase Y-Limit to prevent cutoff
    # A 5V spike + 2.5 sigma noise can hit ~12V
    y_limit_min, y_limit_max = -10, 13 

    # --- Plot 1: Real D6 ---
    d6_chunk = d6[start:end]
    axes[0].plot(time_axis, d6_chunk, color='red', alpha=0.8, label='Real D6 (Target)')
    axes[0].set_title("TARGET: Real D6 Recording (<0 dB SNR)", fontweight='bold')
    axes[0].legend(loc='upper right')

    # --- Plot 2: Clean D1 ---
    axes[1].plot(time_axis, d1_chunk, color='green', linewidth=1.5, label='Clean D1')
    axes[1].set_title("BASELINE: Clean D1 (Ground Truth Locations)", fontweight='bold')
    
    # --- Simulated Plots ---
    for i, sigma in enumerate(noise_levels):
        ax = axes[i+2]
        noise = np.random.randn(len(d1_chunk)) * sigma
        simulated = d1_chunk + noise
        
        ax.plot(time_axis, simulated, color='purple', alpha=0.7, label=f'D1 + σ={sigma}')
        ax.set_title(f"SIMULATION: D1 Augmented with σ={sigma} Noise", fontweight='bold')
        ax.legend(loc='upper right')

    # Apply Markers and Axis Formatting to all plots
    for ax in axes:
        ax.set_ylim(y_limit_min, y_limit_max)
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Draw Ground Truth Markers [cite: 45, 46, 47]
        for spk_idx, cls in zip(local_spikes, local_classes):
            ax.axvline(x=spk_idx, color='black', linestyle=':', alpha=0.4)
            if ax == axes[1]: # Only label the clean plot to avoid clutter
                ax.text(spk_idx, y_limit_max-2, f"C{cls}", ha='center', fontsize=10, fontweight='bold')

    plt.xlabel("Sample Index")
    print("Plotting complete. Compare the vertical 'thickness' of σ=2.5 to the Red D6 plot.")
    plt.show()

if __name__ == "__main__":
    visualize_noise_calibrated()