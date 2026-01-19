"""Verify predictions for D2-D6 datasets."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
data_dir = Path(__file__).parent.parent / 'datasets'
pred_dir = Path(__file__).parent.parent / 'predictions'
analysis_dir = Path(__file__).parent.parent / 'analysis'
analysis_dir.mkdir(exist_ok=True)

# Datasets to verify
datasets = ['D2', 'D3', 'D4', 'D5', 'D6']

# Collect summary data
print("\n" + "="*80)
print(f"{'Dataset':<10} | {'Spikes':<8} | {'Class 1':>8} | {'Class 2':>8} | {'Class 3':>8} | {'Class 4':>8} | {'Class 5':>8}")
print("="*80)

# Colors for each class
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure with 5 subplots
fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
fig.suptitle('Predictions Overlay: First 1.0 Second of Raw Signal', fontsize=14, fontweight='bold')

sample_rate = 25000
samples_to_plot = int(1.0 * sample_rate)  # 1.0 second = 25000 samples

for i, dataset in enumerate(datasets):
    # Load raw signal
    raw_data = sio.loadmat(data_dir / f'{dataset}.mat')
    d = raw_data['d'].flatten()

    # Load predictions
    pred_data = sio.loadmat(pred_dir / f'{dataset}_predictions.mat')
    index = pred_data['Index'].flatten()
    classes = pred_data['Class'].flatten()

    # Count spikes per class
    class_counts = {c: np.sum(classes == c) for c in range(1, 6)}
    total_spikes = len(index)

    # Print summary row
    print(f"{dataset:<10} | {total_spikes:<8} | {class_counts[1]:>8} | {class_counts[2]:>8} | {class_counts[3]:>8} | {class_counts[4]:>8} | {class_counts[5]:>8}")

    # Plot raw signal (first 1.0 second)
    ax = axes[i]
    time_ms = np.arange(samples_to_plot) / sample_rate * 1000  # Convert to ms
    ax.plot(time_ms, d[:samples_to_plot], 'k-', linewidth=0.5, alpha=0.7)

    # Overlay detected spikes colored by class
    for c in range(1, 6):
        # Find spikes of this class within the time window
        mask = (index <= samples_to_plot) & (classes == c)
        spike_times = index[mask]

        if len(spike_times) > 0:
            spike_amplitudes = d[(spike_times - 1).astype(int)]  # -1 for 0-indexing
            ax.scatter(spike_times / sample_rate * 1000, spike_amplitudes,
                      c=colors[c-1], s=40, label=f'Class {c}', zorder=5, alpha=0.8)

    ax.set_ylabel(f'{dataset}\nAmplitude')
    ax.set_xlim(0, 1000)

    # Add legend only to first subplot
    if i == 0:
        ax.legend(loc='upper right', ncol=5, fontsize=8)

axes[-1].set_xlabel('Time (ms)')

plt.tight_layout()
plt.savefig(analysis_dir / 'predictions_overlay.png', dpi=150, bbox_inches='tight')
print("="*80)
print(f"\nSaved: {analysis_dir / 'predictions_overlay.png'}")

plt.show()
