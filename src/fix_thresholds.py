"""Investigate and fix threshold problem across datasets."""

import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'datasets'
analysis_dir = Path(__file__).parent.parent / 'analysis'

# Load all datasets
datasets = {}
for name in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
    data = sio.loadmat(data_dir / f'{name}.mat')
    datasets[name] = data['d'].flatten()

# D1 ground truth
d1_data = sio.loadmat(data_dir / 'D1.mat')
d1_index = d1_data['Index'].flatten()
d1_class = d1_data['Class'].flatten()

# Bandpass filter function
def bandpass_filter(data, sample_rate=25000):
    nyquist = sample_rate / 2
    low = 300 / nyquist
    high = 3000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

print("="*70)
print("ANALYSIS: Spike Amplitudes vs Noise Levels")
print("="*70)

# Analyze each dataset
results = {}
for name, d in datasets.items():
    filtered = bandpass_filter(d)

    # Noise estimation
    mad = np.median(np.abs(filtered - np.median(filtered)))
    sigma = mad / 0.6745

    results[name] = {
        'sigma': sigma,
        'filtered': filtered
    }

print("\n--- Noise Levels (σ) ---")
for name in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
    print(f"{name}: σ = {results[name]['sigma']:.4f}")

# Get D1 spike amplitudes (ground truth)
d1_filtered = results['D1']['filtered']
d1_spike_amplitudes = []
for idx in d1_index:
    idx_0 = int(idx) - 1
    start = max(0, idx_0 - 10)
    end = min(len(d1_filtered), idx_0 + 10)
    local_max = np.max(d1_filtered[start:end])
    d1_spike_amplitudes.append(local_max)

d1_spike_amplitudes = np.array(d1_spike_amplitudes)

print("\n--- D1 Ground Truth Spike Amplitudes ---")
print(f"Min: {np.min(d1_spike_amplitudes):.4f}")
print(f"5th percentile: {np.percentile(d1_spike_amplitudes, 5):.4f}")
print(f"10th percentile: {np.percentile(d1_spike_amplitudes, 10):.4f}")
print(f"Median: {np.median(d1_spike_amplitudes):.4f}")
print(f"Mean: {np.mean(d1_spike_amplitudes):.4f}")
print(f"Max: {np.max(d1_spike_amplitudes):.4f}")

# THE KEY INSIGHT: Use a FIXED voltage threshold based on D1's spike distribution
# If we want to catch 95% of spikes, use the 5th percentile as threshold
fixed_threshold_95 = np.percentile(d1_spike_amplitudes, 5)
fixed_threshold_90 = np.percentile(d1_spike_amplitudes, 10)

print(f"\n--- Recommended Fixed Thresholds ---")
print(f"To catch 95% of spikes: {fixed_threshold_95:.4f}")
print(f"To catch 90% of spikes: {fixed_threshold_90:.4f}")

# What σ multiplier would this be for each dataset?
print(f"\n--- Equivalent σ Multipliers ---")
print(f"{'Dataset':<8} {'σ':<10} {'95% thresh':<12} {'90% thresh':<12}")
for name in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
    sigma = results[name]['sigma']
    mult_95 = fixed_threshold_95 / sigma
    mult_90 = fixed_threshold_90 / sigma
    print(f"{name:<8} {sigma:<10.4f} {mult_95:<12.2f} {mult_90:<12.2f}")

# Now let's test detection with fixed threshold on D1 and D2
print("\n" + "="*70)
print("TESTING: Fixed Threshold Detection")
print("="*70)

def detect_with_threshold(data_filtered, voltage_threshold, min_distance=30):
    """Detect spikes using fixed voltage threshold."""
    peaks, _ = signal.find_peaks(
        data_filtered,
        height=voltage_threshold,
        distance=min_distance,
        prominence=voltage_threshold * 0.3
    )
    return peaks

# Test on D1
print("\n--- D1 Detection Test ---")
for thresh_pct, thresh_val in [('5th pct', fixed_threshold_95), ('10th pct', fixed_threshold_90)]:
    peaks = detect_with_threshold(d1_filtered, thresh_val)

    # Match to ground truth
    matched = 0
    used_gt = set()
    for peak in peaks:
        for i, gt_idx in enumerate(d1_index):
            if i not in used_gt and abs(peak + 1 - gt_idx) <= 50:
                matched += 1
                used_gt.add(i)
                break

    fp = len(peaks) - matched
    fn = len(d1_index) - matched
    precision = matched / len(peaks) if len(peaks) > 0 else 0
    recall = matched / len(d1_index)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Threshold {thresh_pct} ({thresh_val:.4f}): {len(peaks)} detected, "
          f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

# Test on D2
print("\n--- D2 Detection Test ---")
d2_filtered = results['D2']['filtered']
for thresh_pct, thresh_val in [('5th pct', fixed_threshold_95), ('10th pct', fixed_threshold_90)]:
    peaks = detect_with_threshold(d2_filtered, thresh_val)
    print(f"Threshold {thresh_pct} ({thresh_val:.4f}): {len(peaks)} detected")

# Let's try even lower thresholds
print("\n--- Testing Various Fixed Thresholds on D2 ---")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    peaks = detect_with_threshold(d2_filtered, thresh)
    print(f"Threshold {thresh:.2f}: {len(peaks)} detected")

# Plot spike amplitude distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, name in enumerate(['D1', 'D2', 'D3', 'D4', 'D5', 'D6']):
    ax = axes[i]
    filtered = results[name]['filtered']
    sigma = results[name]['sigma']

    # Find all peaks above a low threshold
    all_peaks, props = signal.find_peaks(filtered, height=0.3, distance=30)
    peak_heights = props['peak_heights']

    ax.hist(peak_heights, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(fixed_threshold_95, color='r', linestyle='--', label=f'5th pct: {fixed_threshold_95:.2f}')
    ax.axvline(fixed_threshold_90, color='orange', linestyle='--', label=f'10th pct: {fixed_threshold_90:.2f}')
    ax.axvline(sigma * 5, color='green', linestyle=':', label=f'5σ: {sigma*5:.2f}')
    ax.set_title(f'{name} (σ={sigma:.3f})')
    ax.set_xlabel('Peak Height')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # Count peaks above each threshold
    above_5pct = np.sum(peak_heights >= fixed_threshold_95)
    above_10pct = np.sum(peak_heights >= fixed_threshold_90)
    ax.text(0.95, 0.95, f'≥5th: {above_5pct}\n≥10th: {above_10pct}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(analysis_dir / 'peak_distributions.png', dpi=150)
print(f"\nSaved: {analysis_dir / 'peak_distributions.png'}")

# Final recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
The problem: Using σ-based thresholds fails because spike amplitudes are
constant but noise increases. A 6.5σ threshold in D2 = 1.50V, which is
HIGHER than many real spikes!

Solution: Use a FIXED VOLTAGE THRESHOLD derived from D1's spike distribution.
- 5th percentile of D1 spikes = {:.4f}V (catches 95% of D1 spikes)
- 10th percentile of D1 spikes = {:.4f}V (catches 90% of D1 spikes)

For noisier datasets, we may need to go even lower and accept more false
positives, relying on the classifier to filter them out.
""".format(fixed_threshold_95, fixed_threshold_90))

plt.show()
