"""Exploratory analysis of D1.mat neural spike data."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter

# Load D1.mat
data = sio.loadmat('../datasets/D1.mat')
d = data['d'].flatten()  # Raw signal
index = data['Index'].flatten()  # Spike locations (1-indexed from MATLAB)
classes = data['Class'].flatten()  # Spike class labels (1-5)

# Print spike statistics
print(f"Total number of spikes: {len(index)}")
print("\nSpike counts per class:")
class_counts = Counter(classes)
for cls in sorted(class_counts.keys()):
    print(f"  Class {cls}: {class_counts[cls]} spikes ({100*class_counts[cls]/len(classes):.1f}%)")

# Extract spike waveforms (60 samples: 30 before, 30 after)
window_before = 30
window_after = 30
window_size = window_before + window_after

# Group waveforms by class
waveforms_by_class = {c: [] for c in range(1, 6)}

for i, (spike_idx, spike_class) in enumerate(zip(index, classes)):
    # Convert to 0-indexed
    spike_idx = int(spike_idx) - 1

    # Check bounds
    if spike_idx - window_before >= 0 and spike_idx + window_after <= len(d):
        waveform = d[spike_idx - window_before : spike_idx + window_after]
        waveforms_by_class[spike_class].append(waveform)

# Convert to arrays
for c in waveforms_by_class:
    waveforms_by_class[c] = np.array(waveforms_by_class[c])

# Plot 1: Average waveforms for each class
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
fig.suptitle('Average Spike Waveforms by Class', fontsize=14)

time_axis = np.arange(-window_before, window_after) / 25  # Convert to ms (25 kHz sample rate)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for c in range(1, 6):
    ax = axes[c-1]
    waveforms = waveforms_by_class[c]
    mean_waveform = np.mean(waveforms, axis=0)
    std_waveform = np.std(waveforms, axis=0)

    ax.fill_between(time_axis, mean_waveform - std_waveform, mean_waveform + std_waveform,
                    alpha=0.3, color=colors[c-1])
    ax.plot(time_axis, mean_waveform, color=colors[c-1], linewidth=2)
    ax.set_title(f'Class {c}\n(n={len(waveforms)})')
    ax.set_xlabel('Time (ms)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    if c == 1:
        ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('class_morphologies.png', dpi=150, bbox_inches='tight')
print("\nSaved: class_morphologies.png")

# Plot 2: First 0.5 seconds of raw signal with spike markers
sample_rate = 25000  # 25 kHz
duration_samples = int(0.5 * sample_rate)  # 0.5 seconds

fig, ax = plt.subplots(figsize=(14, 4))

time_ms = np.arange(duration_samples) / sample_rate * 1000  # Convert to ms
ax.plot(time_ms, d[:duration_samples], 'k-', linewidth=0.5, alpha=0.7)

# Overlay spike markers
for c in range(1, 6):
    # Find spikes in this time window for this class
    mask = (index <= duration_samples) & (classes == c)
    spike_times = index[mask]
    spike_amplitudes = d[(spike_times - 1).astype(int)]  # -1 for 0-indexing

    ax.scatter(spike_times / sample_rate * 1000, spike_amplitudes,
               c=colors[c-1], s=50, label=f'Class {c}', zorder=5, alpha=0.8)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('Raw Signal (First 0.5 seconds) with Spike Locations')
ax.legend(loc='upper right', ncol=5)
ax.set_xlim(0, 500)

plt.tight_layout()
plt.savefig('raw_signal_view.png', dpi=150, bbox_inches='tight')
print("Saved: raw_signal_view.png")

plt.show()
