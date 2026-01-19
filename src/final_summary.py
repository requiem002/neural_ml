"""Generate final summary of all predictions."""

import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import Counter

pred_dir = Path(__file__).parent.parent / 'predictions'
data_dir = Path(__file__).parent.parent / 'datasets'

print("="*90)
print("FINAL PREDICTION SUMMARY")
print("="*90)

# Header
print(f"\n{'Dataset':<10} {'Total':<10} {'Class 1':<10} {'Class 2':<10} {'Class 3':<10} {'Class 4':<10} {'Class 5':<10}")
print("-"*90)

# Also include D1 for reference
d1_data = sio.loadmat(data_dir / 'D1.mat')
d1_class = d1_data['Class'].flatten()
d1_counts = Counter(d1_class)
print(f"{'D1 (GT)':<10} {len(d1_class):<10} {d1_counts[1]:<10} {d1_counts[2]:<10} {d1_counts[3]:<10} {d1_counts[4]:<10} {d1_counts[5]:<10}")

print("-"*90)

# Predictions
for name in ['D2', 'D3', 'D4', 'D5', 'D6']:
    data = sio.loadmat(pred_dir / f'{name}_predictions.mat')
    classes = data['Class'].flatten()
    counts = Counter(classes)

    total = len(classes)
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    c3 = counts.get(3, 0)
    c4 = counts.get(4, 0)
    c5 = counts.get(5, 0)

    print(f"{name:<10} {total:<10} {c1:<10} {c2:<10} {c3:<10} {c4:<10} {c5:<10}")

print("-"*90)

# Print percentages
print(f"\n{'Dataset':<10} {'Total':<10} {'Class 1':<10} {'Class 2':<10} {'Class 3':<10} {'Class 4':<10} {'Class 5':<10}")
print("-"*90)

# D1 percentages
total = len(d1_class)
print(f"{'D1 (GT)':<10} {'100%':<10} {100*d1_counts[1]/total:<9.1f}% {100*d1_counts[2]/total:<9.1f}% {100*d1_counts[3]/total:<9.1f}% {100*d1_counts[4]/total:<9.1f}% {100*d1_counts[5]/total:<9.1f}%")
print("-"*90)

for name in ['D2', 'D3', 'D4', 'D5', 'D6']:
    data = sio.loadmat(pred_dir / f'{name}_predictions.mat')
    classes = data['Class'].flatten()
    counts = Counter(classes)

    total = len(classes)
    c1 = 100 * counts.get(1, 0) / total
    c2 = 100 * counts.get(2, 0) / total
    c3 = 100 * counts.get(3, 0) / total
    c4 = 100 * counts.get(4, 0) / total
    c5 = 100 * counts.get(5, 0) / total

    print(f"{name:<10} {'100%':<10} {c1:<9.1f}% {c2:<9.1f}% {c3:<9.1f}% {c4:<9.1f}% {c5:<9.1f}%")

print("="*90)

print("""
NOTES:
- D1 Ground Truth: 2,176 spikes with balanced classes (~20% each)
- D2 prediction: 4,086 spikes (peer benchmark: ~3,700)
- Class 3 tends to be over-predicted in noisy datasets because:
  1. Class 3 has the smallest amplitude spikes
  2. Noisy waveforms often look like low-amplitude spikes
- Higher SNR (D2, D3) have more balanced distributions
- Lower SNR (D5, D6) have more Class 3 bias due to noise
""")
