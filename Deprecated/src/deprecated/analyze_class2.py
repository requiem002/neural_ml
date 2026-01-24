"""Forensic analysis of Class 2 collapse issue."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load D1 training data
    data_dir = Path(__file__).parent.parent / 'datasets'
    data = sio.loadmat(data_dir / 'D1.mat')
    d = data['d'].flatten()
    index = data['Index'].flatten()
    classes = data['Class'].flatten()

    # Extract waveforms for each class
    window_before = 30
    window_after = 30

    class_waveforms = {c: [] for c in range(1, 6)}
    class_amplitudes = {c: [] for c in range(1, 6)}

    for idx, cls in zip(index, classes):
        idx_0 = int(idx) - 1
        start = idx_0 - window_before
        end = idx_0 + window_after

        if start >= 0 and end <= len(d):
            wf = d[start:end]
            class_waveforms[cls].append(wf)

            # Calculate amplitude (peak - baseline)
            baseline = np.mean(wf[:5])
            peak = np.max(wf)
            amplitude = peak - baseline
            class_amplitudes[cls].append(amplitude)

    # Convert to arrays
    for c in range(1, 6):
        class_waveforms[c] = np.array(class_waveforms[c])
        class_amplitudes[c] = np.array(class_amplitudes[c])

    # Print amplitude statistics
    print("=" * 60)
    print("AMPLITUDE ANALYSIS BY CLASS (D1 Ground Truth)")
    print("=" * 60)
    print(f"\n{'Class':<10} {'Count':<10} {'Mean Amp':<12} {'Std Amp':<12} {'Min Amp':<12} {'Max Amp':<12}")
    print("-" * 60)

    for c in range(1, 6):
        amps = class_amplitudes[c]
        print(f"{c:<10} {len(amps):<10} {np.mean(amps):<12.3f} {np.std(amps):<12.3f} {np.min(amps):<12.3f} {np.max(amps):<12.3f}")

    # Compute correlation between class templates (shape similarity)
    print("\n" + "=" * 60)
    print("TEMPLATE SHAPE SIMILARITY (Correlation after normalization)")
    print("=" * 60)

    templates = {}
    templates_normalized = {}

    for c in range(1, 6):
        templates[c] = np.mean(class_waveforms[c], axis=0)
        # Normalize to unit peak (same as what we do in feature extraction)
        baseline = np.mean(templates[c][:5])
        centered = templates[c] - baseline
        peak = np.max(np.abs(centered))
        templates_normalized[c] = centered / peak if peak > 0 else centered

    print("\nCorrelation matrix (normalized templates):")
    print(f"{'':>8}", end="")
    for c in range(1, 6):
        print(f"Class {c:>3}", end="  ")
    print()

    for c1 in range(1, 6):
        print(f"Class {c1}:", end="")
        for c2 in range(1, 6):
            corr = np.corrcoef(templates_normalized[c1], templates_normalized[c2])[0, 1]
            print(f"  {corr:>6.3f}", end="")
        print()

    # Check specifically Class 1 vs Class 2
    print("\n" + "=" * 60)
    print("CLASS 1 vs CLASS 2 COMPARISON")
    print("=" * 60)

    amp1 = class_amplitudes[1]
    amp2 = class_amplitudes[2]

    print(f"\nClass 1 amplitude: {np.mean(amp1):.3f} ± {np.std(amp1):.3f}")
    print(f"Class 2 amplitude: {np.mean(amp2):.3f} ± {np.std(amp2):.3f}")
    print(f"Amplitude ratio (Class2/Class1): {np.mean(amp2)/np.mean(amp1):.3f}")

    corr_12 = np.corrcoef(templates_normalized[1], templates_normalized[2])[0, 1]
    print(f"Shape correlation (normalized): {corr_12:.4f}")

    # Plot templates
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Plot all class templates
    ax = axes[0, 0]
    for c in range(1, 6):
        ax.plot(templates[c], label=f'Class {c}')
    ax.set_title('Raw Templates (with amplitude)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot normalized templates
    ax = axes[0, 1]
    for c in range(1, 6):
        ax.plot(templates_normalized[c], label=f'Class {c}')
    ax.set_title('Normalized Templates (unit peak)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Normalized Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amplitude distribution
    ax = axes[0, 2]
    for c in range(1, 6):
        ax.hist(class_amplitudes[c], bins=30, alpha=0.5, label=f'Class {c}')
    ax.set_title('Amplitude Distribution by Class')
    ax.set_xlabel('Peak Amplitude')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Class 1 vs Class 2 templates
    ax = axes[1, 0]
    ax.plot(templates[1], 'b-', label='Class 1', linewidth=2)
    ax.plot(templates[2], 'r--', label='Class 2', linewidth=2)
    ax.set_title('Class 1 vs Class 2 (Raw)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Class 1 vs Class 2 normalized
    ax = axes[1, 1]
    ax.plot(templates_normalized[1], 'b-', label='Class 1', linewidth=2)
    ax.plot(templates_normalized[2], 'r--', label='Class 2', linewidth=2)
    ax.set_title('Class 1 vs Class 2 (Normalized)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Normalized Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference between Class 1 and Class 2
    ax = axes[1, 2]
    ax.plot(templates[1] - templates[2], 'g-', label='Raw difference', linewidth=2)
    ax.plot(templates_normalized[1] - templates_normalized[2], 'm--', label='Normalized difference', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Class 1 - Class 2 Difference')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'class2_analysis.png', dpi=150)
    print(f"\nPlot saved to: class2_analysis.png")

    # Key finding summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    if np.mean(amp2) < np.mean(amp1) and corr_12 > 0.9:
        print("\n⚠️  CONFIRMED: Class 2 appears to be a 'smaller amplitude Class 1'")
        print(f"   - Class 2 has {100*(1-np.mean(amp2)/np.mean(amp1)):.1f}% smaller amplitude than Class 1")
        print(f"   - Shape correlation is {corr_12:.3f} (very similar shapes)")
        print("\n   DIAGNOSIS: Normalizing waveforms destroys the amplitude")
        print("   information that distinguishes Class 2 from Class 1!")

    print()

if __name__ == '__main__':
    main()
