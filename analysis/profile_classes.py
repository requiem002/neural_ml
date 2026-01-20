"""
Deep Forensic Analysis: Class Fingerprinting for Spike Classification

This script analyzes the ground truth (D1) to create detailed "fingerprints"
for each neuron class, specifically looking for subtle differences between
Class 4 and Class 5 (the "Twin Problem").

Physics-based features extracted:
1. Peak-to-Peak Amplitude
2. Pulse Width (FWHM - Full Width at Half Maximum)
3. Symmetry Ratio (Positive Peak vs Negative Trough)
4. Recovery Time (time to return to baseline)
5. Rise Time / Fall Time ratio
6. Second derivative features (curvature)
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_d1_data():
    """Load D1 dataset with ground truth labels."""
    data_dir = Path(__file__).parent.parent / 'datasets'
    data = sio.loadmat(data_dir / 'D1.mat')
    d = data['d'].flatten()
    index = data['Index'].flatten()
    classes = data['Class'].flatten()
    return d, index, classes


def extract_waveforms_by_class(d, index, classes, window_before=30, window_after=30):
    """Extract waveforms grouped by class."""
    class_waveforms = {c: [] for c in range(1, 6)}

    for idx, cls in zip(index, classes):
        idx_0 = int(idx) - 1  # Convert to 0-indexed
        start = idx_0 - window_before
        end = idx_0 + window_after

        if start >= 0 and end <= len(d):
            wf = d[start:end]
            class_waveforms[cls].append(wf)

    # Convert to arrays
    for c in range(1, 6):
        class_waveforms[c] = np.array(class_waveforms[c])

    return class_waveforms


def calculate_fwhm(waveform, sample_rate=25000):
    """
    Calculate Full Width at Half Maximum (FWHM) of the spike.
    Returns width in milliseconds.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    peak_idx = np.argmax(wf_centered)
    peak_val = wf_centered[peak_idx]

    half_max = peak_val / 2

    # Find crossing points
    above_half = wf_centered > half_max

    # Find first and last crossing
    crossings = np.where(np.diff(above_half.astype(int)))[0]

    if len(crossings) >= 2:
        # Interpolate for more accurate crossing points
        left_cross = crossings[0]
        right_cross = crossings[-1]

        # Linear interpolation for sub-sample accuracy
        if left_cross > 0:
            left_interp = left_cross + (half_max - wf_centered[left_cross]) / (wf_centered[left_cross + 1] - wf_centered[left_cross])
        else:
            left_interp = left_cross

        if right_cross < len(wf_centered) - 1:
            right_interp = right_cross + (half_max - wf_centered[right_cross]) / (wf_centered[right_cross + 1] - wf_centered[right_cross])
        else:
            right_interp = right_cross

        fwhm_samples = right_interp - left_interp
        fwhm_ms = fwhm_samples / sample_rate * 1000
        return fwhm_ms
    else:
        return np.nan


def calculate_symmetry_ratio(waveform):
    """
    Calculate symmetry ratio: |positive peak| / |negative trough|
    Values > 1 mean stronger positive peak, < 1 mean stronger negative trough.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    positive_peak = np.max(wf_centered)
    negative_trough = np.min(wf_centered)

    if abs(negative_trough) > 0.01:  # Avoid division by zero
        return abs(positive_peak) / abs(negative_trough)
    else:
        return np.nan


def calculate_recovery_time(waveform, sample_rate=25000, threshold_percent=10):
    """
    Calculate recovery time: time from peak to return to within threshold% of baseline.
    Returns time in milliseconds.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    peak_idx = np.argmax(np.abs(wf_centered))
    peak_val = np.abs(wf_centered[peak_idx])

    threshold = peak_val * (threshold_percent / 100)

    # Look for recovery after peak
    post_peak = np.abs(wf_centered[peak_idx:])

    recovery_idx = np.where(post_peak < threshold)[0]

    if len(recovery_idx) > 0:
        recovery_samples = recovery_idx[0]
        recovery_ms = recovery_samples / sample_rate * 1000
        return recovery_ms
    else:
        return np.nan


def calculate_rise_fall_ratio(waveform, sample_rate=25000):
    """
    Calculate ratio of rise time to fall time.
    Rise time: 10% to 90% of peak on rising edge
    Fall time: 90% to 10% of peak on falling edge
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    peak_idx = np.argmax(wf_centered)
    peak_val = wf_centered[peak_idx]

    if peak_val < 0.1:  # No clear peak
        return np.nan

    threshold_10 = 0.1 * peak_val
    threshold_90 = 0.9 * peak_val

    # Rising edge
    rising = wf_centered[:peak_idx + 1]
    rise_10 = np.where(rising > threshold_10)[0]
    rise_90 = np.where(rising > threshold_90)[0]

    if len(rise_10) > 0 and len(rise_90) > 0:
        rise_time = rise_90[0] - rise_10[0]
    else:
        rise_time = np.nan

    # Falling edge
    falling = wf_centered[peak_idx:]
    fall_90 = np.where(falling < threshold_90)[0]
    fall_10 = np.where(falling < threshold_10)[0]

    if len(fall_90) > 0 and len(fall_10) > 0:
        fall_time = fall_10[0] - fall_90[0]
    else:
        fall_time = np.nan

    if not np.isnan(rise_time) and not np.isnan(fall_time) and fall_time > 0:
        return rise_time / fall_time
    else:
        return np.nan


def calculate_curvature_at_peak(waveform):
    """
    Calculate second derivative (curvature) at the peak.
    This can reveal subtle shape differences.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    # Compute second derivative
    second_deriv = np.diff(wf_centered, n=2)

    peak_idx = np.argmax(wf_centered)

    # Get curvature around peak (mean of few samples around peak)
    if peak_idx >= 2 and peak_idx < len(second_deriv):
        start = max(0, peak_idx - 3)
        end = min(len(second_deriv), peak_idx + 3)
        curvature = np.mean(second_deriv[start:end])
        return curvature
    else:
        return np.nan


def calculate_area_ratio(waveform):
    """
    Calculate ratio of positive area to negative area under the curve.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    positive_area = np.sum(wf_centered[wf_centered > 0])
    negative_area = np.abs(np.sum(wf_centered[wf_centered < 0]))

    if negative_area > 0.01:
        return positive_area / negative_area
    else:
        return np.nan


def calculate_repolarization_slope(waveform, sample_rate=25000):
    """
    Calculate the slope during repolarization (after peak).
    This is often different between neuron types.
    """
    baseline = np.mean(waveform[:5])
    wf_centered = waveform - baseline

    peak_idx = np.argmax(wf_centered)

    # Look at the falling phase (from peak to trough)
    if peak_idx < len(wf_centered) - 5:
        trough_idx = peak_idx + np.argmin(wf_centered[peak_idx:peak_idx + 20])

        if trough_idx > peak_idx:
            # Linear fit to falling phase
            x = np.arange(trough_idx - peak_idx)
            y = wf_centered[peak_idx:trough_idx]

            if len(x) > 2:
                slope = np.polyfit(x, y, 1)[0]
                return slope * sample_rate / 1000  # Convert to V/ms

    return np.nan


def analyze_class(waveforms, class_num):
    """Compute all features for a single class."""
    n_spikes = len(waveforms)

    # Initialize feature arrays
    amplitudes = []
    fwhm_values = []
    symmetry_ratios = []
    recovery_times = []
    rise_fall_ratios = []
    curvatures = []
    area_ratios = []
    repol_slopes = []

    for wf in waveforms:
        # Peak-to-Peak Amplitude
        baseline = np.mean(wf[:5])
        wf_centered = wf - baseline
        amp = np.max(wf_centered) - np.min(wf_centered)
        amplitudes.append(amp)

        # FWHM
        fwhm_values.append(calculate_fwhm(wf))

        # Symmetry Ratio
        symmetry_ratios.append(calculate_symmetry_ratio(wf))

        # Recovery Time
        recovery_times.append(calculate_recovery_time(wf))

        # Rise/Fall Ratio
        rise_fall_ratios.append(calculate_rise_fall_ratio(wf))

        # Curvature at Peak
        curvatures.append(calculate_curvature_at_peak(wf))

        # Area Ratio
        area_ratios.append(calculate_area_ratio(wf))

        # Repolarization Slope
        repol_slopes.append(calculate_repolarization_slope(wf))

    # Compute statistics
    features = {
        'class': class_num,
        'count': n_spikes,
        'amplitude_mean': np.nanmean(amplitudes),
        'amplitude_std': np.nanstd(amplitudes),
        'fwhm_mean': np.nanmean(fwhm_values),
        'fwhm_std': np.nanstd(fwhm_values),
        'symmetry_mean': np.nanmean(symmetry_ratios),
        'symmetry_std': np.nanstd(symmetry_ratios),
        'recovery_mean': np.nanmean(recovery_times),
        'recovery_std': np.nanstd(recovery_times),
        'rise_fall_mean': np.nanmean(rise_fall_ratios),
        'rise_fall_std': np.nanstd(rise_fall_ratios),
        'curvature_mean': np.nanmean(curvatures),
        'curvature_std': np.nanstd(curvatures),
        'area_ratio_mean': np.nanmean(area_ratios),
        'area_ratio_std': np.nanstd(area_ratios),
        'repol_slope_mean': np.nanmean(repol_slopes),
        'repol_slope_std': np.nanstd(repol_slopes),
    }

    return features


def create_fingerprint_plot(class_waveforms, output_path):
    """Create visualization of class fingerprints."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Color scheme
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

    # Time axis (in ms, assuming 25kHz sampling)
    time_ms = np.arange(60) / 25  # 60 samples at 25kHz

    # Plot 1: All templates overlaid (raw)
    ax = axes[0, 0]
    for c in range(1, 6):
        template = np.mean(class_waveforms[c], axis=0)
        baseline = np.mean(template[:5])
        ax.plot(time_ms, template - baseline, color=colors[c], linewidth=2, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Raw Average Waveforms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Normalized templates (shape comparison)
    ax = axes[0, 1]
    for c in range(1, 6):
        template = np.mean(class_waveforms[c], axis=0)
        baseline = np.mean(template[:5])
        template_centered = template - baseline
        peak = np.max(np.abs(template_centered))
        template_norm = template_centered / peak
        ax.plot(time_ms, template_norm, color=colors[c], linewidth=2, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Voltage')
    ax.set_title('Normalized Waveforms (Shape Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Class 4 vs Class 5 detailed comparison
    ax = axes[0, 2]
    for c in [4, 5]:
        template = np.mean(class_waveforms[c], axis=0)
        baseline = np.mean(template[:5])
        template_centered = template - baseline
        peak = np.max(np.abs(template_centered))
        template_norm = template_centered / peak
        ax.plot(time_ms, template_norm, color=colors[c], linewidth=2.5, label=f'Class {c}')

    # Add difference
    t4 = np.mean(class_waveforms[4], axis=0)
    t5 = np.mean(class_waveforms[5], axis=0)
    t4_norm = (t4 - np.mean(t4[:5])) / np.max(np.abs(t4 - np.mean(t4[:5])))
    t5_norm = (t5 - np.mean(t5[:5])) / np.max(np.abs(t5 - np.mean(t5[:5])))
    diff = t4_norm - t5_norm
    ax.plot(time_ms, diff * 5, 'k--', linewidth=1.5, alpha=0.7, label='Difference (×5)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Voltage')
    ax.set_title('Class 4 vs Class 5 (The "Twin Problem")')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: First derivative (velocity)
    ax = axes[1, 0]
    for c in range(1, 6):
        template = np.mean(class_waveforms[c], axis=0)
        baseline = np.mean(template[:5])
        template_centered = template - baseline
        peak = np.max(np.abs(template_centered))
        template_norm = template_centered / peak
        deriv = np.diff(template_norm) * 25  # Scale by sample rate
        ax.plot(time_ms[:-1], deriv, color=colors[c], linewidth=1.5, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('dV/dt (normalized)')
    ax.set_title('First Derivative (Velocity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Second derivative (acceleration/curvature)
    ax = axes[1, 1]
    for c in range(1, 6):
        template = np.mean(class_waveforms[c], axis=0)
        baseline = np.mean(template[:5])
        template_centered = template - baseline
        peak = np.max(np.abs(template_centered))
        template_norm = template_centered / peak
        deriv2 = np.diff(template_norm, n=2) * 625  # Scale by sample rate squared
        ax.plot(time_ms[:-2], deriv2, color=colors[c], linewidth=1.5, label=f'Class {c}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('d²V/dt² (normalized)')
    ax.set_title('Second Derivative (Curvature)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Amplitude distribution by class
    ax = axes[1, 2]
    for c in range(1, 6):
        wfs = class_waveforms[c]
        amplitudes = []
        for wf in wfs:
            baseline = np.mean(wf[:5])
            amp = np.max(wf - baseline) - np.min(wf - baseline)
            amplitudes.append(amp)
        ax.hist(amplitudes, bins=30, alpha=0.5, color=colors[c], label=f'Class {c}')
    ax.set_xlabel('Peak-to-Peak Amplitude (V)')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")


def print_fingerprint_table(all_features):
    """Print formatted table of class fingerprints."""
    print("\n" + "=" * 120)
    print("CLASS FINGERPRINT TABLE - Deep Forensic Analysis")
    print("=" * 120)

    # Header
    print(f"\n{'Feature':<25} {'Class 1':<18} {'Class 2':<18} {'Class 3':<18} {'Class 4':<18} {'Class 5':<18}")
    print("-" * 120)

    # Count
    print(f"{'Spike Count':<25}", end="")
    for c in range(1, 6):
        print(f"{all_features[c]['count']:<18}", end="")
    print()

    # Amplitude
    print(f"{'Peak-to-Peak Amp (V)':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['amplitude_mean']:.3f} ± {all_features[c]['amplitude_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # FWHM
    print(f"{'FWHM (ms)':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['fwhm_mean']:.3f} ± {all_features[c]['fwhm_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Symmetry Ratio
    print(f"{'Symmetry Ratio':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['symmetry_mean']:.3f} ± {all_features[c]['symmetry_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Recovery Time
    print(f"{'Recovery Time (ms)':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['recovery_mean']:.3f} ± {all_features[c]['recovery_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Rise/Fall Ratio
    print(f"{'Rise/Fall Ratio':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['rise_fall_mean']:.3f} ± {all_features[c]['rise_fall_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Curvature at Peak
    print(f"{'Peak Curvature':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['curvature_mean']:.3f} ± {all_features[c]['curvature_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Area Ratio
    print(f"{'Area Ratio (+/-)':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['area_ratio_mean']:.3f} ± {all_features[c]['area_ratio_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    # Repolarization Slope
    print(f"{'Repol. Slope (V/ms)':<25}", end="")
    for c in range(1, 6):
        val = f"{all_features[c]['repol_slope_mean']:.3f} ± {all_features[c]['repol_slope_std']:.3f}"
        print(f"{val:<18}", end="")
    print()

    print("-" * 120)

    # Special analysis: Class 4 vs Class 5
    print("\n" + "=" * 80)
    print("TWIN PROBLEM ANALYSIS: Class 4 vs Class 5")
    print("=" * 80)

    for feature_name, key in [
        ('Peak-to-Peak Amplitude', 'amplitude_mean'),
        ('FWHM', 'fwhm_mean'),
        ('Symmetry Ratio', 'symmetry_mean'),
        ('Recovery Time', 'recovery_mean'),
        ('Rise/Fall Ratio', 'rise_fall_mean'),
        ('Peak Curvature', 'curvature_mean'),
        ('Area Ratio', 'area_ratio_mean'),
        ('Repolarization Slope', 'repol_slope_mean'),
    ]:
        val4 = all_features[4][key]
        val5 = all_features[5][key]
        diff = abs(val4 - val5)
        avg = (abs(val4) + abs(val5)) / 2
        pct_diff = (diff / avg * 100) if avg > 0.001 else 0

        distinguishable = "YES" if pct_diff > 10 else "MAYBE" if pct_diff > 5 else "NO"

        print(f"{feature_name:<25}: Class4={val4:>8.4f}, Class5={val5:>8.4f}, "
              f"Diff={pct_diff:>5.1f}%, Distinguishable: {distinguishable}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Determine key distinguishing features
    amp4 = all_features[4]['amplitude_mean']
    amp5 = all_features[5]['amplitude_mean']
    amp_diff = abs(amp4 - amp5) / ((amp4 + amp5) / 2) * 100

    fwhm4 = all_features[4]['fwhm_mean']
    fwhm5 = all_features[5]['fwhm_mean']
    fwhm_diff = abs(fwhm4 - fwhm5) / ((fwhm4 + fwhm5) / 2) * 100

    sym4 = all_features[4]['symmetry_mean']
    sym5 = all_features[5]['symmetry_mean']
    sym_diff = abs(sym4 - sym5) / ((sym4 + sym5) / 2) * 100

    print(f"\n1. AMPLITUDE: Class 4 ({amp4:.3f}V) vs Class 5 ({amp5:.3f}V) = {amp_diff:.1f}% difference")
    if amp_diff > 20:
        print("   → Amplitude IS a distinguishing feature (requires accurate amplitude measurement)")

    print(f"\n2. PULSE WIDTH (FWHM): Class 4 ({fwhm4:.3f}ms) vs Class 5 ({fwhm5:.3f}ms) = {fwhm_diff:.1f}% difference")
    if fwhm_diff > 5:
        print("   → Pulse width MAY be a subtle distinguishing feature")

    print(f"\n3. SYMMETRY: Class 4 ({sym4:.3f}) vs Class 5 ({sym5:.3f}) = {sym_diff:.1f}% difference")
    if sym_diff > 5:
        print("   → Symmetry MAY be a subtle distinguishing feature")

    repol4 = all_features[4]['repol_slope_mean']
    repol5 = all_features[5]['repol_slope_mean']
    repol_diff = abs(repol4 - repol5) / ((abs(repol4) + abs(repol5)) / 2) * 100

    print(f"\n4. REPOLARIZATION SLOPE: Class 4 ({repol4:.3f}V/ms) vs Class 5 ({repol5:.3f}V/ms) = {repol_diff:.1f}% difference")
    if repol_diff > 10:
        print("   → Repolarization slope IS a distinguishing feature")

    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR CNN:")
    print("=" * 80)
    print("""
    Based on this analysis, a CNN should focus on:
    1. Raw waveform shape (to capture subtle timing differences)
    2. Derivative features (rise/fall rates)
    3. Multi-scale analysis (different filter sizes to capture different timescales)
    4. Amplitude as a SEPARATE input channel (not normalized away)

    The "Twin Problem" (Class 4 vs 5) may be solvable if the CNN can learn:
    - Subtle differences in repolarization dynamics
    - Amplitude ratios
    - Timing differences in rise/fall phases
    """)


def main():
    print("=" * 80)
    print("DEEP FORENSIC ANALYSIS: Class Fingerprinting")
    print("=" * 80)

    # Load data
    print("\nLoading D1 ground truth data...")
    d, index, classes = load_d1_data()
    print(f"Loaded {len(index)} labeled spikes")

    # Extract waveforms by class
    print("\nExtracting waveforms by class...")
    class_waveforms = extract_waveforms_by_class(d, index, classes)

    for c in range(1, 6):
        print(f"  Class {c}: {len(class_waveforms[c])} waveforms")

    # Analyze each class
    print("\nComputing feature fingerprints...")
    all_features = {}
    for c in range(1, 6):
        print(f"  Analyzing Class {c}...")
        all_features[c] = analyze_class(class_waveforms[c], c)

    # Create visualization
    output_dir = Path(__file__).parent
    plot_path = output_dir / 'class_fingerprints.png'
    print(f"\nGenerating fingerprint plot...")
    create_fingerprint_plot(class_waveforms, plot_path)

    # Print table
    print_fingerprint_table(all_features)

    return all_features, class_waveforms


if __name__ == '__main__':
    main()
