#!/usr/bin/env python3
"""
FINAL SUBMISSION VALIDATION SCRIPT
==================================

Comprehensive validation of submission files for EE40098 Coursework C.

This script performs:
1. Format compliance check (1xN row vectors, uppercase filenames)
2. Class distribution audit vs D1 "Golden Ratio"
3. Spike raster plots (temporal distribution)
4. Inter-Spike Interval (ISI) violation analysis
5. PCA cluster visualization
6. Sanity checks (uniqueness, ordering, bounds)

Usage:
    python src/validate_submission.py           # Full validation
    python src/validate_submission.py --fix     # Fix format issues first
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import Counter
import argparse
import os

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / 'datasets'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'
ANALYSIS_DIR = BASE_DIR / 'analysis'

# D1 "Golden Ratio" - target class distribution
D1_DISTRIBUTION = {
    'total': 2176,
    1: 457, 2: 442, 3: 407, 4: 444, 5: 426
}
D1_PERCENTAGES = {c: 100 * cnt / D1_DISTRIBUTION['total'] for c, cnt in D1_DISTRIBUTION.items() if c != 'total'}

# Signal parameters
SAMPLE_RATE = 25000  # 25 kHz
TOTAL_SAMPLES = 1_440_000
REFRACTORY_PERIOD_MS = 2.0
REFRACTORY_SAMPLES = int(REFRACTORY_PERIOD_MS * SAMPLE_RATE / 1000)  # 50 samples


def print_header(title):
    """Print formatted section header."""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subheader(title):
    """Print formatted subsection header."""
    print()
    print(f"─── {title} ───")


def fix_submission_format():
    """
    Fix submission files to match D1.mat format:
    - Convert from Nx1 (column) to 1xN (row) vectors
    - Rename files to uppercase (d2.mat -> D2.mat)
    - Ensure correct dtypes (int32 for Index, uint8 for Class)
    """
    print_header("FIXING SUBMISSION FORMAT")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    fixed_count = 0

    for i in range(2, 7):
        lowercase_file = SUBMISSIONS_DIR / f'd{i}.mat'
        uppercase_file = SUBMISSIONS_DIR / f'D{i}.mat'

        # Determine source file
        if lowercase_file.exists():
            source_file = lowercase_file
        elif uppercase_file.exists():
            source_file = uppercase_file
        else:
            print(f"  ✗ D{i}.mat: Source file not found!")
            continue

        try:
            # Load data
            data = sio.loadmat(str(source_file))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

            # Convert to row vectors with correct dtypes
            indices_row = indices.reshape(1, -1).astype(np.int32)
            classes_row = classes.reshape(1, -1).astype(np.uint8)

            # Save with uppercase filename
            sio.savemat(str(uppercase_file), {
                'Index': indices_row,
                'Class': classes_row
            }, do_compression=False)

            # Remove lowercase file if different
            if lowercase_file.exists() and lowercase_file != uppercase_file:
                lowercase_file.unlink()

            print(f"  ✓ D{i}.mat: Fixed to (1, {len(indices)}) row vectors")
            fixed_count += 1

        except Exception as e:
            print(f"  ✗ D{i}.mat: Error - {e}")

    print(f"\n  Fixed {fixed_count}/5 files")
    return fixed_count == 5


def verify_format_compliance():
    """Verify all submission files match D1.mat format exactly."""
    print_header("[1] FORMAT COMPLIANCE CHECK")

    # Load D1 as reference
    d1_data = sio.loadmat(str(DATASETS_DIR / 'D1.mat'))
    ref_index_shape_type = f"(1, N) {d1_data['Index'].dtype}"
    ref_class_shape_type = f"(1, N) {d1_data['Class'].dtype}"

    print(f"  Reference (D1.mat):")
    print(f"    Index: shape (1, 2176), dtype {d1_data['Index'].dtype}")
    print(f"    Class: shape (1, 2176), dtype {d1_data['Class'].dtype}")
    print()

    all_compliant = True
    results = {}

    for i in range(2, 7):
        filepath = SUBMISSIONS_DIR / f'D{i}.mat'

        if not filepath.exists():
            print(f"  ✗ D{i}.mat: FILE NOT FOUND")
            all_compliant = False
            continue

        data = sio.loadmat(str(filepath))
        idx_shape = data['Index'].shape
        cls_shape = data['Class'].shape
        idx_dtype = data['Index'].dtype
        cls_dtype = data['Class'].dtype

        # Check row vector format
        is_row_vector = idx_shape[0] == 1 and cls_shape[0] == 1
        dtype_ok = idx_dtype in [np.int32, np.int64] and cls_dtype in [np.uint8, np.int32, np.int64]

        if is_row_vector and dtype_ok:
            print(f"  ✓ D{i}.mat: Index {idx_shape} {idx_dtype}, Class {cls_shape} {cls_dtype}")
            results[f'D{i}'] = {'compliant': True, 'data': data}
        else:
            print(f"  ✗ D{i}.mat: Index {idx_shape} {idx_dtype}, Class {cls_shape} {cls_dtype}")
            all_compliant = False
            results[f'D{i}'] = {'compliant': False, 'data': data}

    if all_compliant:
        print("\n  ✓ ALL FILES FORMAT COMPLIANT")
    else:
        print("\n  ✗ FORMAT ISSUES DETECTED - Run with --fix to repair")

    return all_compliant, results


def audit_class_distributions():
    """Print detailed class distribution audit table."""
    print_header("[2] CLASS DISTRIBUTION AUDIT")

    # Header
    print(f"  {'Dataset':<10} {'Spikes':>8} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10}")
    print("  " + "─" * 72)

    # D1 Ground Truth
    total = D1_DISTRIBUTION['total']
    row = f"  {'D1 (GT)':<10} {total:>8}"
    for c in range(1, 6):
        pct = D1_PERCENTAGES[c]
        row += f" {pct:>5.1f}%{D1_DISTRIBUTION[c]:>4}"
    print(row)
    print("  " + "─" * 72)

    # Load and analyze each submission
    all_data = {}
    for i in range(2, 7):
        filepath = SUBMISSIONS_DIR / f'D{i}.mat'

        if not filepath.exists():
            print(f"  {'D' + str(i):<10} {'N/A':>8} File not found")
            continue

        data = sio.loadmat(str(filepath))
        indices = data['Index'].flatten()
        classes = data['Class'].flatten()

        all_data[f'D{i}'] = {'indices': indices, 'classes': classes}

        total = len(indices)
        dist = Counter(classes)

        row = f"  {'D' + str(i):<10} {total:>8}"
        for c in range(1, 6):
            cnt = dist.get(c, 0)
            pct = 100 * cnt / total if total > 0 else 0
            row += f" {pct:>5.1f}%{cnt:>4}"
        print(row)

    print("  " + "─" * 72)

    # Deviation analysis
    print_subheader("Deviation from D1 Golden Ratio")
    print(f"  {'Dataset':<10} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'Max Dev':>10}")
    print("  " + "─" * 72)

    for dataset_name, data in all_data.items():
        total = len(data['classes'])
        dist = Counter(data['classes'])

        deviations = []
        row = f"  {dataset_name:<10}"
        for c in range(1, 6):
            cnt = dist.get(c, 0)
            pct = 100 * cnt / total if total > 0 else 0
            dev = pct - D1_PERCENTAGES[c]
            deviations.append(abs(dev))
            sign = "+" if dev >= 0 else ""
            row += f" {sign}{dev:>5.1f}%   "

        max_dev = max(deviations)
        row += f" {max_dev:>6.1f}%"
        print(row)

    return all_data


def check_isi_violations(all_data=None):
    """Check for Inter-Spike Interval violations (same neuron firing < 2ms apart)."""
    print_header("[3] INTER-SPIKE INTERVAL (ISI) VIOLATION CHECK")
    print(f"  Refractory period: {REFRACTORY_PERIOD_MS}ms ({REFRACTORY_SAMPLES} samples)")
    print()

    # Header
    print(f"  {'Dataset':<10} {'C1':>8} {'C2':>8} {'C3':>8} {'C4':>8} {'C5':>8} {'Total':>8}")
    print("  " + "─" * 62)

    warnings = []

    for i in range(2, 7):
        dataset_name = f'D{i}'

        if all_data and dataset_name in all_data:
            indices = all_data[dataset_name]['indices']
            classes = all_data[dataset_name]['classes']
        else:
            filepath = SUBMISSIONS_DIR / f'D{i}.mat'
            if not filepath.exists():
                continue
            data = sio.loadmat(str(filepath))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

        row = f"  {dataset_name:<10}"
        total_violations = 0
        total_spikes = 0

        for c in range(1, 6):
            class_mask = classes == c
            class_indices = np.sort(indices[class_mask])

            if len(class_indices) < 2:
                row += f" {'0.0%':>8}"
                continue

            isi = np.diff(class_indices)
            violations = np.sum(isi < REFRACTORY_SAMPLES)
            violation_rate = 100 * violations / len(class_indices)

            total_violations += violations
            total_spikes += len(class_indices)

            if violation_rate > 1.0:
                row += f" {violation_rate:>5.1f}%⚠"
                warnings.append(f"{dataset_name} Class {c}: {violation_rate:.1f}%")
            else:
                row += f" {violation_rate:>6.1f}%"

        total_rate = 100 * total_violations / total_spikes if total_spikes > 0 else 0
        row += f" {total_rate:>6.1f}%"
        print(row)

    print("  " + "─" * 62)

    if warnings:
        print("\n  ⚠ WARNINGS (>1% violation rate):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  ✓ All datasets have acceptable ISI violation rates (<1%)")


def run_sanity_checks(all_data=None):
    """Run comprehensive sanity checks on submission data."""
    print_header("[4] SANITY CHECKS")

    all_passed = True

    for i in range(2, 7):
        dataset_name = f'D{i}'
        print_subheader(f"{dataset_name}")

        if all_data and dataset_name in all_data:
            indices = all_data[dataset_name]['indices']
            classes = all_data[dataset_name]['classes']
        else:
            filepath = SUBMISSIONS_DIR / f'D{i}.mat'
            if not filepath.exists():
                print(f"    ✗ File not found")
                all_passed = False
                continue
            data = sio.loadmat(str(filepath))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

        passed = True

        # Check 1: Array lengths match
        if len(indices) == len(classes):
            print(f"    ✓ Index/Class arrays same length: {len(indices)}")
        else:
            print(f"    ✗ Length mismatch: Index={len(indices)}, Class={len(classes)}")
            passed = False

        # Check 2: No duplicate indices
        unique_indices = len(np.unique(indices))
        if unique_indices == len(indices):
            print(f"    ✓ No duplicate indices")
        else:
            dup_count = len(indices) - unique_indices
            print(f"    ✗ {dup_count} duplicate indices found")
            passed = False

        # Check 3: Indices in valid range
        min_idx, max_idx = indices.min(), indices.max()
        if min_idx >= 1 and max_idx <= TOTAL_SAMPLES:
            print(f"    ✓ Indices in valid range [{min_idx}, {max_idx}]")
        else:
            print(f"    ✗ Indices out of range: [{min_idx}, {max_idx}] (valid: 1-{TOTAL_SAMPLES})")
            passed = False

        # Check 4: Class values in range [1, 5]
        unique_classes = np.unique(classes)
        if all(c in [1, 2, 3, 4, 5] for c in unique_classes):
            print(f"    ✓ All class values valid: {list(unique_classes)}")
        else:
            invalid = [c for c in unique_classes if c not in [1, 2, 3, 4, 5]]
            print(f"    ✗ Invalid class values: {invalid}")
            passed = False

        # Check 5: All 5 classes represented
        if set(unique_classes) == {1, 2, 3, 4, 5}:
            print(f"    ✓ All 5 classes represented")
        else:
            missing = set([1, 2, 3, 4, 5]) - set(unique_classes)
            print(f"    ⚠ Missing classes: {missing}")

        # Check 6: Indices mostly sorted (ascending)
        sorted_count = np.sum(np.diff(indices) > 0)
        sort_pct = 100 * sorted_count / (len(indices) - 1) if len(indices) > 1 else 100
        if sort_pct > 95:
            print(f"    ✓ Indices mostly ascending ({sort_pct:.1f}%)")
        else:
            print(f"    ⚠ Indices not well sorted ({sort_pct:.1f}% ascending)")

        # Check 7: Temporal distribution (spikes across recording)
        bins = np.linspace(1, TOTAL_SAMPLES, 11)
        hist, _ = np.histogram(indices, bins=bins)
        min_bin, max_bin = hist.min(), hist.max()
        if max_bin > 0 and min_bin / max_bin > 0.1:  # At least 10% ratio
            print(f"    ✓ Spikes distributed across recording (bin range: {min_bin}-{max_bin})")
        else:
            print(f"    ⚠ Uneven temporal distribution (bin range: {min_bin}-{max_bin})")

        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ ALL SANITY CHECKS PASSED")
    else:
        print("  ✗ SOME CHECKS FAILED - Review warnings above")

    return all_passed


def plot_raster(all_data=None, save_path=None):
    """Create spike raster plots showing temporal distribution."""
    if not HAS_MATPLOTLIB:
        print("  ⚠ matplotlib not available - skipping raster plots")
        return

    print_header("[5] SPIKE RASTER PLOTS")

    if save_path is None:
        ANALYSIS_DIR.mkdir(exist_ok=True)
        save_path = ANALYSIS_DIR / 'raster_plots.png'

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}
    class_names = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5'}

    # Show first 1 second only
    time_limit_samples = SAMPLE_RATE  # 25000 samples = 1 second

    for idx, i in enumerate(range(2, 7)):
        ax = axes[idx]
        dataset_name = f'D{i}'

        if all_data and dataset_name in all_data:
            indices = all_data[dataset_name]['indices']
            classes = all_data[dataset_name]['classes']
        else:
            filepath = SUBMISSIONS_DIR / f'D{i}.mat'
            if not filepath.exists():
                ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
                ax.set_ylabel(dataset_name)
                continue
            data = sio.loadmat(str(filepath))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

        # Filter to first 1 second
        mask = indices <= time_limit_samples
        plot_indices = indices[mask]
        plot_classes = classes[mask]

        # Convert to time in seconds
        times = plot_indices / SAMPLE_RATE

        # Plot each class
        for c in range(1, 6):
            class_mask = plot_classes == c
            class_times = times[class_mask]
            ax.scatter(class_times, np.full_like(class_times, c),
                      c=colors[c], s=10, alpha=0.7, label=class_names[c])

        ax.set_ylabel(dataset_name, fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['C1', 'C2', 'C3', 'C4', 'C5'])
        ax.grid(True, alpha=0.3)

        # Count spikes in first second
        spike_count = len(plot_indices)
        ax.text(0.98, 0.95, f'n={spike_count}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_title('Spike Raster Plots - First 1 Second of Each Dataset', fontsize=14)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:5], labels[:5], loc='upper right', ncol=5, fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved raster plots to: {save_path}")


def plot_pca_clusters(all_data=None, save_path=None):
    """Create PCA cluster visualization of predicted waveforms."""
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("  ⚠ matplotlib or sklearn not available - skipping PCA plots")
        return

    print_header("[6] PCA CLUSTER VISUALIZATION")

    if save_path is None:
        ANALYSIS_DIR.mkdir(exist_ok=True)
        save_path = ANALYSIS_DIR / 'final_clusters.png'

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

    waveform_halfwidth = 30  # samples on each side of spike
    max_waveforms = 1000  # limit for performance

    for idx, i in enumerate(range(2, 7)):
        ax = axes[idx]
        dataset_name = f'D{i}'

        # Load signal
        signal_path = DATASETS_DIR / f'D{i}.mat'
        if not signal_path.exists():
            ax.text(0.5, 0.5, 'Signal not found', ha='center', va='center')
            ax.set_title(dataset_name)
            continue

        signal_data = sio.loadmat(str(signal_path))
        signal = signal_data['d'].flatten()

        # Load predictions
        if all_data and dataset_name in all_data:
            indices = all_data[dataset_name]['indices']
            classes = all_data[dataset_name]['classes']
        else:
            filepath = SUBMISSIONS_DIR / f'D{i}.mat'
            if not filepath.exists():
                ax.text(0.5, 0.5, 'Predictions not found', ha='center', va='center')
                ax.set_title(dataset_name)
                continue
            data = sio.loadmat(str(filepath))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

        # Subsample if too many
        if len(indices) > max_waveforms:
            sample_idx = np.random.choice(len(indices), max_waveforms, replace=False)
            indices = indices[sample_idx]
            classes = classes[sample_idx]

        # Extract waveforms
        waveforms = []
        valid_classes = []
        for j, spike_idx in enumerate(indices):
            start = int(spike_idx) - waveform_halfwidth
            end = int(spike_idx) + waveform_halfwidth
            if start >= 0 and end < len(signal):
                waveforms.append(signal[start:end])
                valid_classes.append(classes[j])

        if len(waveforms) < 10:
            ax.text(0.5, 0.5, 'Too few valid waveforms', ha='center', va='center')
            ax.set_title(dataset_name)
            continue

        waveforms = np.array(waveforms)
        valid_classes = np.array(valid_classes)

        # PCA
        pca = PCA(n_components=2)
        waveforms_2d = pca.fit_transform(waveforms)

        # Plot each class
        for c in range(1, 6):
            mask = valid_classes == c
            if np.sum(mask) > 0:
                ax.scatter(waveforms_2d[mask, 0], waveforms_2d[mask, 1],
                          c=colors[c], s=5, alpha=0.5, label=f'C{c}')

        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.0f}%)', fontsize=9)
        if idx == 0:
            ax.set_ylabel(f'PC2', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle('PCA Projection of Predicted Spike Waveforms', fontsize=14, y=1.08)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved PCA clusters to: {save_path}")


def print_final_summary(format_ok, sanity_ok):
    """Print final validation summary."""
    print()
    print("=" * 80)
    print(" FINAL SUBMISSION VALIDATION SUMMARY")
    print("=" * 80)
    print()

    if format_ok and sanity_ok:
        print("  ╔═══════════════════════════════════════════════════════════════════════╗")
        print("  ║                                                                       ║")
        print("  ║   ✓ SUBMISSION FILES ARE READY                                       ║")
        print("  ║                                                                       ║")
        print("  ║   All files:                                                         ║")
        print("  ║   - Format compliant (1xN row vectors, uppercase names)              ║")
        print("  ║   - Pass sanity checks (valid indices, classes, no duplicates)       ║")
        print("  ║   - Contain predictions for all 5 neuron classes                     ║")
        print("  ║                                                                       ║")
        print("  ╚═══════════════════════════════════════════════════════════════════════╝")
    else:
        print("  ╔═══════════════════════════════════════════════════════════════════════╗")
        print("  ║                                                                       ║")
        print("  ║   ⚠ ISSUES DETECTED - REVIEW ABOVE                                   ║")
        print("  ║                                                                       ║")
        if not format_ok:
            print("  ║   - Format issues: Run with --fix flag                               ║")
        if not sanity_ok:
            print("  ║   - Sanity check failures: Review warnings                           ║")
        print("  ║                                                                       ║")
        print("  ╚═══════════════════════════════════════════════════════════════════════╝")

    print()
    print(f"  Output directory: {SUBMISSIONS_DIR}")
    print(f"  Analysis plots:   {ANALYSIS_DIR}")
    print()


def main():
    """Run full validation pipeline."""
    parser = argparse.ArgumentParser(description='Validate submission files')
    parser.add_argument('--fix', action='store_true',
                       help='Fix format issues (convert to row vectors, uppercase names)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    args = parser.parse_args()

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " FINAL SUBMISSION VALIDATION REPORT ".center(78) + "║")
    print("║" + " EE40098 Coursework C - Neural Spike Sorting ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # Fix format if requested
    if args.fix:
        fix_submission_format()

    # Run validation checks
    format_ok, _ = verify_format_compliance()

    if not format_ok and not args.fix:
        print("\n  ⚠ Format issues detected. Run with --fix to repair.")
        print("     python src/validate_submission.py --fix")
        return

    # Class distribution audit
    all_data = audit_class_distributions()

    # ISI violation check
    check_isi_violations(all_data)

    # Sanity checks
    sanity_ok = run_sanity_checks(all_data)

    # Generate plots
    if not args.no_plots:
        plot_raster(all_data)
        plot_pca_clusters(all_data)

    # Final summary
    print_final_summary(format_ok, sanity_ok)


if __name__ == '__main__':
    main()
