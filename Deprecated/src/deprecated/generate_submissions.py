#!/usr/bin/env python3
"""
Generate Final Submission Files for EE40098 Coursework C

This script generates the final submission .mat files for datasets D2-D6
using the CNN classifier with improved post-processing.

Output Structure:
    submissions/
    ├── D2.mat  (Index, Class as 1xN row vectors - MATLAB compatible)
    ├── D3.mat
    ├── D4.mat
    ├── D5.mat
    └── D6.mat

Marking Scheme:
    F_Dataset = 0.3 × F_Detection + 0.7 × F_Classification
    F_Final = 0.1×D2 + 0.15×D3 + 0.2×D4 + 0.25×D5 + 0.3×D6
    (D5 + D6 = 55% of final score!)
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import Counter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from deprecated.cnn_experiment import CNNExperiment


def generate_submissions():
    """Generate all submission files using CNN classifier."""

    print("=" * 80)
    print("GENERATING FINAL SUBMISSION FILES")
    print("=" * 80)

    # Create output directory
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'submissions'
    output_dir.mkdir(exist_ok=True)

    # Dataset configurations
    # Thresholds tuned based on noise levels:
    # D2: 60dB SNR, D3: 40dB, D4: 20dB, D5: ~0dB, D6: <0dB
    datasets = {
        'D2': {'voltage_threshold': 0.80},
        'D3': {'voltage_threshold': 0.95},
        'D4': {'voltage_threshold': 1.50},
        'D5': {'voltage_threshold': 2.80},
        'D6': {'voltage_threshold': 4.00},
    }

    # Initialize CNN experiment
    print("\nLoading CNN model...")
    cnn_exp = CNNExperiment()
    cnn_exp.load_model()

    # Generate predictions for each dataset
    all_results = {}

    for dataset_name, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")

        indices, classes = cnn_exp.predict_dataset(
            dataset_name,
            config['voltage_threshold'],
            verbose=True,
            use_improved_correction=True
        )

        # Store results
        all_results[dataset_name] = {
            'indices': indices,
            'classes': classes,
            'count': len(indices),
            'distribution': Counter(classes) if len(classes) > 0 else {}
        }

        # Save to submission format (MATLAB-compatible: 1xN row vectors, uppercase names)
        output_filename = f'D{dataset_name[1]}.mat'  # D2 -> D2.mat (uppercase!)
        output_path = output_dir / output_filename

        if len(indices) > 0:
            # CRITICAL: Use row vectors (1, N) to match D1.mat format
            # Index: int32, Class: uint8 (same as D1.mat)
            sio.savemat(str(output_path), {
                'Index': indices.reshape(1, -1).astype(np.int32),
                'Class': classes.reshape(1, -1).astype(np.uint8)
            }, do_compression=False)
            print(f"Saved: {output_path}")
        else:
            print(f"WARNING: No spikes detected for {dataset_name}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUBMISSION SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':<10} {'Spikes':<10} {'C1':<12} {'C2':<12} {'C3':<12} {'C4':<12} {'C5':<12}")
    print("-" * 82)

    # D1 ground truth for reference
    print(f"{'D1 (GT)':<10} {'2176':<10} {'21.0%':<12} {'20.3%':<12} {'18.7%':<12} {'20.4%':<12} {'19.6%':<12}")
    print("-" * 82)

    for dataset_name in ['D2', 'D3', 'D4', 'D5', 'D6']:
        result = all_results[dataset_name]
        count = result['count']
        dist = result['distribution']

        row = f"{dataset_name:<10} {count:<10}"
        for c in range(1, 6):
            cnt = dist.get(c, 0)
            pct = 100 * cnt / count if count > 0 else 0
            row += f"{pct:>5.1f}%{cnt:>6} "
        print(row)

    # Verify output files
    print("\n" + "=" * 80)
    print("OUTPUT FILES VERIFICATION")
    print("=" * 80)

    for dataset_name in ['D2', 'D3', 'D4', 'D5', 'D6']:
        output_filename = f'D{dataset_name[1]}.mat'  # Uppercase!
        output_path = output_dir / output_filename

        if output_path.exists():
            data = sio.loadmat(str(output_path))
            indices = data['Index'].flatten()
            classes = data['Class'].flatten()
            print(f"✓ {output_filename}: {len(indices)} spikes, classes {np.unique(classes)}")
        else:
            print(f"✗ {output_filename}: NOT FOUND")

    print(f"\n{'='*80}")
    print(f"Submission files saved to: {output_dir}")
    print(f"{'='*80}")

    return all_results


def verify_submission_format():
    """Verify that submission files match the required format."""

    print("\n" + "=" * 80)
    print("SUBMISSION FORMAT VERIFICATION")
    print("=" * 80)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'submissions'

    required_files = ['D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat']  # Uppercase!
    all_valid = True

    for filename in required_files:
        filepath = output_dir / filename
        print(f"\nChecking {filename}...")

        if not filepath.exists():
            print(f"  ✗ File not found!")
            all_valid = False
            continue

        try:
            data = sio.loadmat(str(filepath))

            # Check for required fields
            if 'Index' not in data:
                print(f"  ✗ Missing 'Index' field")
                all_valid = False
                continue

            if 'Class' not in data:
                print(f"  ✗ Missing 'Class' field")
                all_valid = False
                continue

            indices = data['Index'].flatten()
            classes = data['Class'].flatten()

            # Validate data
            if len(indices) != len(classes):
                print(f"  ✗ Index and Class arrays have different lengths")
                all_valid = False
                continue

            if len(indices) == 0:
                print(f"  ✗ Empty arrays")
                all_valid = False
                continue

            # Check class values are 1-5
            unique_classes = np.unique(classes)
            invalid_classes = unique_classes[(unique_classes < 1) | (unique_classes > 5)]
            if len(invalid_classes) > 0:
                print(f"  ✗ Invalid class values: {invalid_classes}")
                all_valid = False
                continue

            # Check indices are positive integers
            if np.any(indices < 1):
                print(f"  ✗ Invalid index values (must be >= 1)")
                all_valid = False
                continue

            print(f"  ✓ Valid: {len(indices)} spikes, classes {list(unique_classes)}")

        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            all_valid = False

    if all_valid:
        print("\n" + "=" * 80)
        print("✓ ALL SUBMISSION FILES ARE VALID")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ SOME FILES ARE INVALID - PLEASE CHECK")
        print("=" * 80)

    return all_valid


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate submission files')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing submission files')
    args = parser.parse_args()

    if args.verify:
        verify_submission_format()
    else:
        generate_submissions()
        verify_submission_format()
