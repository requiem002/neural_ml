"""Generate predictions for test datasets D2-D6."""

import sys
from pathlib import Path
import zipfile
from pipeline import SpikeSortingPipeline


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / 'datasets'
    model_dir = Path(__file__).parent.parent / 'models'
    output_dir = Path(__file__).parent.parent / 'predictions'
    output_dir.mkdir(exist_ok=True)

    # Load trained pipeline
    print("Loading trained pipeline...")
    pipeline = SpikeSortingPipeline()
    pipeline.load(model_dir / 'spike_sorter.pkl')

    # Dataset configurations
    # Based on D1 spike amplitude analysis:
    # - D1 5th percentile = 0.67V (catches 95% of spikes)
    # - D1 mean spike amplitude ~1.9V
    #
    # Strategy:
    # - D2/D3: Low noise, use voltage threshold (standard detection)
    # - D4/D5/D6: High noise, use matched filtering with correlation threshold
    #
    # Matched filtering is more robust in high noise because it looks for
    # the SHAPE of spikes rather than just amplitude.
    #
    # Noise levels (σ): D1=0.11, D2=0.23, D3=0.31, D4=0.49, D5=0.92, D6=1.37
    # Peer benchmark D4: ~2,600 spikes with balanced class distribution
    # Strategy: Use voltage-based detection with thresholds tuned to each dataset's noise level
    # Target: ~2600 spikes per dataset (similar to peer benchmark)
    #
    # Noise levels (σ): D1=0.11, D2=0.23, D3=0.31, D4=0.49, D5=0.92, D6=1.37
    # Spike amplitudes: Mean ~1.9V, 5th percentile ~0.67V
    #
    # For noisier datasets, need higher thresholds to avoid false positives
    datasets = {
        'D2': {'voltage_threshold': 0.80, 'matched_filter': False},   # 60dB SNR
        'D3': {'voltage_threshold': 0.95, 'matched_filter': False},   # 40dB SNR
        'D4': {'voltage_threshold': 1.50, 'matched_filter': False},   # 20dB SNR - ~2600 spikes
        'D5': {'voltage_threshold': 2.80, 'matched_filter': False},   # 0dB SNR - higher to reduce false positives
        'D6': {'voltage_threshold': 4.00, 'matched_filter': False},   # <0dB SNR - much higher
    }

    # Process each test dataset
    for dataset_name, config in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print('='*50)

        # Load data
        filepath = data_dir / f'{dataset_name}.mat'
        d, _, _ = pipeline.load_data(filepath)
        print(f"Signal length: {len(d)} samples")

        # Predict using appropriate method
        indices, classes = pipeline.predict(
            d,
            use_matched_filter=config.get('matched_filter', False),
            voltage_threshold=config.get('voltage_threshold'),
            correlation_threshold=config.get('correlation_threshold')
        )

        print(f"Detected {len(indices)} spikes")

        # Print class distribution
        from collections import Counter
        class_dist = Counter(classes)
        print("Class distribution:")
        for c in sorted(class_dist.keys()):
            print(f"  Class {c}: {class_dist[c]} ({100*class_dist[c]/len(classes):.1f}%)")

        # Save predictions
        mat_filepath = output_dir / f'{dataset_name}_predictions.mat'
        pipeline.save_predictions(indices, classes, mat_filepath)

        # Create ZIP file
        zip_filepath = output_dir / f'{dataset_name}.zip'
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(mat_filepath, f'{dataset_name}_predictions.mat')
        print(f"ZIP file created: {zip_filepath}")

    print("\n" + "="*50)
    print("ALL PREDICTIONS COMPLETE")
    print("="*50)
    print(f"\nPredictions saved to: {output_dir}")


if __name__ == '__main__':
    main()
