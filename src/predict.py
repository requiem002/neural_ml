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

    # Dataset configurations using FIXED VOLTAGE THRESHOLDS
    # Based on D1 spike amplitude analysis:
    # - D1 5th percentile = 0.67V (catches 95% of spikes)
    # - D1 10th percentile = 0.75V (catches 90% of spikes)
    #
    # Key insight: As noise increases, we need HIGHER voltage thresholds
    # to avoid detecting noise peaks as spikes. The threshold should be
    # above the noise floor while still catching spikes.
    #
    # Noise levels (σ): D1=0.11, D2=0.23, D3=0.31, D4=0.49, D5=0.92, D6=1.37
    datasets = {
        'D2': {'voltage_threshold': 0.80, 'matched_filter': False},   # 60dB SNR
        'D3': {'voltage_threshold': 0.95, 'matched_filter': False},   # 40dB SNR
        'D4': {'voltage_threshold': 1.40, 'matched_filter': False},   # 20dB SNR
        'D5': {'voltage_threshold': 2.50, 'matched_filter': False},   # 0dB SNR
        'D6': {'voltage_threshold': 3.50, 'matched_filter': False},   # <0dB SNR
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

        # Predict using fixed voltage threshold
        indices, classes = pipeline.predict(
            d,
            use_matched_filter=config['matched_filter'],
            voltage_threshold=config['voltage_threshold']
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
