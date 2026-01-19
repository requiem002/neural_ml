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
    # Lower thresholds for noisier data to maintain recall
    # Use matched filtering for very noisy datasets
    datasets = {
        'D2': {'threshold': 5.0, 'matched_filter': False},   # 60dB SNR
        'D3': {'threshold': 4.5, 'matched_filter': False},   # 40dB SNR
        'D4': {'threshold': 4.0, 'matched_filter': True},    # 20dB SNR
        'D5': {'threshold': 3.5, 'matched_filter': True},    # 0dB SNR
        'D6': {'threshold': 3.0, 'matched_filter': True},    # <0dB SNR
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

        # Predict
        indices, classes = pipeline.predict(
            d,
            use_matched_filter=config['matched_filter'],
            threshold_factor=config['threshold']
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
