"""Train the spike sorting pipeline on D1 dataset."""

import sys
from pathlib import Path
from pipeline import SpikeSortingPipeline


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / 'datasets'
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SpikeSortingPipeline(
        window_before=30,
        window_after=30,
        n_pca_components=10
    )

    # Load training data
    print("\nLoading D1.mat...")
    d, index, classes = pipeline.load_data(data_dir / 'D1.mat')
    print(f"Signal length: {len(d)} samples")
    print(f"Number of labeled spikes: {len(index)}")

    # Train pipeline
    print("\n" + "="*50)
    print("TRAINING PIPELINE")
    print("="*50)
    pipeline.train(d, index, classes)

    # Evaluate on training data with detection
    print("\n" + "="*50)
    print("FULL EVALUATION ON D1 (Training Data)")
    print("="*50)
    results = pipeline.evaluate_full(d, index, classes, voltage_threshold=0.75)

    # Save trained model
    pipeline.save(model_dir / 'spike_sorter.pkl')

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nModel saved to: {model_dir / 'spike_sorter.pkl'}")
    print(f"Combined F1 on D1: {results['combined_f1']:.4f}")


if __name__ == '__main__':
    main()
