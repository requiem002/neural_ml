"""
Threshold Optimization for Post-Processing Rules

This module optimizes the hardcoded thresholds in the post-processing pipeline
using synthetic D6-like data with known ground truth labels.

Strategy:
1. Create synthetic D6 by adding noise (sigma=1.37V) to D1 ground truth waveforms
2. Use differential evolution to optimize thresholds
3. Maximize weighted F1 on synthetic D6
4. Constraint: Each class must have >1% representation (avoid class collapse)
"""

import numpy as np
import scipy.io as sio
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
from collections import Counter
import json
import torch
import pickle
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from deprecated.feature_extractor import extract_waveforms_at_indices


class ThresholdOptimizer:
    """Optimize post-processing thresholds using synthetic noisy data."""

    # Default thresholds (current hardcoded values from cnn_experiment.py:686-743)
    DEFAULT_THRESHOLDS = {
        'c3_amp_cap': 4.0,           # Redirect high-amp C3 predictions
        'c4_amp_cap': 5.5,           # Redirect high-amp C4 predictions
        'c5_fwhm_thresh': 0.85,      # Rescue wide pulses to C5
        'c5_amp_low': 3.0,           # Guard against stealing from C2 (lower bound)
        'c5_amp_high': 5.5,          # Guard against stealing from C2 (upper bound)
        'c2_narrow_fwhm': 0.58,      # Redirect narrow C2 to C1
        'c2_wide_fwhm': 0.82,        # Redirect wide C2 to C5
        'low_amp_thresh': 2.5,       # Rescue low-amp to C3/C4
        'confidence_thresh': 0.85,   # When to apply corrections
    }

    # Search bounds for each parameter
    SEARCH_BOUNDS = {
        'c3_amp_cap': (3.5, 4.5),
        'c4_amp_cap': (5.0, 6.0),
        'c5_fwhm_thresh': (0.78, 0.92),
        'c5_amp_low': (2.5, 3.5),
        'c5_amp_high': (5.0, 6.0),
        'c2_narrow_fwhm': (0.50, 0.65),
        'c2_wide_fwhm': (0.75, 0.88),
        'low_amp_thresh': (2.0, 3.2),
        'confidence_thresh': (0.78, 0.92),
    }

    def __init__(self, device=None):
        self.device = device if device else (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'datasets'
        self.model_dir = self.base_dir / 'models'
        self.config_dir = self.base_dir / 'configs'

        self.model = None
        self.amp_mean = None
        self.amp_std = None

        # D6 noise level (from MAD estimator)
        self.d6_sigma = 1.37  # D6 is 12.54x noisier than D1

        # Cached data
        self.synthetic_waveforms = None
        self.synthetic_labels = None
        self.synthetic_amp_features = None
        self.synthetic_fwhm = None
        self.synthetic_symmetry = None
        self.raw_predictions = None
        self.raw_confidences = None

    def load_model(self):
        """Load the trained CNN model."""
        from deprecated.cnn_experiment import DualBranchSpikeNet

        filepath = self.model_dir / 'cnn_model.pkl'
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.model = DualBranchSpikeNet(
            waveform_size=60, num_classes=5, num_amp_features=8
        ).to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.model.eval()

        self.amp_mean = state['amp_mean']
        self.amp_std = state['amp_std']

        print(f"Model loaded from: {filepath}")

    def create_synthetic_d6(self):
        """
        Create synthetic D6 data by adding noise to D1 ground truth waveforms.
        This gives us labeled "D6-like" data for optimization.
        """
        print("\n" + "="*60)
        print("CREATING SYNTHETIC D6 DATA")
        print("="*60)

        # Load D1 ground truth
        data = sio.loadmat(self.data_dir / 'D1.mat')
        d = data['d'].flatten()
        index = data['Index'].flatten()
        classes = data['Class'].flatten()

        # Extract waveforms at ground truth locations
        waveforms, valid_indices = extract_waveforms_at_indices(
            d, index, window_before=30, window_after=30
        )
        valid_mask = np.isin(index, valid_indices)
        valid_classes = classes[valid_mask]

        print(f"D1 ground truth: {len(waveforms)} spikes")
        print(f"D6 noise sigma: {self.d6_sigma}V")

        # Add Gaussian noise to simulate D6 conditions
        np.random.seed(42)  # Reproducibility
        noisy_waveforms = waveforms + np.random.randn(*waveforms.shape) * self.d6_sigma

        # Also create multiple noise realizations for more robust optimization
        all_waveforms = [noisy_waveforms]
        all_labels = [valid_classes]

        for seed in [123, 456, 789]:
            np.random.seed(seed)
            noisy = waveforms + np.random.randn(*waveforms.shape) * self.d6_sigma
            all_waveforms.append(noisy)
            all_labels.append(valid_classes)

        self.synthetic_waveforms = np.vstack(all_waveforms)
        self.synthetic_labels = np.hstack(all_labels)

        print(f"Synthetic D6 dataset: {len(self.synthetic_waveforms)} samples "
              f"(4x noise realizations)")
        print(f"Class distribution: {Counter(self.synthetic_labels)}")

        return self.synthetic_waveforms, self.synthetic_labels

    def extract_features(self, waveforms):
        """Extract amplitude features for classification."""
        from deprecated.cnn_experiment import CNNExperiment

        exp = CNNExperiment(device=self.device)
        exp.amp_mean = self.amp_mean
        exp.amp_std = self.amp_std

        # Get raw amplitude features
        raw_features = exp.extract_amplitude_features(waveforms)

        # Normalize for CNN
        amp_features = (raw_features - self.amp_mean) / self.amp_std

        # Normalize waveforms for shape branch
        wf_normalized = np.array([
            exp.preprocess_waveform(wf, normalize_for_shape=True) for wf in waveforms
        ])

        return wf_normalized, amp_features, raw_features

    def get_raw_predictions(self):
        """Get CNN predictions before post-processing."""
        if self.raw_predictions is not None:
            return self.raw_predictions, self.raw_confidences

        print("\nGenerating raw CNN predictions...")

        wf_normalized, amp_features, raw_features = self.extract_features(
            self.synthetic_waveforms
        )

        # Store for post-processing
        self.synthetic_amp_features = raw_features[:, 0]  # Peak amplitude
        self.synthetic_fwhm = raw_features[:, 3]  # FWHM in ms
        self.synthetic_symmetry = raw_features[:, 5]  # Symmetry ratio

        # Run CNN
        X_wf = torch.FloatTensor(wf_normalized).to(self.device)
        X_amp = torch.FloatTensor(amp_features).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_wf, X_amp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = outputs.max(1)
            classes = predicted.cpu().numpy() + 1  # Back to 1-indexed
            confidences = probs.max(axis=1)

        self.raw_predictions = classes
        self.raw_confidences = confidences

        # Report raw performance
        raw_f1 = f1_score(self.synthetic_labels, classes, average='weighted')
        print(f"Raw CNN F1 (no post-processing): {raw_f1:.4f}")

        return self.raw_predictions, self.raw_confidences

    def apply_post_processing(self, thresholds):
        """Apply post-processing rules with given thresholds."""
        raw_preds, confidences = self.get_raw_predictions()

        corrected = raw_preds.copy()

        for i, (pred_class, amp, fwhm, conf, sym) in enumerate(zip(
                raw_preds,
                self.synthetic_amp_features,
                self.synthetic_fwhm,
                confidences,
                self.synthetic_symmetry)):

            # RULE 1: Class 3 amplitude correction
            if pred_class == 3 and amp > thresholds['c3_amp_cap']:
                if fwhm > 0.78:
                    corrected[i] = 5
                elif fwhm > 0.65:
                    corrected[i] = 2
                else:
                    corrected[i] = 1

            # RULE 2: Class 4 amplitude correction
            elif pred_class == 4 and amp > thresholds['c4_amp_cap']:
                if fwhm > 0.78:
                    corrected[i] = 5
                elif fwhm > 0.65:
                    corrected[i] = 2
                else:
                    corrected[i] = 1

            # RULE 3: Class 5 rescue
            elif (fwhm > thresholds['c5_fwhm_thresh'] and
                  thresholds['c5_amp_low'] < amp < thresholds['c5_amp_high'] and
                  pred_class != 5):
                corrected[i] = 5

            # RULE 4: C2 over-prediction correction
            elif pred_class == 2:
                if fwhm < thresholds['c2_narrow_fwhm'] and amp < 5.5:
                    corrected[i] = 1
                elif fwhm > thresholds['c2_wide_fwhm'] and amp < 5.3:
                    corrected[i] = 5
                elif conf < thresholds['confidence_thresh']:
                    if 4.0 < amp < 5.2 and fwhm < 0.68:
                        corrected[i] = 1

            # RULE 5: Low amplitude spike correction
            elif pred_class in [1, 2, 5] and amp < thresholds['low_amp_thresh']:
                if conf < thresholds['confidence_thresh']:
                    corrected[i] = 3 if amp < 1.8 else 4

        return corrected

    def evaluate_thresholds(self, threshold_values):
        """Evaluate a set of thresholds and return negative F1 (for minimization)."""
        # Convert array to dict
        param_names = list(self.SEARCH_BOUNDS.keys())
        thresholds = {name: val for name, val in zip(param_names, threshold_values)}

        # Apply post-processing
        predictions = self.apply_post_processing(thresholds)

        # Check class collapse constraint
        class_counts = Counter(predictions)
        total = len(predictions)

        for c in range(1, 6):
            if class_counts.get(c, 0) / total < 0.01:
                return 1.0  # Heavy penalty for class collapse

        # Compute weighted F1
        f1 = f1_score(self.synthetic_labels, predictions, average='weighted')

        return -f1  # Negative because we're minimizing

    def optimize(self, maxiter=100, popsize=15, verbose=True):
        """
        Run differential evolution optimization.

        Args:
            maxiter: Maximum iterations
            popsize: Population size multiplier
            verbose: Whether to print progress

        Returns:
            Optimized thresholds dict
        """
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION")
        print("="*60)

        # Prepare data
        if self.synthetic_waveforms is None:
            self.create_synthetic_d6()

        if self.model is None:
            self.load_model()

        # Get raw predictions
        self.get_raw_predictions()

        # Get original F1
        original_preds = self.apply_post_processing(self.DEFAULT_THRESHOLDS)
        original_f1 = f1_score(self.synthetic_labels, original_preds, average='weighted')
        print(f"\nOriginal thresholds F1: {original_f1:.4f}")

        # Setup bounds
        param_names = list(self.SEARCH_BOUNDS.keys())
        bounds = [self.SEARCH_BOUNDS[name] for name in param_names]

        print(f"\nOptimizing {len(param_names)} parameters...")
        print(f"Max iterations: {maxiter}, Population: {popsize * len(param_names)}")

        # Callback for progress
        iteration_count = [0]
        best_f1 = [0]

        def callback(xk, convergence):
            iteration_count[0] += 1
            f1 = -self.evaluate_thresholds(xk)
            if f1 > best_f1[0]:
                best_f1[0] = f1
            if verbose and iteration_count[0] % 10 == 0:
                print(f"  Iteration {iteration_count[0]}: Best F1 = {best_f1[0]:.4f}")

        # Run optimization
        result = differential_evolution(
            self.evaluate_thresholds,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=callback,
            disp=verbose,
            workers=1,  # Single-threaded for reproducibility
            updating='immediate'
        )

        # Extract optimized thresholds
        optimized = {name: val for name, val in zip(param_names, result.x)}
        optimized_f1 = -result.fun

        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"\nOriginal F1: {original_f1:.4f}")
        print(f"Optimized F1: {optimized_f1:.4f}")
        print(f"Improvement: {100*(optimized_f1 - original_f1)/original_f1:+.2f}%")

        print("\nParameter changes:")
        print(f"{'Parameter':<20} {'Original':>10} {'Optimized':>10} {'Delta':>10}")
        print("-" * 52)
        for name in param_names:
            orig = self.DEFAULT_THRESHOLDS[name]
            opt = optimized[name]
            delta = opt - orig
            print(f"{name:<20} {orig:>10.3f} {opt:>10.3f} {delta:>+10.3f}")

        return optimized, optimized_f1, original_f1

    def save_optimized(self, optimized_thresholds, optimized_f1, original_f1):
        """Save optimized thresholds to JSON file."""
        output = {
            **optimized_thresholds,
            'optimization_f1': optimized_f1,
            'original_f1': original_f1,
            'improvement_pct': 100 * (optimized_f1 - original_f1) / original_f1,
            'timestamp': datetime.now().isoformat(),
            'd6_sigma_used': self.d6_sigma,
        }

        filepath = self.config_dir / 'optimized_thresholds.json'
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nOptimized thresholds saved to: {filepath}")
        return filepath

    def evaluate_per_class(self, thresholds):
        """Evaluate per-class performance with given thresholds."""
        predictions = self.apply_post_processing(thresholds)

        print("\nPer-class F1 scores:")
        print(f"{'Class':<8} {'Count':>8} {'Correct':>8} {'F1':>8}")
        print("-" * 36)

        for c in range(1, 6):
            mask = self.synthetic_labels == c
            true_count = np.sum(mask)
            pred_count = np.sum(predictions == c)
            correct = np.sum((predictions == c) & mask)

            precision = correct / pred_count if pred_count > 0 else 0
            recall = correct / true_count if true_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Class {c:<3} {true_count:>8} {correct:>8} {f1:>8.3f}")

        print("\nConfusion Matrix:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.synthetic_labels, predictions)
        print(cm)


def run_threshold_optimization(maxiter=100, verbose=True):
    """Main function to run threshold optimization."""
    optimizer = ThresholdOptimizer()

    # Create synthetic D6
    optimizer.create_synthetic_d6()

    # Load model
    optimizer.load_model()

    # Run optimization
    optimized, opt_f1, orig_f1 = optimizer.optimize(maxiter=maxiter, verbose=verbose)

    # Save results
    optimizer.save_optimized(optimized, opt_f1, orig_f1)

    # Show per-class breakdown
    optimizer.evaluate_per_class(optimized)

    return optimized


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Threshold Optimization')
    parser.add_argument('--maxiter', type=int, default=100, help='Max iterations')
    parser.add_argument('--quick', action='store_true', help='Quick run (20 iterations)')
    args = parser.parse_args()

    maxiter = 20 if args.quick else args.maxiter
    run_threshold_optimization(maxiter=maxiter)
