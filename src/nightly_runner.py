#!/usr/bin/env python3
"""
OPERATION NIGHTWATCH - Automated Overnight Optimization

This is the master orchestration script that runs:
1. Safety backup of current submissions
2. Threshold optimization with synthetic D6
3. CNN ensemble training (5 models)
4. Optional LSTM experiment (if ensemble underperforms)
5. Final prediction generation
6. Validation and report generation

Usage:
    python src/nightly_runner.py 2>&1 | tee nightly_log.txt

    # Or for true unattended operation:
    nohup python src/nightly_runner.py > nightly_log.txt 2>&1 &
"""

import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import scipy.io as sio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class NightlyRunner:
    """Master orchestration for overnight optimization."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.src_dir = self.base_dir / 'src'
        self.submissions_dir = self.base_dir / 'submissions'
        self.backup_dir = self.base_dir / 'submissions_SAFE_BACKUP'
        self.nightly_dir = self.base_dir / 'submissions_nightly'
        self.config_dir = self.base_dir / 'configs'
        self.model_dir = self.base_dir / 'models'

        self.start_time = datetime.now()
        self.results = {
            'backup_created': False,
            'threshold_optimization': None,
            'ensemble_training': None,
            'lstm_experiment': None,
            'predictions_generated': False,
            'errors': []
        }

    def log(self, message, level='INFO'):
        """Print timestamped log message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")

    def backup_safe_submission(self):
        """Create backup of current submissions."""
        self.log("="*60)
        self.log("STEP 1: SAFETY BACKUP")
        self.log("="*60)

        if self.backup_dir.exists():
            self.log(f"Backup already exists at {self.backup_dir}")
            self.log("Skipping backup to preserve existing safe submission.")
            self.results['backup_created'] = True
            return True

        if not self.submissions_dir.exists():
            self.log("No submissions directory found. Creating empty backup.", 'WARN')
            self.backup_dir.mkdir(exist_ok=True)
            self.results['backup_created'] = True
            return True

        try:
            shutil.copytree(self.submissions_dir, self.backup_dir)
            self.log(f"Created backup: {self.backup_dir}")

            # Verify backup
            backup_files = list(self.backup_dir.glob('*.mat'))
            self.log(f"Backed up {len(backup_files)} .mat files")
            self.results['backup_created'] = True
            return True

        except Exception as e:
            self.log(f"Backup failed: {e}", 'ERROR')
            self.results['errors'].append(f"Backup: {str(e)}")
            return False

    def run_threshold_optimization(self, maxiter=100):
        """Run threshold optimization."""
        self.log("")
        self.log("="*60)
        self.log("STEP 2: THRESHOLD OPTIMIZATION")
        self.log("="*60)

        try:
            from optimize_thresholds import ThresholdOptimizer

            optimizer = ThresholdOptimizer()
            optimizer.create_synthetic_d6()
            optimizer.load_model()

            optimized, opt_f1, orig_f1 = optimizer.optimize(
                maxiter=maxiter, verbose=True
            )
            optimizer.save_optimized(optimized, opt_f1, orig_f1)

            self.results['threshold_optimization'] = {
                'original_f1': orig_f1,
                'optimized_f1': opt_f1,
                'improvement_pct': 100 * (opt_f1 - orig_f1) / orig_f1,
                'thresholds': optimized
            }

            self.log(f"Optimization complete: F1 {orig_f1:.4f} -> {opt_f1:.4f}")
            return optimized

        except Exception as e:
            self.log(f"Threshold optimization failed: {e}", 'ERROR')
            traceback.print_exc()
            self.results['errors'].append(f"Threshold optimization: {str(e)}")

            # Fall back to defaults
            from optimize_thresholds import ThresholdOptimizer
            return ThresholdOptimizer.DEFAULT_THRESHOLDS

    def train_ensemble(self, n_models=5, epochs=100):
        """Train CNN ensemble."""
        self.log("")
        self.log("="*60)
        self.log("STEP 3: ENSEMBLE TRAINING")
        self.log("="*60)

        try:
            from cnn_ensemble import CNNEnsemble

            ensemble = CNNEnsemble(n_models=n_models)
            val_f1s = ensemble.train_ensemble(epochs=epochs)

            self.results['ensemble_training'] = {
                'n_models': n_models,
                'validation_f1s': val_f1s,
                'mean_f1': float(np.mean(val_f1s)),
                'std_f1': float(np.std(val_f1s))
            }

            self.log(f"Ensemble training complete: Mean F1 = {np.mean(val_f1s):.4f}")
            return ensemble

        except Exception as e:
            self.log(f"Ensemble training failed: {e}", 'ERROR')
            traceback.print_exc()
            self.results['errors'].append(f"Ensemble training: {str(e)}")
            return None

    def run_lstm_experiment(self):
        """Run optional LSTM experiment."""
        self.log("")
        self.log("="*60)
        self.log("STEP 4: LSTM EXPERIMENT (OPTIONAL)")
        self.log("="*60)

        # Skip if ensemble mean F1 is already high (>0.92)
        if self.results['ensemble_training']:
            mean_f1 = self.results['ensemble_training']['mean_f1']
            if mean_f1 > 0.92:
                self.log(f"Ensemble F1 ({mean_f1:.4f}) is already high. Skipping LSTM.")
                self.results['lstm_experiment'] = {
                    'skipped': True,
                    'reason': 'Ensemble F1 already high'
                }
                return None

        try:
            from lstm_experiment import LSTMExperiment

            exp = LSTMExperiment()
            f1, success = exp.train(epochs=100, early_abort_epochs=50)

            self.results['lstm_experiment'] = {
                'validation_f1': f1,
                'success': success,
                'skipped': False
            }

            if success:
                self.log(f"LSTM successful: F1 = {f1:.4f}")
            else:
                self.log("LSTM did not outperform CNN. Sticking with ensemble.")

            return exp if success else None

        except Exception as e:
            self.log(f"LSTM experiment failed: {e}", 'ERROR')
            traceback.print_exc()
            self.results['errors'].append(f"LSTM experiment: {str(e)}")
            return None

    def generate_predictions(self, ensemble, thresholds):
        """Generate predictions using ensemble."""
        self.log("")
        self.log("="*60)
        self.log("STEP 5: GENERATING PREDICTIONS")
        self.log("="*60)

        try:
            if ensemble is None:
                # Load ensemble if not provided
                from cnn_ensemble import CNNEnsemble
                ensemble = CNNEnsemble()
                ensemble.load_ensemble()

            results = ensemble.generate_predictions(thresholds=thresholds)
            self.results['predictions_generated'] = True

            # Store distribution for report
            self.results['predictions'] = {
                name: {
                    'count': data['count'],
                    'distribution': {int(k): v for k, v in data['distribution'].items()}
                }
                for name, data in results.items()
            }

            self.log("Predictions generated successfully")
            return results

        except Exception as e:
            self.log(f"Prediction generation failed: {e}", 'ERROR')
            traceback.print_exc()
            self.results['errors'].append(f"Prediction: {str(e)}")
            return None

    def load_backup_distributions(self):
        """Load class distributions from backup for comparison."""
        distributions = {}

        for dataset in ['D2', 'D3', 'D4', 'D5', 'D6']:
            filepath = self.backup_dir / f'{dataset}.mat'
            if filepath.exists():
                data = sio.loadmat(filepath)
                classes = data['Class'].flatten()
                distributions[dataset] = {
                    'count': len(classes),
                    'distribution': dict(Counter(classes))
                }
            else:
                distributions[dataset] = {'count': 0, 'distribution': {}}

        return distributions

    def generate_report(self):
        """Generate final report."""
        self.log("")
        self.log("="*60)
        self.log("STEP 6: GENERATING REPORT")
        self.log("="*60)

        end_time = datetime.now()
        duration = end_time - self.start_time

        # Load backup distributions for comparison
        backup_dist = self.load_backup_distributions()

        # Build report
        lines = []
        lines.append("=" * 78)
        lines.append(" " * 20 + "OPERATION NIGHTWATCH - FINAL REPORT")
        lines.append("=" * 78)
        lines.append("")
        lines.append(f"EXECUTION TIME: {duration}")
        lines.append(f"STARTED AT: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"COMPLETED AT: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Threshold optimization
        lines.append("=" * 78)
        lines.append("THRESHOLD OPTIMIZATION RESULTS")
        lines.append("=" * 78)
        lines.append("")

        if self.results['threshold_optimization']:
            opt = self.results['threshold_optimization']
            lines.append(f"Original F1:  {opt['original_f1']:.4f}")
            lines.append(f"Optimized F1: {opt['optimized_f1']:.4f}")
            lines.append(f"Improvement:  {opt['improvement_pct']:+.2f}%")
            lines.append("")

            lines.append(f"{'Parameter':<20} {'Optimized':>12}")
            lines.append("-" * 34)
            for param, val in opt['thresholds'].items():
                lines.append(f"{param:<20} {val:>12.3f}")
        else:
            lines.append("Threshold optimization failed or skipped")
        lines.append("")

        # Ensemble results
        lines.append("=" * 78)
        lines.append("ENSEMBLE TRAINING RESULTS")
        lines.append("=" * 78)
        lines.append("")

        if self.results['ensemble_training']:
            ens = self.results['ensemble_training']
            lines.append(f"Number of models: {ens['n_models']}")
            lines.append(f"Mean validation F1: {ens['mean_f1']:.4f}")
            lines.append(f"Std validation F1:  {ens['std_f1']:.4f}")
            lines.append("")
            lines.append("Individual model F1 scores:")
            for i, f1 in enumerate(ens['validation_f1s']):
                lines.append(f"  Model {i}: {f1:.4f}")
        else:
            lines.append("Ensemble training failed or skipped")
        lines.append("")

        # LSTM results
        lines.append("=" * 78)
        lines.append("LSTM EXPERIMENT RESULTS")
        lines.append("=" * 78)
        lines.append("")

        if self.results['lstm_experiment']:
            lstm = self.results['lstm_experiment']
            if lstm.get('skipped'):
                lines.append(f"Skipped: {lstm.get('reason', 'N/A')}")
            else:
                lines.append(f"Validation F1: {lstm.get('validation_f1', 'N/A')}")
                lines.append(f"Outperformed CNN: {lstm.get('success', False)}")
        else:
            lines.append("LSTM experiment not run or failed")
        lines.append("")

        # Class distributions
        lines.append("=" * 78)
        lines.append("CLASS DISTRIBUTION COMPARISON")
        lines.append("=" * 78)
        lines.append("")

        # D1 ground truth for reference
        try:
            d1_data = sio.loadmat(self.base_dir / 'datasets' / 'D1.mat')
            d1_classes = d1_data['Class'].flatten()
            d1_dist = Counter(d1_classes)
            d1_total = len(d1_classes)

            lines.append("D1 Ground Truth (Reference):")
            dist_str = "  "
            for c in range(1, 6):
                pct = 100 * d1_dist.get(c, 0) / d1_total
                dist_str += f"C{c}:{pct:4.1f}%  "
            lines.append(dist_str)
            lines.append("")
        except Exception:
            pass

        lines.append(f"{'Dataset':<8} {'Source':<12} {'Total':>6}  " +
                     "".join(f"{'C'+str(c):>8}" for c in range(1, 6)))
        lines.append("-" * 78)

        for dataset in ['D2', 'D3', 'D4', 'D5', 'D6']:
            # Nightly
            if self.results.get('predictions') and dataset in self.results['predictions']:
                nightly = self.results['predictions'][dataset]
                total = nightly['count']
                line = f"{dataset:<8} {'Nightly':<12} {total:>6}  "
                for c in range(1, 6):
                    cnt = nightly['distribution'].get(c, 0)
                    pct = 100 * cnt / total if total > 0 else 0
                    line += f"{pct:>7.1f}%"
                lines.append(line)

            # Backup
            if dataset in backup_dist:
                backup = backup_dist[dataset]
                total = backup['count']
                if total > 0:
                    line = f"{'':<8} {'Backup':<12} {total:>6}  "
                    for c in range(1, 6):
                        cnt = backup['distribution'].get(c, 0)
                        pct = 100 * cnt / total if total > 0 else 0
                        line += f"{pct:>7.1f}%"
                    lines.append(line)

            lines.append("-" * 78)

        lines.append("")

        # Errors
        if self.results['errors']:
            lines.append("=" * 78)
            lines.append("ERRORS ENCOUNTERED")
            lines.append("=" * 78)
            lines.append("")
            for error in self.results['errors']:
                lines.append(f"  - {error}")
            lines.append("")

        # Recommendation
        lines.append("=" * 78)
        lines.append("RECOMMENDATION")
        lines.append("=" * 78)
        lines.append("")

        if self.results['predictions_generated'] and not self.results['errors']:
            lines.append("[*] USE NIGHTLY ENSEMBLE - Optimization completed successfully")
            lines.append("")
            lines.append("Next steps:")
            lines.append("1. Review class distribution for biological plausibility")
            lines.append("2. Run: python src/validate_submission.py")
            lines.append("3. Compare D5/D6 class distributions vs SAFE backup")
            lines.append("4. If satisfied, copy submissions_nightly/* to submissions/")
        else:
            lines.append("[*] KEEP SAFE BACKUP - Errors occurred during optimization")
            lines.append("")
            lines.append("Review errors above and fix before retrying.")

        lines.append("")
        lines.append("=" * 78)

        # Write report
        report_text = "\n".join(lines)
        report_path = self.nightly_dir / 'nightly_report.txt'

        # Make sure directory exists
        self.nightly_dir.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(report_text)

        # Also save JSON version
        json_path = self.nightly_dir / 'nightly_results.json'
        with open(json_path, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                **self.results
            }, f, indent=2, default=str)

        self.log(f"Report saved to: {report_path}")
        self.log(f"JSON results saved to: {json_path}")

        # Print report to console
        print("\n" + report_text)

        return report_path

    def run(self, skip_lstm=False, threshold_maxiter=100, n_ensemble_models=5,
            ensemble_epochs=100):
        """
        Run the complete overnight optimization pipeline.

        Args:
            skip_lstm: Skip LSTM experiment entirely
            threshold_maxiter: Max iterations for threshold optimization
            n_ensemble_models: Number of models in ensemble
            ensemble_epochs: Training epochs per model
        """
        print("""
        ╔═══════════════════════════════════════════════════════════════╗
        ║                                                               ║
        ║              OPERATION NIGHTWATCH                             ║
        ║              Automated Overnight Optimization                  ║
        ║                                                               ║
        ╚═══════════════════════════════════════════════════════════════╝
        """)

        self.log(f"Starting OPERATION NIGHTWATCH at {self.start_time}")
        self.log(f"Configuration:")
        self.log(f"  - Threshold optimization iterations: {threshold_maxiter}")
        self.log(f"  - Ensemble models: {n_ensemble_models}")
        self.log(f"  - Training epochs: {ensemble_epochs}")
        self.log(f"  - Skip LSTM: {skip_lstm}")

        # Step 1: Backup
        if not self.backup_safe_submission():
            self.log("Critical: Backup failed. Aborting.", 'ERROR')
            return

        # Step 2: Threshold optimization
        thresholds = self.run_threshold_optimization(maxiter=threshold_maxiter)

        # Step 3: Ensemble training
        ensemble = self.train_ensemble(
            n_models=n_ensemble_models,
            epochs=ensemble_epochs
        )

        # Step 4: Optional LSTM
        if not skip_lstm and ensemble is not None:
            self.run_lstm_experiment()

        # Step 5: Generate predictions
        if ensemble is not None:
            self.generate_predictions(ensemble, thresholds)

        # Step 6: Generate report
        self.generate_report()

        self.log("")
        self.log("="*60)
        self.log("OPERATION NIGHTWATCH COMPLETE")
        self.log("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='OPERATION NIGHTWATCH - Automated Overnight Optimization'
    )
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM experiment')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run (reduced iterations/epochs)')
    parser.add_argument('--threshold-iter', type=int, default=100,
                        help='Threshold optimization iterations')
    parser.add_argument('--n-models', type=int, default=5,
                        help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per model')
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.threshold_iter = 20
        args.epochs = 30
        args.skip_lstm = True
        print("*** QUICK MODE: Reduced iterations for testing ***\n")

    runner = NightlyRunner()
    runner.run(
        skip_lstm=args.skip_lstm,
        threshold_maxiter=args.threshold_iter,
        n_ensemble_models=args.n_models,
        ensemble_epochs=args.epochs
    )


if __name__ == '__main__':
    main()
