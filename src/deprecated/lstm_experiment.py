"""
BiLSTM Experiment for Spike Classification (OPTIONAL)

This is a LOW PRIORITY experiment. The CNN approach is preferred because:
1. Fixed-length input (60 samples): CNNs excel at fixed-length patterns
2. Limited training data (~2000 samples): LSTMs need more data
3. No temporal dependencies: Spikes are isolated events, not sequences
4. Training time: Bi-LSTM would take 3-5x longer than CNN

This is implemented as an OPTIONAL FALLBACK only if the ensemble underperforms.
Abort condition: If validation F1 < CNN baseline after 50 epochs, skip LSTM.
"""

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
from collections import Counter
import pickle
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from spike_detector import detect_spikes
from feature_extractor import extract_waveforms_at_indices
from cnn_experiment import CNNExperiment


class BiLSTMSpikeNet(nn.Module):
    """
    Bidirectional LSTM for spike classification.

    Architecture:
    - BiLSTM: Processes waveform sequence in both directions
    - Amplitude branch: Same as CNN (MLP for amplitude features)
    - Combined classifier: Fuses LSTM output with amplitude features
    """

    def __init__(self, input_size=60, hidden_size=64, num_layers=2,
                 num_amp_features=8, num_classes=5, dropout=0.3):
        super(BiLSTMSpikeNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM for waveform sequence
        self.lstm = nn.LSTM(
            input_size=1,  # Single channel input
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism (optional but helps)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Amplitude branch (same as CNN)
        self.amp_fc1 = nn.Linear(num_amp_features, 32)
        self.amp_bn1 = nn.BatchNorm1d(32)
        self.amp_fc2 = nn.Linear(32, 32)
        self.amp_bn2 = nn.BatchNorm1d(32)

        # Combined classifier
        # hidden_size * 2 (bidirectional) + 32 (amplitude)
        combined_size = hidden_size * 2 + 32
        self.fc1 = nn.Linear(combined_size, 48)
        self.fc2 = nn.Linear(48, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, waveform, amp_features):
        """
        Forward pass.

        Args:
            waveform: (batch, seq_len) - normalized waveform
            amp_features: (batch, 8) - amplitude features
        """
        batch_size = waveform.size(0)

        # Reshape for LSTM: (batch, seq_len, 1)
        x = waveform.unsqueeze(-1)

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * 2)

        # Attention-weighted sum
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_size * 2)

        # Alternative: use last hidden states
        # context = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Amplitude branch
        amp = self.amp_fc1(amp_features)
        amp = self.amp_bn1(amp)
        amp = self.relu(amp)
        amp = self.amp_fc2(amp)
        amp = self.amp_bn2(amp)
        amp = self.relu(amp)

        # Combined classifier
        combined = torch.cat([context, amp], dim=1)
        combined = self.dropout(combined)
        combined = self.relu(self.fc1(combined))
        combined = self.dropout(combined)
        output = self.fc2(combined)

        return output


class LSTMExperiment:
    """LSTM experiment for spike classification (optional fallback)."""

    def __init__(self, device=None):
        self.device = device if device else (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f"LSTMExperiment using device: {self.device}")

        self.model = None
        self.amp_mean = None
        self.amp_std = None

        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'datasets'
        self.model_dir = self.base_dir / 'models'

        # CNN baseline F1 for abort condition
        self.cnn_baseline_f1 = None

    def get_cnn_baseline(self):
        """Get CNN baseline F1 for comparison."""
        if self.cnn_baseline_f1 is not None:
            return self.cnn_baseline_f1

        # Try to load from ensemble metadata
        ensemble_meta = self.model_dir / 'ensemble' / 'ensemble_metadata.json'
        if ensemble_meta.exists():
            import json
            with open(ensemble_meta) as f:
                meta = json.load(f)
                self.cnn_baseline_f1 = meta.get('mean_f1', 0.90)
        else:
            # Default baseline
            self.cnn_baseline_f1 = 0.90

        print(f"CNN baseline F1: {self.cnn_baseline_f1:.4f}")
        return self.cnn_baseline_f1

    def train(self, epochs=100, batch_size=64, lr=0.001, patience=15,
              early_abort_epochs=50):
        """
        Train LSTM model.

        Args:
            epochs: Maximum training epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            early_abort_epochs: Epochs before checking abort condition
        """
        print("\n" + "="*60)
        print("LSTM EXPERIMENT (OPTIONAL FALLBACK)")
        print("="*60)

        cnn_baseline = self.get_cnn_baseline()

        # Create base experiment for data loading
        base_exp = CNNExperiment(device=self.device)

        # Load D1 data
        print("\nLoading D1 training data...")
        d, index, classes = base_exp.load_d1_data()

        # Extract waveforms
        waveforms, valid_indices = extract_waveforms_at_indices(
            d, index, window_before=30, window_after=30
        )
        valid_mask = np.isin(index, valid_indices)
        valid_classes = classes[valid_mask]

        print(f"Total waveforms: {len(waveforms)}")

        # Augment data
        aug_waveforms, aug_labels = base_exp.augment_data(waveforms, valid_classes)
        print(f"Augmented samples: {len(aug_waveforms)}")

        # Split
        train_wf, val_wf, train_lbl, val_lbl = train_test_split(
            aug_waveforms, aug_labels, test_size=0.2,
            random_state=42, stratify=aug_labels
        )

        # Prepare data
        train_wf_norm, train_amp, train_lbl = base_exp.prepare_data(
            train_wf, train_lbl, fit_scalers=True
        )
        self.amp_mean = base_exp.amp_mean
        self.amp_std = base_exp.amp_std

        val_wf_norm, val_amp, val_lbl = base_exp.prepare_data(val_wf, val_lbl)

        # Convert to tensors
        X_wf_train = torch.FloatTensor(train_wf_norm).to(self.device)
        X_amp_train = torch.FloatTensor(train_amp).to(self.device)
        y_train = torch.LongTensor(train_lbl - 1).to(self.device)

        X_wf_val = torch.FloatTensor(val_wf_norm).to(self.device)
        X_amp_val = torch.FloatTensor(val_amp).to(self.device)
        y_val = torch.LongTensor(val_lbl - 1).to(self.device)

        # Create data loaders
        class_counts = Counter(train_lbl)
        weights = [1.0 / class_counts[lbl] for lbl in train_lbl]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dataset = TensorDataset(X_wf_train, X_amp_train, y_train)
        val_dataset = TensorDataset(X_wf_val, X_amp_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = BiLSTMSpikeNet(
            input_size=60, hidden_size=64, num_layers=2,
            num_amp_features=8, num_classes=5, dropout=0.3
        ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel parameters: {n_params:,}")

        # Loss with class weights
        class_weights = torch.FloatTensor(
            [1.0 / class_counts[c] for c in range(1, 6)]
        ).to(self.device)
        class_weights = class_weights / class_weights.sum() * 5
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

        # Training loop
        print("\nTraining...")
        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch_wf, batch_amp, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_wf, batch_amp)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch_wf, batch_amp, batch_y in val_loader:
                    outputs = self.model(batch_wf, batch_amp)
                    _, predicted = outputs.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())

            val_f1 = f1_score(val_true, val_preds, average='weighted')
            scheduler.step(val_f1)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss={train_loss/len(train_loader):.4f}, "
                      f"Val F1={val_f1:.4f}")

            # Check abort condition after early_abort_epochs
            if epoch + 1 == early_abort_epochs:
                if val_f1 < cnn_baseline - 0.05:  # 5% below CNN
                    print(f"\n*** ABORT: LSTM F1 ({val_f1:.4f}) < "
                          f"CNN baseline ({cnn_baseline:.4f}) - 0.05 ***")
                    print("LSTM experiment aborted. CNN remains the better choice.")
                    return None, False

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        print("\n" + "="*60)
        print("LSTM EXPERIMENT RESULTS")
        print("="*60)
        print(f"\nBest validation F1: {best_val_f1:.4f}")
        print(f"CNN baseline F1: {cnn_baseline:.4f}")

        if best_val_f1 > cnn_baseline:
            print("\n*** LSTM OUTPERFORMS CNN ***")
            improvement = 100 * (best_val_f1 - cnn_baseline) / cnn_baseline
            print(f"Improvement: +{improvement:.2f}%")
            success = True
        else:
            print("\nLSTM does not improve over CNN. Sticking with CNN.")
            success = False

        # Save if successful
        if success:
            self.save_model()

        return best_val_f1, success

    def save_model(self):
        """Save trained model."""
        state = {
            'model_state': self.model.state_dict(),
            'amp_mean': self.amp_mean,
            'amp_std': self.amp_std,
            'timestamp': datetime.now().isoformat(),
        }
        filepath = self.model_dir / 'lstm_model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"LSTM model saved to: {filepath}")


def run_lstm_experiment(max_epochs=100, early_abort=50):
    """Run LSTM experiment as optional fallback."""
    exp = LSTMExperiment()
    f1, success = exp.train(epochs=max_epochs, early_abort_epochs=early_abort)

    if success:
        return f1, exp.model
    else:
        return None, None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LSTM Experiment (Optional)')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--early-abort', type=int, default=50,
                        help='Epochs before checking abort condition')
    args = parser.parse_args()

    print("NOTE: This is an OPTIONAL experiment.")
    print("If LSTM underperforms CNN after 50 epochs, it will be aborted.")

    run_lstm_experiment(max_epochs=args.epochs, early_abort=args.early_abort)
