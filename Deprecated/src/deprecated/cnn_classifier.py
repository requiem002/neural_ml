"""1D CNN classifier for spike classification - more robust to noise."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import signal


class SpikeNet(nn.Module):
    """1D CNN for spike waveform classification."""

    def __init__(self, input_size=60, num_classes=5):
        super(SpikeNet, self).__init__()

        # Convolutional layers - learn hierarchical features from waveforms
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate size after convolutions and pooling
        # After 3 pools: input_size / 8
        conv_output_size = input_size // 8 * 128

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CNNSpikeClassifier:
    """CNN-based spike classifier wrapper."""

    def __init__(self, input_size=60, num_classes=5, device=None):
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SpikeNet(input_size=input_size, num_classes=num_classes).to(self.device)
        self.scaler = StandardScaler()

        # Filter for preprocessing
        nyquist = 25000 / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        self.filter_b, self.filter_a = signal.butter(2, [low, high], btype='band')

    def preprocess(self, waveforms):
        """Preprocess waveforms: filter and normalize."""
        processed = []
        for wf in waveforms:
            # Bandpass filter
            try:
                wf_filt = signal.filtfilt(self.filter_b, self.filter_a, wf, padlen=min(15, len(wf)-1))
            except ValueError:
                wf_filt = wf

            # Subtract baseline
            baseline = np.mean(wf_filt[:5])
            wf_centered = wf_filt - baseline

            processed.append(wf_centered)

        return np.array(processed)

    def fit(self, waveforms, labels, epochs=100, batch_size=64, lr=0.001):
        """Train the CNN classifier."""
        # Preprocess
        X = self.preprocess(waveforms)
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(labels - 1).to(self.device)  # Classes 0-4 for PyTorch

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            acc = 100. * correct / total
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')

        return self

    def predict(self, waveforms):
        """Predict class labels for waveforms."""
        # Preprocess
        X = self.preprocess(waveforms)
        X = self.scaler.transform(X)

        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy() + 1  # Convert back to classes 1-5

    def predict_proba(self, waveforms):
        """Predict class probabilities."""
        X = self.preprocess(waveforms)
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def evaluate(self, waveforms, labels, verbose=True):
        """Evaluate classifier performance."""
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        y_pred = self.predict(waveforms)

        f1_weighted = f1_score(labels, y_pred, average='weighted')
        f1_per_class = f1_score(labels, y_pred, average=None)

        if verbose:
            print("\nClassification Report:")
            print(classification_report(labels, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(labels, y_pred))
            print(f"\nWeighted F1 Score: {f1_weighted:.4f}")

        return {
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'predictions': y_pred
        }

    def save(self, filepath):
        """Save model state."""
        import pickle
        state = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath):
        """Load model state."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.scaler = state['scaler']
        self.input_size = state['input_size']
        self.num_classes = state['num_classes']
        self.model = SpikeNet(input_size=self.input_size, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(state['model_state'])
        return self
