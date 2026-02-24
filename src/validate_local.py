import numpy as np
import scipy.io as sio
from generate_submissions_fixed import detect_and_classify_matched_filter, extract_templates_aligned

# 1. Load Clean D1
data = sio.loadmat('datasets/D1.mat')
d_clean = data['d'].flatten()
ground_truth_indices = data['Index'].flatten()

# 2. Generate "Fake D6" (Add heavy noise)
# D6 is <0dB, meaning Noise Power > Signal Power. 
# We estimate noise sigma based on signal amplitude.
np.random.seed(42)
noise_sigma = np.std(d_clean) * 1.5  # Adjust 1.5 until SNR looks like D6
d_noisy = d_clean + np.random.normal(0, noise_sigma, len(d_clean))

# 3. Extract Templates (from clean data)
templates = extract_templates_aligned('datasets/D1.mat')

# 4. Run YOUR Pipeline on Fake D6
print("Running pipeline on Fake D6...")
# Try different thresholds here to find the sweet spot!
indices, waveforms, classes, corrs = detect_and_classify_matched_filter(
    d_noisy, templates, correlation_threshold=0.75 
)

# 5. Calculate Score Locally
print(f"\nDetected: {len(indices)} (Ground Truth: {len(ground_truth_indices)})")

# Simple "Match" check (within 50 samples)
hits = 0
for true_idx in ground_truth_indices:
    # Find closest detected spike
    dist = np.min(np.abs(indices - true_idx))
    if dist < 50:
        hits += 1

recall = hits / len(ground_truth_indices)
precision = hits / len(indices) if len(indices) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) 

print(f"ESTIMATED D6 PERFORMANCE:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")