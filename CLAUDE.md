# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **neural spike sorting and classification** project (EE40098 Coursework C). The goal is to detect and classify neuronal action potentials (spikes) from simulated brain recordings into 5 neuron classes. Please refer to the "EE40098 CourseWork C Assessment brief" pdf document for more information.

## Critical Resources (Forensic Analysis)

You must use these files to ground your decisions:
1.  **`feedback file from friend.pdf` (The Benchmark):** Can be treated as somehwhat of an exemplar, particularly for the datasets 2 to 4. Datasets 5 and 6 are also decent but definitely not perfect. Use the Confusion Matrices along with the precision and recall scores for each dataset in this file to reverse-engineer the *approximate true* spike count per class.
2.  **`saad_feedback_latest.pdf` (Our Baseline):** My latest feedback file for the submission datasets you produced. Poor scores, especially for the later datasets. Even dataset 2 isn't good enough. Shows where we are failing (e.g., D6 Recall = 0.31).
3.  **`EE40098 Coursework C...pdf` (The Brief):** Contains the marking scheme. Note that D6 is weighted highest (0.3).

**Key challenge:** Build a system robust to decreasing signal-to-noise ratios (80dB to <0dB). This is particularly important in datasets 5 and 6, which have the lowest SNRs. 

## Datasets

Located in `datasets/`:
- **D1.mat** - Training data (80dB SNR, fully labeled with Index and Class vectors)
- **D2-D6.mat** - Test data (progressively noisier: 60dB → <0dB, unlabeled)

Dataset structure (MATLAB format):
- `d` - Recording signal (1×1,440,000 samples at 25kHz = 57.6 seconds)
- `Index` - Spike locations in samples (only in D1)
- `Class` - Neuron class labels 1-5 (only in D1)

Load with: `scipy.io.loadmat('datasets/D1.mat')`

## Expected Architecture

The solution should implement:

1. **Spike Detection** - Find spike occurrences in time-domain signal
2. **Feature Extraction** - Extract waveform features around detected peaks
3. **Classification** - Assign detected spikes to one of 5 neuron classes
4. **Output Generation** - Save Index and Class vectors to .mat format

## Evaluation Metrics

- Detection tolerance: ±50 samples from ground truth
- Combined F1 score: `F = 0.3×F_detection + 0.7×F_classification`
- Final score weighted by dataset difficulty (D6 weighted highest at 0.3)

## Output Format

For each test dataset (D2-D6), produce a .mat file containing:
- `Index` - Detected spike locations (sample indices)
- `Class` - Predicted neuron class (1-5) for each spike

For each test dataset (D2-D6), produce a `.mat` file within the submissions folder containing:
-   **Filename:** Uppercase (e.g., `D2.mat`, `D3.mat`).
-   **Structure:** `1xN` Row Vectors (NOT Column vectors).
-   **Keys:** `'Index'` and `'Class'` (Case sensitive).

## Python Dependencies (Expected)

```
scipy          # .mat file I/O
numpy          # Numerical computing
matplotlib     # Visualization
scikit-learn   # ML algorithms and metrics
```
