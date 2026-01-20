# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **neural spike sorting and classification** project (EE40098 Coursework C). The goal is to detect and classify neuronal action potentials (spikes) from simulated brain recordings into 5 neuron classes. Please refer to the "EE40098 CourseWork C Assessment brief" pdf document for more information.

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

Package predictions in one file named "submissions", and each of the .mat files must be labelled as "d2.mat, d3.mat" through to "d6.mat".

## Python Dependencies (Expected)

```
scipy          # .mat file I/O
numpy          # Numerical computing
matplotlib     # Visualization
scikit-learn   # ML algorithms and metrics
```
