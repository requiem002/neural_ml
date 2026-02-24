Neural Spike Sorting Coursework
========================================================================

1. PROJECT OVERVIEW
-------------------
This project implements a hybrid approach for spike sorting:
- High SNR Data (D2, D3, D4): Uses a 1D CNN trained on waveform shape.
- Low SNR Data (D5, D6): Uses Template Matching (Matched Filter) for robustness against noise.

2. FOLDER STRUCTURE
-------------------
Submission/
├── README.txt              # This file
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── train_model.py          # CNN training script
│   └── generate_submissions.py # Prediction pipeline
├── datasets/               # Place D1.mat - D6.mat here (omitted the dataset files to avoid moodle submission issues!)
└── models/                 # Saved models (generated during training)

3. INSTALLATION
---------------
Ensure Python 3.8+ is installed. Install dependencies using:
   pip install -r requirements.txt

4. HOW TO RUN
-------------
The pipeline consists of two stages: training and inference.

STEP 1: TRAIN THE MODEL
Run the training script to train the CNN on D1 and validate performance.
   python src/train_model.py --train --validate

   * Output: Saves 'cnn_model_fixed.pkl' to the 'models/' directory.
   * Expected D1 Accuracy: ~92-95%

STEP 2: GENERATE SUBMISSIONS
Run the submission generator to create the final .mat files for D2-D6.
   python src/generate_submissions.py

   * Output: Saves 'D2.mat' ... 'D6.mat' to the 'submissions/' directory.
   * Note: This script automatically selects the best classification strategy
     (CNN vs. Template Matching) based on the noise level of the dataset.

========================================================================