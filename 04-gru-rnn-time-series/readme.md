# GRU / RNN for Time-Series Classification

Applies recurrent neural networks (GRU) to the human activity recognition problem, treating each IMU recording as a time-series sequence rather than a flat feature vector. Explores 18 architecture configurations, uses 5-fold cross-validation, and runs a keras-tuner hyperparameter search to find the best model.

**Task:** Classify person identity + activity from IMU sequences (8 classes: 4 people × walking/running/jumping/stairs).
**Result:** ~91.78% classification accuracy on the held-out test set.

---

## Folder Structure

```
04-gru-rnn-time-series/
├── pickles/                  # Processed DataFrames (train/valid/test splits)
│   ├── train_df.pkl
│   ├── valid_df.pkl
│   ├── test_df.pkl
│   ├── Accelerometer.csv     # Raw sensor data included for reference
│   └── Gyroscope.csv
├── notebooks/
│   ├── gru.ipynb             # Full GRU training, search, and evaluation
│   └── best_model.h5         # Saved best model weights
├── environment.yml
└── readme.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate ml_lab
jupyter notebook
```

## What the Notebook Does

**Stage 1 — Architecture exploration:**
- Tests 18 GRU configurations (varying units, layers, dropout, bidirectionality)
- 5-fold cross-validation on the training set
- Compares simple GRU, stacked GRU, and bidirectional GRU

**Stage 2 — Hyperparameter search:**
- Uses `keras-tuner` to search over the best-performing architecture family
- Selects the optimal config based on validation accuracy
- Trains the final model and saves weights to `best_model.h5`

**Evaluation:**
- Reports accuracy on the held-out test set
- Generates confusion matrix across all 8 classes

## Results

| Configuration | Test Accuracy |
|---------------|--------------|
| Best GRU (8-class) | ~91.78% |

Outputs (saved in notebook): per-fold CV results across all 18 configs, keras-tuner search summary, training curves, confusion matrix.
