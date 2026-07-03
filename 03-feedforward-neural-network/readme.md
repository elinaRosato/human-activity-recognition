# Feedforward Neural Network (MLP)

Replaces the classical classifiers from Lab 2 with a feedforward neural network. Compares two MLP configurations (10 vs 50 hidden neurons), applies learning rate scheduling and early stopping, and evaluates on the same activity dataset.

**Task:** Classify 4 activities — `walking`, `running`, `jumping`, `stairs`.
**Result:** 93.29% test accuracy (50-neuron model) — outperforms KNN and Decision Tree from Lab 2.

---

## Folder Structure

```
03-feedforward-neural-network/
├── data/                     # Raw CSV recordings (same dataset as Lab 2)
├── pickles/                  # Processed DataFrames (same structure as Lab 2)
│   ├── original/
│   ├── cleaned/
│   └── combined/
├── notebooks/
│   ├── setup.ipynb                   # Data loading, cleaning, splitting
│   └── forward_neural_network.ipynb  # MLP training and evaluation
├── environment.yml
└── readme.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate ml_lab
jupyter notebook
```

## Run Order

1. **`setup.ipynb`** — same data pipeline as Lab 2 (re-run if starting fresh)
2. **`forward_neural_network.ipynb`** — train and compare MLP models

## What the Notebook Does

1. Builds two MLP models with 2 hidden layers — 10 neurons and 50 neurons
2. Trains with `EarlyStopping` (patience=10) and `ReduceLROnPlateau`
3. Plots accuracy and loss curves for both models
4. Evaluates on the test set and compares train / validation / test accuracy
5. Generates and plots a confusion matrix

## Results

| Model | Test Accuracy |
|-------|--------------|
| MLP — 10 neurons | ~88% |
| MLP — 50 neurons | 93.29% |

Outputs (saved in notebook): accuracy/loss curves per epoch, confusion matrix, final accuracy scores per split.
