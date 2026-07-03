# Classical ML: KNN & Decision Tree

Classifies human activities from smartphone IMU data using two classical ML approaches: K-Nearest Neighbors and Decision Tree. Includes hyperparameter tuning with GridSearchCV, normalization analysis, and a full comparison via confusion matrices.

**Task:** Classify 4 activities — `walking`, `running`, `jumping`, `stairs`.
**Results:** KNN 91.7% · Decision Tree 87.5% test accuracy.

---

## Folder Structure

```
02-classical-ml-knn-decision-tree/
├── data/                     # Raw CSV recordings (8 sessions per class)
│   ├── walking_1/ ... walking_8/
│   ├── running_1/ ... running_8/
│   ├── jumping_1/ ... jumping_8/
│   └── stairs_1/  ... stairs_8/
├── pickles/
│   ├── original/             # Raw merged DataFrames
│   │   ├── train/            # Sessions 1–5 per class
│   │   ├── valid/            # Sessions 6–7 per class
│   │   └── test/             # Session 8 per class
│   ├── cleaned/              # Trimmed versions (irregular samples removed)
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── combined/             # Single train/valid/test DataFrames
│       ├── train_df.pkl
│       ├── valid_df.pkl
│       └── test_df.pkl
├── notebooks/
│   ├── setup.ipynb           # Data loading, cleaning, splitting
│   ├── normalisation.ipynb   # Effect of MinMaxScaler on model performance
│   ├── knn.ipynb             # KNN classifier
│   └── decision_tree.ipynb   # Decision Tree classifier
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

1. **`setup.ipynb`** — load raw CSVs, clean, split 5/2/1 train/valid/test, save pickles
2. **`normalisation.ipynb`** — compare normalized vs unnormalized KNN performance (optional)
3. **`knn.ipynb`** — KNN with GridSearchCV over k; evaluates on test set
4. **`decision_tree.ipynb`** — Decision Tree with GridSearchCV over depth and criterion; visualizes tree

## Results

| Model | Test Accuracy |
|-------|--------------|
| KNN (best k by GridSearchCV) | 91.7% |
| Decision Tree (best depth + criterion) | 87.5% |

Outputs (saved in notebooks): accuracy scores, confusion matrices, decision tree visualization, GridSearchCV results.
