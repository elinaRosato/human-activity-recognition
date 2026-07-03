# IMU Data Acquisition & Preprocessing

Introduces the full data pipeline: reading raw smartphone IMU recordings (accelerometer + gyroscope), combining sensor streams into DataFrames, and serializing to pickle files for downstream use.

**Task:** Classify 3 phone positions — `laying_down`, `standing_up`, `circle`.
**Note:** This lab focuses on data handling, not model training. The cleaned dataset is used directly in subsequent labs.

---

## Folder Structure

```
01-imu-data-acquisition/
├── data/                     # Raw CSV recordings (Accelerometer & Gyroscope)
│   ├── laying_down_1/
│   ├── laying_down_2/
│   ├── laying_down_3/
│   ├── standing_up_1/
│   ├── standing_up_2/
│   ├── standing_up_3/
│   ├── circle_1/
│   ├── circle_2/
│   └── circle_3/
├── pickles/                  # Serialized DataFrames
│   ├── training/
│   │   └── concat/           # Combined training set
│   └── test/
├── notebooks/
│   └── lab1.ipynb
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

1. Loads `Accelerometer.csv` and `Gyroscope.csv` from each recording session
2. Merges sensor streams into a single DataFrame per session
3. Splits sessions into training (2 sessions/class) and test (1 session/class)
4. Saves individual and concatenated pickle files to `pickles/`

Outputs (saved in notebook): data shape summaries, sample plots of accelerometer and gyroscope readings.
