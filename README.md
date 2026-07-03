# Human Activity Recognition — IMU ML Pipeline

End-to-end machine learning pipeline for classifying human activities from smartphone IMU sensor data (accelerometer + gyroscope). Built as a progressive series of labs, starting from raw data acquisition and advancing through classical ML to deep recurrent neural networks.

## Labs

| Lab | What it does | Key result |
|-----|-------------|------------|
| [01-imu-data-acquisition](./01-imu-data-acquisition/) | Load raw CSV sensor recordings, combine accelerometer + gyroscope, save pickles | 3-class phone position dataset ready for downstream use |
| [02-classical-ml-knn-decision-tree](./02-classical-ml-knn-decision-tree/) | KNN and Decision Tree classifiers with GridSearchCV tuning; normalization analysis | KNN 91.7% · Decision Tree 87.5% test accuracy |
| [03-feedforward-neural-network](./03-feedforward-neural-network/) | MLP with 2 hidden layers (10 vs 50 neurons), EarlyStopping, confusion matrix analysis | 93.29% test accuracy (50-neuron model) |
| [04-gru-rnn-time-series](./04-gru-rnn-time-series/) | Time-series GRU with 18 architecture configs, 5-fold CV, keras-tuner hyperparameter search | ~91.78% person + activity classification accuracy |

## Dataset

- **Lab 1:** Phone position data — `laying_down`, `standing_up`, `circle`. 3 recording sessions per class.
- **Labs 2–4:** Human activity data — `walking`, `running`, `jumping`, `stairs`. 8 recording sessions per class, split 5 train / 2 validation / 1 test.

Data was collected using a smartphone IMU (accelerometer + gyroscope) and is included in each lab's `data/` and `pickles/` directories.

## Stack

- Python · Jupyter Notebooks · Anaconda
- pandas · numpy · matplotlib · seaborn
- scikit-learn (KNN, Decision Tree, GridSearchCV, MinMaxScaler)
- TensorFlow / Keras (MLP, GRU, RNN)
- keras-tuner

## Outputs

All notebooks include saved outputs — confusion matrices, training curves, and accuracy metrics are embedded directly in the `.ipynb` files and can be viewed on GitHub without running the code.

## Setup

Each lab has its own `environment.yml`. To set up any lab:

```bash
cd <lab-folder>
conda env create -f environment.yml
conda activate ml_lab
jupyter notebook
```
