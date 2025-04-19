# Time-Series Forecasting using MLP

This project focuses on predicting the next state in a sequence of temporal data using a **Multilayer Perceptron (MLP)** model. The goal is to model the underlying patterns of state trajectories over time and make accurate one-step-ahead predictions. The project was developed as part of an individual assignment and evaluated via a private Kaggle competition leaderboard.

## Dataset

The dataset consists of time-series trajectories representing sequences of system states. It is divided into three parts:

- train.csv — used to train the model (with thousands of state trajectories)
- val.csv — used to validate the model performance
- test.csv — used for generating predictions for competition evaluation

Each trajectory is composed of time-dependent features and is indexed by a unique ID. The model uses a **sliding window approach** to frame each prediction task as a supervised learning problem.

## Methods

### ➤ Data Preprocessing:
- Converted raw trajectories into overlapping windows using a sliding window view.
- Separated input sequences and their corresponding target (next state).

### ➤ Model Architecture:
- Implemented a **Multilayer Perceptron (MLP)** with:
  - Input layer matching window size
  - Two hidden layers with ReLU activations
  - Output layer for next state prediction

### ➤ Training:
- Used **Mean Squared Error (MSE)** loss function
- Optimized using **Adam optimizer**
- Batch processing via `DataLoader`

### ➤ Validation:
- Evaluated on the validation set using MSE
- Generated plots comparing actual vs predicted sequences for qualitative analysis

## Tools and Libraries

- **Python 3.9+**
- **PyTorch** — for building and training the MLP model
- **NumPy** — numerical processing
- **Pandas** — data manipulation
- **Matplotlib** — plotting predicted vs actual trajectories
- **Google Colab** — development and training environment
- **Kaggle** — competition platform for performance evaluation
