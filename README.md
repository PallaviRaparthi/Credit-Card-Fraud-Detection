
# 💳 Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent
credit card transactions using Logistic Regression and Random Forest.

## Features

- Transaction data preprocessing
- Feature engineering using transaction time and location
- Logistic Regression and Random Forest model comparison
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrix visualization
- Fraud probability threshold analysis
- Interactive Streamlit dashboard
- New transaction risk prediction

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/kartik2112/fraud-detection

Place `fraudTrain.csv` and `fraudTest.csv` in the project folder.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Evaluation Metrics

The application compares models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Disclaimer

This is an educational fraud-detection prototype using a
simulated dataset. Predictions are estimates and should not
be treated as proof of fraud or production banking decisions.
