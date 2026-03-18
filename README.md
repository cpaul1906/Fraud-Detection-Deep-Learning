# Credit Card Fraud Detection with Deep Learning

This project builds a deep learning workflow for credit card fraud detection using Python. It demonstrates how machine learning can be applied to transaction data to identify suspicious behavior, manage class imbalance, and evaluate fraud risk in a financial context.

## Overview

Fraud detection is a classic high-impact machine learning problem because fraudulent transactions are rare, costly, and difficult to identify in real time. In this project, I apply a deep learning approach to transaction data to classify fraudulent versus legitimate activity.

The notebook is designed to show an end-to-end fraud analytics workflow, from preprocessing and modeling to evaluation and interpretation.

## Project Goal

The goal of this project is to build a practical fraud detection pipeline that can learn patterns in transaction data and help distinguish normal activity from potentially fraudulent behavior.

## Tools and Technologies

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## What the Notebook Does

This project includes the following steps:

- Loads and explores credit card transaction data
- Cleans and prepares the dataset for modeling
- Addresses class imbalance in fraud labels
- Builds and trains a deep learning classification model
- Evaluates performance using classification metrics
- Interprets results in the context of fraud detection

## Dataset and Problem Context

Fraud datasets are typically highly imbalanced, with legitimate transactions greatly outnumbering fraudulent ones. That makes this a strong example of a real-world classification challenge where accuracy alone is not enough.

The project focuses on detecting rare fraud events while balancing predictive performance and interpretability.

## Model Summary

The notebook uses a deep learning model for binary classification:

- **Task:** Fraudulent vs. legitimate transaction classification
- **Approach:** Supervised deep learning
- **Focus areas:** Preprocessing, imbalance-aware evaluation, prediction performance

### Core workflow includes:
- Feature preparation
- Train/test split
- Model training
- Prediction and threshold-based classification
- Performance review

## Evaluation

The model is evaluated using standard classification metrics appropriate for fraud detection. Because this is an imbalanced classification problem, the notebook emphasizes meaningful performance analysis rather than relying only on raw accuracy.

## Business Relevance

This project reflects a real financial analytics use case and demonstrates skills relevant to:

- Fraud detection
- Risk analytics
- Anomaly detection
- Applied machine learning in finance
- Classification under class imbalance

A workflow like this could support fraud monitoring teams by helping flag suspicious transactions for review and improving decision support in high-volume payment environments.

## Next Steps

Potential future improvements include:

- Comparing the deep learning model against logistic regression, random forest, and gradient boosting
- Adding precision, recall, F1-score, and ROC-AUC analysis in more detail
- Testing threshold optimization for fraud screening
- Applying explainability tools to understand model predictions
- Extending the workflow to near-real-time fraud scoring

## File

- `credit_card_fraud_detection_deep_learning_portfolio.ipynb` — notebook containing preprocessing, model training, evaluation, and fraud analysis

## Note

This project was originally developed as graduate coursework and is being shared as part of my technical portfolio to demonstrate applied machine learning, financial risk modeling, and fraud detection workflow development.

## Author

Christopher Paul
