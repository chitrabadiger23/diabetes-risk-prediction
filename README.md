
# 🩺 Diabetes Risk Predictor

A machine learning based web application that predicts diabetes status using patient health parameters.

## Features

- Diabetes prediction using SVM
- Model-estimated probability
- Interactive Streamlit dashboard
- Patient parameter comparison
- Confusion matrix
- Accuracy, Precision, Recall and F1 Score
- User-friendly interface

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Machine Learning Model

Support Vector Machine (SVM) is used for classification.

The input features are standardized using StandardScaler before prediction.

## Input Parameters

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```
