# ♻️ AI Driven Circular Economy Tracker  
### Predicting Circularity-Score & Detecting High-Waste Events with Machine Learning

This project applies **AI and data-driven analytics** to track and optimize **circular economy performance** in industrial facilities.  
Using historical operations data, the system predicts:

- **Regression:** `Circularity_score` → Sustainability Benchmarking (representing how circular a facility's material usage is)   
- **Classification:** `high_waste_flag` → Binary indicator of unusually high waste events  

The goal is to help organizations **reduce waste**, **increase resource recovery**, and **move toward sustainable manufacturing**.

### Project Workflow :
1. Data Preprocessing :

Handling missing values
Outlier fixing
Feature engineering 
Dropping redundant columns 
Encoding categorical variables 
Scaling numerical variables using StandardScaler

2️.Train-Test Split (Before Scaling)

Ensures no data leakage.

3️. Feature Scaling

Only continuous numeric variables were scaled.

4️. Modeling

Ridge Regression for predicting circularity score

Logistic Regression for classifying high waste events

5️. Hyperparameter Tuning

RandomizedSearchCV used for:

Ridge regression (alpha, solver, etc.)

Logistic regression (C, penalty, solver)

6️. Model Saving

Models saved as .pkl files for deployment:

best_ridge_model.pkl

best_logistic_model.pkl

scaler.pkl

Encoders (if applicable)

7️. Deployment (Flask)

A Flask API receives user input, applies preprocessing, and outputs predictions.

## Model Performance Summary :
### Ridge Regression :

Good performance with reduced overfitting

Stable because of L2 regularization

### Logistic Regression :

Efficient for binary waste prediction

Works well with standardized inputs

## Technologies Used

Python

Pandas / NumPy

Matplotlib / Seaborn

Scikit-Learn

RandomizedSearchCV

Pickle

Flask

## Conclusion :

This project showcases a complete, production-ready AI pipeline for predicting and analyzing circular economy KPIs. By integrating feature engineering, model training, hyperparameter tuning, and Flask deployment, it provides a powerful template for real-world sustainability intelligence systems.
