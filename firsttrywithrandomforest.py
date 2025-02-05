#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:09:35 2025

@author: irakulkarni
"""

import pandas as pd
og = pd.read_csv("heart_disease_uci.csv")  


#cleaning data set
columns_to_keep = ["age", "trestbps", "chol", "fbs", "num"]
df = og[columns_to_keep]

#normalizing data set
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
print(df["fbs"].isnull().sum())
df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
df["fbs"] = df["fbs"].astype(int)
df["trestbps"].fillna(df["trestbps"].median(), inplace=True)
df["chol"].fillna(df["chol"].median(), inplace=True)

#putting rest of the scores between 0 and 1
from scipy.stats import zscore
df[["age", "trestbps", "chol"]] = df[["age", "trestbps", "chol"]].apply(zscore)


print(df.head())

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=["num"])
y = df["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#checking accuracy score
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

#checking whether training data is inherently biased 
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print("Class Distribution in Training Set:")
print(dict(zip(unique, counts)))

#checking whether test data is inherently biased 
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("Class Distribution in Test Set:")
print(dict(zip(unique_test, counts_test)))

#creating confusion matrix to determine false negatives/positives
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm) 

#because false negatives high and low accuracy, trying random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Compute accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)


#trying to tune random forest model
rf_model_tuned = RandomForestClassifier(
    n_estimators=200,   # Increase number of trees
    max_depth=10,       # Limit tree depth to avoid overfitting
    min_samples_split=5,  # Require more samples to split nodes
    random_state=42
)

rf_model_tuned.fit(X_train, y_train)

y_pred_rf_tuned = rf_model_tuned.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred_rf_tuned)
print("Tuned Random Forest Accuracy:", tuned_accuracy)

# checking why cholesterol has a negative coefficient 
print("Summary of Cholesterol Column:")
print(df["chol"].describe())  # Get min, max, mean
print("Missing Values in Cholesterol:", df["chol"].isna().sum())
print("Unique Values in Cholesterol:", df["chol"].unique())

# Plot cholesterol vs. heart attack risk in og dataset
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
sns.boxplot(x=og["num"], y=og["chol"])
plt.xlabel("Heart Attack")
plt.ylabel("Cholesterol Levels")
plt.title("Cholesterol Distribution by Heart Attack Risk")
plt.show()


