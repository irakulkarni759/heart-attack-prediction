#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 21:41:06 2025

@author: irakulkarni
"""

import pandas as pd
og = pd.read_csv("heart_disease_uci.csv")  


#cleaning data set
columns_to_keep = ["age", "trestbps", "fbs", "num"]
df = og[columns_to_keep]

#normalizing data set
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
print(df["fbs"].isnull().sum())
df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
df["fbs"] = df["fbs"].astype(int)
df["trestbps"].fillna(df["trestbps"].median(), inplace=True)


#putting rest of the scores between 0 and 1
from scipy.stats import zscore
df[["age", "trestbps"]] = df[["age", "trestbps"]].apply(zscore)


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

import numpy as np

#checking whether test data is inherently biased 
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("Class Distribution in Test Set:")
print(dict(zip(unique_test, counts_test)))

#creating confusion matrix to determine false negatives/positives
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm) 

y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probability scores
y_pred_adjusted = (y_pred_prob > 0.4).astype(int)  # Lower threshold from 0.5 to 0.4

# Compute new confusion matrix
from sklearn.metrics import confusion_matrix
cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print("Confusion Matrix after Threshold Adjustment:\n", cm_adjusted)


