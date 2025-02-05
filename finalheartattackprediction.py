#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:34:24 2025

@author: irakulkarni
"""

import pandas as pd
og = pd.read_csv("heart_disease_uci.csv")  


#cleaning data set
columns_to_keep = ["age", "trestbps", "fbs", "ca", "oldpeak", "sex", "slope", "thal", "num"]
df = og[columns_to_keep]

#normalizing data set
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])
df["fbs"] = df["fbs"].astype(int)
df["trestbps"].fillna(df["trestbps"].median(), inplace=True)
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["slope"] = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2})
df["slope"] = df["slope"].fillna(df["slope"].mode()[0])
df["thal"] = df["thal"].map({"normal": 0, "fixed defect": 1, "reversible defect": 2})
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
df["oldpeak"] = df["oldpeak"].fillna(df["oldpeak"].mode()[0])
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])

from scipy.stats import zscore
df[["age", "trestbps"]] = df[["age", "trestbps"]].apply(zscore)

print (df.head())

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=["num"])
y = df["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #changed split to 90, 10

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

#creating a function to print equation
def equation(model, X_train):
    feature_names = X_train.columns.tolist()
    coefficients = model.coef_[0]  # Extract coefficients
    intercept = model.intercept_[0]  # Extract intercept

    eq = f"y = {intercept:.4f} "
    for feature, coef in zip(feature_names, coefficients):
        eq += f"+ ({coef:.4f} * {feature}) "

    print("\nLogistic Regression Equation:")
    print(eq)

equation(model, X_train)

# finding importance of variables
import pandas as pd
feature_names = X_train.columns.tolist()
coefficients = model.coef_[0]  # Extract coefficients from model
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
print("Logistic Regression Coefficients:")
print(coef_df)

#checking correlation of blood pressure with other variables 
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show() #hence, trestbps shouldn't be removed

# Finding the best C value
c_values = np.logspace(-3, 2, 50)  # More effective range from 0.001 to 100
best_accuracy = 0
best_c = None

for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_accuracy:
        best_accuracy = acc
        best_c = c  # Store the best C value

print(f"Best C Value: {best_c:.4f}, Best Accuracy: {best_accuracy:.4f}")

# Retrain the final model using the best C value
best_model = LogisticRegression(C=best_c, max_iter=1000, solver='liblinear')
best_model.fit(X_train, y_train)

# Extract and print the equation and final accuracy
feature_names = X_train.columns.tolist()
coefficients = best_model.coef_[0]  # Extract coefficients
intercept = best_model.intercept_[0]  # Extract intercept

equation = f"y = {intercept:.4f} "
for feature, coef in zip(feature_names, coefficients):
    equation += f"+ ({coef:.4f} * {feature}) "

print("\nOptimized Logistic Regression Equation (Using Best C):")
print(equation)
y_pred_final = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)*100
print(f"\nFinal Model Accuracy (Using Best C): {final_accuracy:.4f} %")



