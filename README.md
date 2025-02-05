# heart-attack-prediction
A simple machine learning model to predict heart attack risk using UCI Heart Disease dataset.

**Building a Heart Attack Prediction Model: An Iterative Approach**

When I first started working with the UCI Heart Disease dataset, I had around 900 data samples with "num" as the target variable, indicating the presence or absence of heart disease. Given the dataset's size and the binary nature of the classification problem, I decided to use Logistic Regression as my initial model. To validate its performance, I split the data into training and testing sets (80-20 split) using train_test_split from sklearn.model_selection.

I preprocessed the dataset by:

Selecting key numerical features: age, trestbps (resting blood pressure), chol (cholesterol), and fbs (fasting blood sugar).
Filling missing values using median imputation for continuous variables and mode imputation for categorical ones (fillna from pandas).
Normalizing numerical data using Z-score normalization (zscore from scipy.stats).
Encoding the target variable as binary (1 = disease, 0 = no disease).
After training the Logistic Regression model (LogisticRegression from sklearn.linear_model), I checked the accuracy (accuracy_score) and confusion matrix (confusion_matrix from sklearn.metrics). However, the model had low accuracy and a high false-negative rate, which meant it was misclassifying some patients with heart disease as healthy.

To improve the model, I tried Random Forest (RandomForestClassifier from sklearn.ensemble), assuming a non-linear model might better capture interactions. I tuned n_estimators (number of trees), max_depth, and min_samples_split to reduce overfitting, but Random Forest didn’t significantly improve accuracy. Additionally, I noticed that cholesterol had a negative coefficient, which was counterintuitive.

**Version 2: Removing Cholesterol**

To test whether cholesterol was misleading the model, I created a second version excluding "chol" as a feature and retrained the model. However, the overall accuracy remained roughly the same, indicating that cholesterol wasn’t the main issue. I also adjusted the decision threshold (from 0.5 to 0.4) using predict_proba, trying to reduce false negatives. While this had some impact, the improvements were minimal.

**Final Version: Adding More Features & Hyperparameter Tuning**

I realized that the original feature set was too limited. For the final version, I expanded the feature set to include categorical and additional clinical variables, such as:

ca (number of major coronary vessels, from fluoroscopy)
oldpeak (ST depression induced by exercise)
sex (1 = Male, 0 = Female)
slope (slope of the ST segment during exercise, mapped as upsloping = 0, flat = 1, downsloping = 2)
thal (a measure of thalassemia, mapped to normal = 0, fixed defect = 1, reversible defect = 2)
I used one-hot encoding (map from pandas) for categorical variables and mode imputation for missing values.

To prevent overfitting, I performed hyperparameter tuning on the Logistic Regression model, optimizing the C value (regularization parameter). I tested different values using logarithmic scaling (np.logspace(-3, 2, 50)) and selected the best one based on validation accuracy.

I also analyzed feature importance using model coefficients (coef_ from Logistic Regression) and created a correlation heatmap (heatmap from seaborn) to visualize feature relationships, ensuring that multicollinearity wasn’t skewing the results.

Final Skills & Tools Used
Python Libraries: pandas, numpy, scipy.stats, sklearn (scikit-learn), seaborn, matplotlib
Machine Learning Models: LogisticRegression, RandomForestClassifier
Feature Engineering: Data imputation (fillna), categorical encoding (map), normalization (zscore)
Model Evaluation: accuracy_score, confusion_matrix, predict_proba, correlation heatmap
Hyperparameter Tuning: Optimizing C for Logistic Regression using np.logspace
This iterative process helped me transform a basic model into a more reliable predictor while deepening my understanding of feature selection, model evaluation, and hyperparameter tuning.Building a Heart Attack Prediction Model: An Iterative Approach**
When I first started working with the UCI Heart Disease dataset, I had around 900 data samples with "num" as the target variable, indicating the presence or absence of heart disease. Given the dataset's size and the binary nature of the classification problem, I decided to use Logistic Regression as my initial model. To validate its performance, I split the data into training and testing sets (80-20 split) using train_test_split from sklearn.model_selection.
