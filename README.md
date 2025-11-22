# Student-Performance-Prediction-Using-Logistic-Regression

Introduction

1. Project Overview

This project aims to analyze and predict student academic performance using machine learning techniques, specifically Logistic Regression. By examining various factors that influence student success, we can identify at-risk students early and provide targeted interventions to improve educational outcomes.

2. Objective

Analyze factors affecting student exam scores
Build a predictive model to classify students into "Good" (≥70) and "Not Good" (<70) performance categories
Identify the most influential features contributing to student success
Provide actionable insights for educators and policymakers

3. Dataset Description

The dataset contains 6,607 student records with 20 variables capturing various aspects of student life and academic environment.

Numerical Variables (7 features):

Hours_Studied: Weekly study hours
Attendance: Class attendance percentage
Sleep_Hours: Average daily sleep hours
Previous_Scores: Scores from previous assessments
Tutoring_Sessions: Number of tutoring sessions attended
Physical_Activity: Hours of physical activity per week
Exam_Score: Final exam score (Target variable)
Categorical Variables (13 features):

Parental_Involvement: Level of parental engagement (Low/Medium/High)
Access_to_Resources: Availability of learning resources (Low/Medium/High)
Extracurricular_Activities: Participation in extracurricular activities (Yes/No)
Motivation_Level: Student's motivation level (Low/Medium/High)
Internet_Access: Access to internet (Yes/No)
Family_Income: Family income level (Low/Medium/High)
Teacher_Quality: Quality of teachers (Low/Medium/High)
School_Type: Type of school (Public/Private)
Peer_Influence: Influence from peers (Positive/Neutral/Negative)
Learning_Disabilities: Presence of learning disabilities (Yes/No)
Parental_Education_Level: Parents' education level (High School/College/Postgraduate)
Distance_from_Home: Distance from home to school (Near/Moderate/Far)
Gender: Student gender (Male/Female)

Import Libraries
We import essential Python libraries for data manipulation, visualization, and machine learning:
pandas & numpy: Data manipulation and numerical operations
matplotlib & seaborn: Data visualization
sklearn: Machine learning algorithms and evaluation metrics

 
[1]
16s
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings("ignore")

Data Loading and Initial Exploration

 
[2]
10s
#Uppload dataset
from google.colab import filesuploaded = files.upload()

 
 
[3]
0s
#Load data from csv file into dataframe
data = pd.read_csv('StudentPerformanceFactors.csv')
 
[4]
0s
#Show fisrt 5 rows of dataframe
data.head()
 
Next steps:
 
[5]
0s
#Check informations of data
data.info()
 <class 'pandas.core.frame.DataFrame'>
RangeIndex: 6607 entries, 0 to 6606
Data columns (total 20 columns):
 #   Column                      Non-Null Count  Dtype 
---  ------                      --------------  ----- 
 0   Hours_Studied               6607 non-null   int64 
 1   Attendance                  6607 non-null   int64 
 2   Parental_Involvement        6607 non-null   object
 3   Access_to_Resources         6607 non-null   object
 4   Extracurricular_Activities  6607 non-null   object
 5   Sleep_Hours                 6607 non-null   int64 
 6   Previous_Scores             6607 non-null   int64 
 7   Motivation_Level            6607 non-null   object
 8   Internet_Access             6607 non-null   object
 9   Tutoring_Sessions           6607 non-null   int64 
 10  Family_Income               6607 non-null   object
 11  Teacher_Quality             6529 non-null   object
 12  School_Type                 6607 non-null   object
 13  Peer_Influence              6607 non-null   object
 14  Physical_Activity           6607 non-null   int64 
 15  Learning_Disabilities       6607 non-null   object
 16  Parental_Education_Level    6517 non-null   object
 17  Distance_from_Home          6540 non-null   object
 18  Gender                      6607 non-null   object
 19  Exam_Score                  6607 non-null   int64 
dtypes: int64(7), object(13)
memory usage: 1.0+ MB
 
[6]
0s
#Get the number of rows and columns in the dataframe
data.shape
 (6607, 20)
The dataset contains 6,607 rows and 20 columns with a mix of numerical (7) and categorical (13) variables.

 
[7]
0s
#Show statistical summary
data.describe(include='all')
 
The describe(include='all') function provides statistical insights into the distribution of numerical variables, including unique, top, frequency, mean, standard deviation, and quartile values.

 
[8]
0s
#Check how many values appear in Exam_Score cloumn
data['Exam_Score'].value_counts()
 
We can observe that the majority of student scores fall within the range of 68-74 points.

Exploratory Data Analysis

We identified missing values in three categorical columns. Since the missing data represents less than 1.5% of total records, we chose to impute with the mode (most frequent value) to preserve the original data distribution.

 
[9]
0s
#Check missing values
data.isnull().sum()
 
With initial inspection reveals missing values in three columns:

Teacher_Quality: 78 missing values Parental_Education_Level: 90 missing values Distance_from_Home: 67 missing values

 
[10]
0s
#Handle missing values by fill with mode
data['Teacher_Quality'].fillna(data['Teacher_Quality'].mode()[0], inplace=True)
data['Parental_Education_Level'].fillna(data['Parental_Education_Level'].mode()[0], inplace=True)
data['Distance_from_Home'].fillna(data['Distance_from_Home'].mode()[0], inplace=True)
 
[11]
0s
#Check missing values again to make sure there are no missing values
data.isnull().sum()
 
 
[12]
0s
#Check duplicated 
data.duplicated().sum()
 np.int64(0)
No duplicate records were found in the dataset, ensuring data integrity.

Data Cleaning

 
[13]
0s
#Show how many values appear in Internet_Access column
data['Internet_Access'].value_counts()
 
 
[14]
0s
#Check how many unique values which appear in Internet_Áccess column
data['Internet_Access'].nunique()
 2
 
[15]
0s
#Show how many values appear in Learning_Disabilities column
data['Learning_Disabilities'].value_counts()
 
 
[16]
0s
#Check how many unique values which appear in Learning_Disabilities column
data['Learning_Disabilities'].nunique()
 2
 
[17]
0s
#Drop 2 cloumns directly in dataframe
data.drop(['Internet_Access', 'Learning_Disabilities'], axis=1, inplace=True)
After examining the Internet_Access and Learning_Disabilities columns, we found they have only 2 unique values each with minimal variation. These columns were dropped to reduce dimensionality and potential multicollinearity issues.

 
[18]
0s
#Check first 5 rows and all columns that there are no Internet_Access and Learning_Disabilities column
data.head()
 
Next steps:
 
[19]
3s
#raws histograms for all numeric columns in the DataFrame.
data.hist(edgecolor='white', figsize=(20, 20));
 
Histograms were plotted for all numerical variables to understand their distributions. Most variables show relatively normal distributions, which is favorable for logistic regression modeling.

Numerical Variables

 
[20]
0s
#Filter numeric columns and preview the first 5 rows. 
data_num = data.select_dtypes(include=['int64', 'float64'])
data_num.head()
 
Next steps:
 
[21]
0s
#Show all index columns name  
data_num.columns
 Index(['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
       'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score'],
      dtype='object')
 
[22]
0s
#Draws heatmap to see the correlation of numerical variables
plt.figure(figsize=(12, 8))
correlation_matrix = data_num.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()
 
A correlation heatmap was generated to visualize relationships between numerical features. Key findings:

Strong positive correlations with Exam_Score:

Hours_Studied: Higher study hours correlate with better performance
Attendance: Regular attendance positively impacts scores
Previous_Scores: Past performance is a strong predictor of future success
Moderate correlations:

Tutoring_Sessions shows moderate positive correlation with exam scores
Weak correlations:

Sleep_Hours and Physical_Activity show relatively weak direct correlations with exam performance
Categorical Variables

Categorical variables were analyzed using frequency distributions to understand the composition of the student population across different categories.

 
[23]
0s
#Filter object columns and preview the first 5 rows. 
data_cat = data.select_dtypes(include=['object'])
data_cat.head()
 
Next steps:
 
[24]
0s
#Show statistic summary by transpose  
data_cat.describe().T
 
Feature Engineering

The continuous Exam_Score variable was transformed into a binary classification target:

1 (Good): Exam_Score ≥ 70
0 (Not Good): Exam_Score < 70
This threshold was chosen based on common academic standards where 70% typically represents a passing or satisfactory grade.

 
[25]
0s
#Create binary target variable then count values appear 
data["Performance_Binary"] = (data["Exam_Score"] >= 70).astype(int)
data["Performance_Binary"].value_counts()

 
 
[26]
0s
#Visualize a bar chart for target distribution (Performance_Binary)
plt.figure(figsize=(12, 8))
sns.countplot(x='Performance_Binary', hue='Performance_Binary', data=data, color=('pink'))
plt.title('Performance Distribution (Binary)')
plt.xlabel('0: Not Good (<70), 1: Good (>=70)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
 
The target variable distribution shows:

Class 0 (Not Good): ~75% of students
Class 1 (Good): ~25% of students
This indicates a moderate class imbalance that should be considered during model evaluation

Logistic Regression modelling

 
[27]
0s
#Show first 5 rows of dataframe 
data.head()
 
Next steps:
Features (X) and target (y) were separated, excluding the original Exam_Score and the newly created Performance_Binary from features.

 
[28]
0s
#Prepare features and target
X=data.drop(['Exam_Score', 'Performance_Binary'], axis=1)
y=data['Performance_Binary']
One-Hot Encoding was applied to convert categorical variables into numerical format suitable for machine learning algorithms. The drop_first=True parameter prevents multicollinearity by dropping one category per feature.

 
[29]
0s
#Encode categorical variables using One-Hot Encoding
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
The data was split into training (70%) and testing (30%) sets with stratification to maintain the original class distribution in both sets.

 
[30]
0s
#Train and test split into 70/30
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

StandardScaler was applied to normalize features, ensuring all variables contribute equally to the model regardless of their original scale.

 
[31]
0s
#Feature Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
A Logistic Regression model was trained on the scaled training data.

 
[32]
0s
#Build Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
 
 
[33]
0s
#Predictions for target
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
 
[34]
0s
#Calculate key evaluation metrics (accuracy, precision, recall, F1, and ROC-AUC) to assess the logistic regression model's performance.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
We evaluated the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Accuracy measures overall correctness, while Precision and Recall provide class-specific insights.

F1-Score offers a balanced metric for imbalanced datasets.

ROC-AUC shows the model’s ability to distinguish between 'Good' and 'Not Good' students across different probability thresholds.

 
[35]
0s
#Evaluation Metrics
pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
 
We evaluated the model using multiple metrics to get a comprehensive view of its performance:

 
[36]
0s
#Display classification report
classification_report_output = classification_report(y_test, y_pred, target_names=['Not Good (<70)', 'Good (>=70)'])
print(classification_report_output)
                 precision    recall  f1-score   support

Not Good (<70)       0.97      0.98      0.98      1495
   Good (>=70)       0.95      0.91      0.93       488

      accuracy                           0.97      1983
     macro avg       0.96      0.95      0.95      1983
  weighted avg       0.97      0.97      0.97      1983

Interpretation:

The model performs exceptionally well for the majority class (Not Good) with 98% recall
For the minority class (Good), the model achieves 91% recall, meaning 9% of good performers may be misclassified
The weighted average accounts for class imbalance and shows overall strong performance
 
[37]
0s
#Plot the confusion matrix to visualize the model’s classification performance.
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Good', 'Good'], 
            yticklabels=['Not Good', 'Good'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

 
The confusion matrix reveals:

True Negatives (TN): Students correctly predicted as "Not Good"
True Positives (TP): Students correctly predicted as "Good"
False Positives (FP): "Not Good" students incorrectly predicted as "Good"
False Negatives (FN): "Good" students incorrectly predicted as "Not Good"
The low number of false predictions indicates the model generalizes well to unseen data.

 
[38]
0s
#Displays the top 10 features ranked by their absolute coefficients from the logistic regression model
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': lr_model.coef_[0]  # For binary or first class
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n=== Top 10 Most Important Features ===")
feature_importance.head(10)
 
Next steps:
 
[39]
0s
# Plot the top 10 important features (positive = green, negative = red).
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
plt.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance (Logistic Regression)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
 
The top 10 most influential features based on logistic regression coefficients:

Positive Coefficients (increase likelihood of "Good" performance):

Attendance: Regular attendance significantly improves performance
Hours_Studied: More study hours lead to better outcomes
Previous_Scores: Historical performance is a strong predictor
Parental_Involvement_High: Highly involved parents contribute to success
Negative Coefficients (decrease likelihood of "Good" performance):

Motivation_Level_Low: Low motivation severely impacts performance
Peer_Influence_Negative: Negative peer influence harms academic success
Family_Income_Low: Financial constraints may affect educational resources
Conclusion

Summary of Findings

This analysis successfully developed a Logistic Regression model to predict student academic performance with 97% accuracy and ROC-AUC of ~0.97, demonstrating excellent ability to distinguish between "Good" and "Not Good" performing students.

Key Predictors of Student Success:

Positive factors: Attendance, Hours Studied, Previous Scores, and High Parental Involvement strongly increase the likelihood of good performance
Negative factors: Low Motivation, Negative Peer Influence, and Low Family Income significantly decrease academic success
Practical Implications

Based on our findings, educational institutions should:

Implement attendance monitoring and early intervention programs
Encourage parental engagement in student learning
Provide tutoring support and motivational counseling for at-risk students
Create positive peer mentorship programs
Limitations

Moderate class imbalance (75% Not Good vs 25% Good) may slightly affect minority class prediction
The binary threshold of 70 is based on common standards but may vary across educational contexts
Logistic Regression assumes linear relationships, which may not capture all complex patterns
Suggestions for improvement:

Apply class balancing (SMOTE or class_weight='balanced')
Try threshold tuning to improve Recall
Test non-linear models (Random Forest, XGBoost)
Hyperparameter tuning with GridSearchCV
Colab paid products - Cancel contracts here
data.drop(['Student_ID', 'Student_Name'], axis=1, inplace=True)
