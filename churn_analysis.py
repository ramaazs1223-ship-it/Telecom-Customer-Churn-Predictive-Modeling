import os
os.system("pip install kagglehub") 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
 
print("Path to dataset files:", path)
 
csv_file = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
 
df = pd.read_csv(csv_file)
 
print(df.info())
 
print(df.isnull().sum())
 
df.drop('customerID', axis=1, inplace=True)
 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
 
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
 
label_encoder = LabelEncoder()
 
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])
 
X = df.drop('Churn', axis=1)
y = df['Churn']
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
 
model = LogisticRegression(max_iter=1000)
 
model.fit(X_train, y_train)
 
predictions = model.predict(X_test)
 
accuracy = accuracy_score(y_test, predictions)
 
print("Accuracy Score:")
print(accuracy)
 
print("Classification Report:")
print(classification_report(y_test, predictions))
 
cm = confusion_matrix(y_test, predictions)
 
plt.figure(figsize=(6,4))
 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
 
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
 
plt.show()
 
plt.figure(figsize=(6,4))
 
sns.countplot(x='Churn', data=df)
 
plt.title("Customer Churn Distribution")
 
plt.show()
 
plt.figure(figsize=(8,5))
 
sns.histplot(df['tenure'], bins=30, kde=True)
 
plt.title("Customer Tenure Distribution")
 
plt.xlabel("Tenure")
 
plt.show()
 
plt.figure(figsize=(8,5))
 
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
 
plt.title("Monthly Charges vs Churn")
 
plt.show()
 
print("Project Completed Successfully!")
 
print(f"Model Accuracy: {accuracy:.2f}")
