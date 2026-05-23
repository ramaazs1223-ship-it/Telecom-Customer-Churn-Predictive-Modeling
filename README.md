Telecom Customer Churn: Predictive Modeling and Analytics
Project Overview
This project addresses a critical business challenge in the telecommunications sector: predicting customer churn. Using the Telco Customer Churn dataset, this data science workflow builds a machine learning pipeline to analyze historical customer profiles, understand underlying churn behavior, and deploy a predictive framework.
Analytical and Modeling Pipeline (Python)
The project is built entirely in Python and executes an end-to-end predictive analytics workflow:
Data Acquisition: Automated ingestion of customer profiles from Kaggle using the Kagglehub API.
Data Preprocessing and Engineering:
Eliminating irrelevant unique identifiers like customerID.
Forcing numeric type conversion on TotalCharges and dynamically filling missing values using mean imputation.
Encoding categorical text columns into numerical data streams using Scikit-Learn LabelEncoder.
Machine Learning Training: Partitioning the dataset into an 80/20 train-test split and fitting a Logistic Regression model optimized over 1,000 iterations.
Evaluation and Metrics: Extracting performance dimensions including overall accuracy scores, detailed classification reports (precision, recall, f1-score), and standard confusion matrices.
Exploratory Data Analysis (EDA): Building custom visualizations via Matplotlib and Seaborn to map churn distributions, customer tenure patterns, and correlation markers against monthly billing rates.
Key Insights Tracked
Dynamic prediction models determining probability matrices of customer churn.
Quantitative tracking of tenure impact on customer retention strategy.
Visual correlation evaluating pricing pressures against attrition risk.
Tools and Libraries Used
Python 3
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-Learn (LogisticRegression, LabelEncoder, train_test_split)
Visualizations: Seaborn, Matplotlib
This project was completed as part of the Advanced Data Mining curriculum at the University of Petra
