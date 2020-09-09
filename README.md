# Credit-Card-Fraud-Detection

# Data Source: https://www.kaggle.com/dalpozz/creditcardfraud/data

It is a CSV file, contains 31 features, the last feature is used to classify the transaction whether it is a fraud            or not.


Information about data set:

    The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
    It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues original features are not provided and more background information about the data is also not present. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
    Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
    The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

# Flow of Project

We have done Exploratory Data Analysis on full data then we have removed outliers using "LocalOutlierFactor", then finally we have used KNN technique to predict to train the data and to predict whether the transaction is Fraud or not. We have also applied T-SNE to visualize the Fraud and genuine transactions in 2-D.

# How to Run the Project

Download the data set and put it in the same folder as the files on the repository and run the files.

# Prerequisites
The minimum requirements for the project are:

Python: 2.7.13
Numpy:  1.14.0
Matplotlib: 2.1.0
Seaborn: 0.8.1
Scipy:  1.0.0
Sklearn: 0.19.1

# Authors
Vuyo Nkadimeng - Complete Work

# Acknowledgments
Applied AI Course
