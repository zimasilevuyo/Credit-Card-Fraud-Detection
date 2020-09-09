#   Group 1
import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import scipy
import sklearn


#   Checking Versions of Packages
#   Minimum Requirements:
#       Python: 2.7.13
#       Numpy:  1.14.0
#       Matplotlib: 2.1.0
#       Seaborn: 0.8.1
#       Scipy:  1.0.0
#       Sklearn: 0.19.1
print('Python: ()'.format(sys.version))
print('Numpy: ()'.format(np.__version__))
print('Pandas: ()'.format(pd.__version__))
print('Matplotlib: ()'.format(matplotlib.__version__))
print('Seaborn: ()'.format(sns.__version__))
print('Scipy: ()'.format(scipy.__version__))
print('Sklearn: ()'.format(sklearn.__version__))

import matplotlib.pyplot as plt


#   Group 2
#   Loading Data Set from csv file using pandas package
data = pd.read_csv('creditcard.csv')


#   Looking around dataset --> Group 2 to 6
#   Group 3
print(data.columns)

#   Group 4
print(data.shape)

#   Group 5
print(data.describe)

#   Group 6
data = data.sample(frac = 0.1, random_state = 1)

print(data.shape)

#   Group 7
#   Plotting histogram of each parameter
data.hist(figsize=(20, 20))
plt.show()

#   Group 8
#   Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(Valid)))

#   Group 9
#   Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#   Group 10
#   Getting all the columns from the DataFrame
columns = data.columns.tolist()

#   Filtering the columns to remove data we do not want
columns = [c for c in columns if c not in ['Class']]

#   Storing the variable we'll be predicting on
target = 'Class'

X = data[columns]
Y = data[target]

#   Print the shapes of X and Y
print(X.shape)
print(Y.shape)

#   Machine Learning Section
#   Group 11
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#   Defining a random state
state = 1

#   Defining the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination = outlier_fraction,
                                        random_state = state),
    "Local Outlier Facctor": LocalOutlierFactor(
        n_neighbors = 20,
        contamination = outlier_fraction)
}

#   Group 12
#   Fitting the Model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    # Fitting the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    #   Reshaping the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    #   Running classification metrics
    print('{}: {}'. format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))





