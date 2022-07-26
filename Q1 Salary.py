################# problem1 ###########################
#1.) Prepare a classification model using Naive Bayes for Salary dataset, train and test datasets are given separately use both datasets for model building. 	

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
salary_test = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\SalaryData_Test.csv",encoding = "ISO-8859-1")
salary_train = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\SalaryData_Train.csv",encoding = "ISO-8859-1")

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
x = {" <=50K" :1," >50K" :2}
salary_test.Salary = [x[item] for item in salary_test.Salary]
salary_train.Salary = [x[item] for item in salary_train.Salary]

#dummy variables for test data
dummies_salary_test = pd.get_dummies(salary_test)
dummies_salary_test.drop(['Salary'],axis = 1,inplace =True)
dummies_salary_test.isna()

#dummy variables for train data
dummies_salary_train = pd.get_dummies(salary_train)
dummies_salary_train.drop(['Salary'],axis = 1,inplace =True)
dummies_salary_train.isna()

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(dummies_salary_train, salary_train.Salary)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(dummies_salary_test)
accuracy_test_m = np.mean(test_pred_m == salary_test.Salary)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, salary_test.Salary) 

pd.crosstab(test_pred_m, salary_test.Salary)

# Training Data accuracy
train_pred_m = classifier_mb.predict(dummies_salary_train)
accuracy_train_m = np.mean(train_pred_m == salary_train.Salary)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(dummies_salary_train, salary_train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(dummies_salary_test)
accuracy_test_lap = np.mean(test_pred_lap == salary_test.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, salary_test.Salary) 

pd.crosstab(test_pred_lap, salary_test.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(dummies_salary_train)
accuracy_train_lap = np.mean(train_pred_lap == salary_train.Salary)
accuracy_train_lap
