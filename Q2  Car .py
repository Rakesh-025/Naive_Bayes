########### problem2 #####################
#Problem Statement: -
#This dataset contains information of users in social network. This social network has several business clients which can put their ads on social network and one of the Client has a car company who has just launched a luxury SUV for ridiculous price. Build the Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV.
#Purchased: - 1 and Not Purchased: - 0

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
suv_car = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\NB_Car_Ad.csv",encoding = "ISO-8859-1")

#dropping first column for analysis
suv_car.drop(['User ID'],axis = 1 ,inplace = True)

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
suv_car [['Age','EstimatedSalary']] = scaler.fit_transform(suv_car [['Age','EstimatedSalary']])


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

suv_car_train, suv_car_test = train_test_split(suv_car, test_size = 0.2)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

#dummies for test and train data
suv_car_dumm_train = pd.get_dummies(suv_car_train)
suv_car_dumm_train.drop(["Purchased"], axis = 1, inplace = True)

suv_car_dumm_test = pd.get_dummies(suv_car_test)
suv_car_dumm_test.drop(["Purchased"], axis = 1, inplace = True)

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(suv_car_dumm_train, suv_car_train.Purchased)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(suv_car_dumm_test)
accuracy_test_m = np.mean(test_pred_m == suv_car_test.Purchased)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, suv_car_test.Purchased) 

pd.crosstab(test_pred_m, suv_car_test.Purchased)

# Training Data accuracy
train_pred_m = classifier_mb.predict(suv_car_dumm_train)
accuracy_train_m = np.mean(train_pred_m == suv_car_train.Purchased)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(suv_car_dumm_train, suv_car_train.Purchased)


# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(suv_car_dumm_test)
accuracy_test_lap = np.mean(test_pred_lap == suv_car_test.Purchased)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, suv_car_test.Purchased) 

pd.crosstab(test_pred_lap, suv_car_test.Purchased)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(suv_car_dumm_train)
accuracy_train_lap = np.mean(train_pred_lap == suv_car_train.Purchased)
accuracy_train_lap
