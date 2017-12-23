
# coding: utf-8

# In[52]:


# Naive Bayes Digit Recognition

# Importing all the required libraries
import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import math
from sklearn import metrics

# Reading the optdigits_raining.csv file as training file

with open('optdigits_raining.csv') as trainingFile:
    reader = csv.reader(trainingFile)
    
    # Declaring variables to store the data
    # X is for storing the attributes and Y is for storing the labels 
    X = []
    Y = []
    
    # Reading each line from the file and appending the data to X and Y
    for row in reader:
        X.append(row[:64])
        Y.append(row[64])


# Formatting the training data to fit the Naive Bayes classifier function
for i in range(0,len(X)):
    lst = X[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X[i] = lst
for i in range(0,len(Y)):
    Y[i] = float(int(Y[i]))

# Classifying the data using Naive Bayes Classifier 
#clf = GaussianNB(priors=None)
clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
#clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
# Using the fit function to fit the data
clf = clf.fit(X,Y)

print("######### No Validation applied #########")

# Predicting the output of training data using the Naive Bayes classifier
output_Predicted = clf.predict(X);
accuracy_training = metrics.accuracy_score(output_Predicted,Y)
print("Accuracy on the Training Data set is :")
print(accuracy_training* 100)


#### Testing data set ####

# Reading the optdigits_test.csv file as training file
with open('optdigits_test.csv') as testingFile:
    reader = csv.reader(testingFile)
    
    # Declaring variables to store the data
    # X_test is for storing the attributes and Y_test is for storing the labels 
    X_test = []
    Y_test = []
    
    # Reading each line from the file and appending the data to X and Y
    for row in reader:
        X_test.append(row[:64])
        Y_test.append(row[64])
        
# Formatting the testing data to fit the Naive Bayes classifier function        
for i in range(0,len(X_test)):
    lst = X_test[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_test[i] = lst
for j in range(0,len(Y_test)):
    Y_test[j] = float(int(Y_test[j]))

# Predicting the output of testing data using the classifier
output_predicted_testing = clf.predict(X_test)
accuracy_testing = metrics.accuracy_score(output_predicted_testing, Y_test)
print("Accuracy on the Testing Dataset is : ")
print(accuracy_testing*100)


# In[53]:

# Importing all the required libraries
import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import math
from sklearn import metrics

# Reading the optdigits_raining.csv file as training file

with open('optdigits_raining.csv') as trainingFile:
    reader = csv.reader(trainingFile)
    
    # Declaring variables to store the data
    # X is for storing the attributes and Y is for storing the labels 
    X = []
    Y = []
    
    # Reading each line from the file and appending the data to X and Y
    for row in reader:
        X.append(row[:64])
        Y.append(row[64])


# Initializing length_TrainingSet to length of input data
length_TrainingSet = len(X)

# Using 80% of the data for training and the rest for validation
percentage_training = 0.6


len_train = math.floor(length_TrainingSet * percentage_training);

# Initializing the X_train and Y_train variables with 70% of length of data
X_train = X[:len_train]
Y_train = Y[:len_train]

# Formatting the training data to fit the Naive Bayes classifier function
for i in range(0,len(X_train)):
    lst = X_train[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_train[i] = lst
for i in range(0,len(Y_train)):
    Y_train[i] = float(int(Y_train[i]))


# Initializing the X_validation and Y_validation variables with rest of the length of data
X_validation = X[len_train:len(X)]
Y_validation = Y[len_train:len(Y)]

# Formatting the validation data to fit the Naive Bayes classifier function
for i in range(0,len(X_validation)):
    lst = X_validation[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_validation[i] = lst
for i in range(0,len(Y_validation)):
    Y_validation[i] = float(int(Y_validation[i]))

# Classifying the data using Naive Bayes Classifier 
#clf = GaussianNB(priors=None)
clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
#clf = BernoulliNB(alpha=1.0, binarize=7.0, fit_prior=True, class_prior=None)
# Using the fit function to fit the data
clf = clf.fit(X_train,Y_train)

print("######### 70 - 30 Validation #########")

# Predicting the output of training data using the Naive Bayes classifier
output_Predicted = clf.predict(X_train);
accuracy_training = metrics.accuracy_score(output_Predicted,Y_train)
print("Accuracy on the Training Data set is :")
print(accuracy_training* 100)

# Predicting the output of validation data using the Naive Bayes classifier
output_predicted_validation = clf.predict(X_validation)
accuracy_validation = metrics.accuracy_score(output_predicted_validation,Y_validation)
print("Accuracy on the Validation Data set is : ")
print(accuracy_validation * 100)


#### Testing data set ####

# Reading the optdigits_test.csv file as training file
with open('optdigits_test.csv') as testingFile:
    reader = csv.reader(testingFile)
    
    # Declaring variables to store the data
    # X_test is for storing the attributes and Y_test is for storing the labels 
    X_test = []
    Y_test = []
    
    # Reading each line from the file and appending the data to X and Y
    for row in reader:
        X_test.append(row[:64])
        Y_test.append(row[64])
        
# Formatting the testing data to fit the Naive Bayes classifier function        
for i in range(0,len(X_test)):
    lst = X_test[i]
    for j in range(0,len(lst)):
        lst[j] = float(int(lst[j]))
    X_test[i] = lst
for j in range(0,len(Y_test)):
    Y_test[j] = float(int(Y_test[j]))

# Predicting the output of testing data using the classifier
output_predicted_testing = clf.predict(X_test)
accuracy_testing = metrics.accuracy_score(output_predicted_testing, Y_test)
print("Accuracy on the Testing Dataset is : ")
print(accuracy_testing*100)


# In[ ]:



