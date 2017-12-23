
# coding: utf-8

# In[3]:

# Decision Tree Amazon Baby Review

import pandas as pd
import numpy as np
import nltk
import string
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


#reading reviews using pandas library from amazon_baby_train.csv file
reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')

# printing the mean and standard deviation of ratings
print(scores.mean())
print(scores.std())


# In[5]:

#grouping the reviews into positive and negative
reviews.groupby('rating')['review'].count()


# In[6]:

# plotting a graph which counts the number of positive and negative labels
reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[7]:

# splitting the positive and negative review and storing them in separate arrays
def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating']== 'neg']
    pos = reviews.loc[Summaries['rating']== 'pos']
    return [pos,neg]


# In[8]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[9]:

# Using lemmatizer to lemmatizze words
lemmatizer = nltk.WordNetLemmatizer()

# using stop words to remove the words which do not contribute to the sentiment
stops = stopwords.words('english')
stops.remove('not')
stops.remove('no')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())   
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[10]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[11]:

# combining the positive and negative reviews
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[12]:

# Splitting the dataset into training set and validation set
[Data_train,Data_test,Train_labels,Test_labels] = train_test_split(data,labels , test_size=0.25, random_state=20160121,stratify=labels)


# In[13]:

#tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[14]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[15]:

# The most common 5000 words
topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))


# In[16]:

#printing the top 200 most common words
word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])


# In[17]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])
print(c_fit)


# In[18]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[19]:

# Transforming the features using Tfidf transformer
ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[20]:

tr_features.shape


# In[58]:

#cte_features = vec.transform(Data_test)
#te_features = tf_vec.transform(cte_features)


# In[ ]:

# Using decision tree classifier to classify the data
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100, learning_rate=0.1)
clf = clf.fit(tr_features, labels)

#predicting the output
tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication,labels)
print("Accuracy on the Training dataset:")
print(tfAccuracy*100)


# In[60]:

# printing the metrics
print(metrics.classification_report(labels, tfPredication))


# In[61]:

#Testing set


# In[62]:

#reading reviews using pandas library from amazon_baby_test.csv file
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')

# calculating the mean of reviews
scores.mean()


# In[63]:

#grouping the reviews into positive and negative
reviews.groupby('rating')['review'].count()


# In[64]:

# plotting a graph which counts the number of positive and negative labels
reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[65]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[66]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[67]:

# combining the positive and negative reviews
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[68]:

#tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[69]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[70]:

# The most common 5002 words
topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[71]:

#printing the top 200 most common words
word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])


# In[72]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[73]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[74]:

# Transforming the features using Tfidf transformer
cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[75]:

te_features.shape


# In[76]:

#predicting the output
tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication,labels)
print(teAccuracy)


# In[77]:

# printing the metrics
print(metrics.classification_report(labels, tePredication))

