
# coding: utf-8

# In[10]:

# Naive Bayes Classifier Amazon Baby Review

import pandas as pd
import numpy as np
import nltk
import string
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
# In[11]:

#reading reviews using pandas library from amazon_baby_train.csv file
reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x > 3 else 0)

# printing the mean and standard deviation of ratings
print("The Mean of the Review Attribute is : ")
print(scores.mean())
print("The Standard Deviation of the Review Attribute is : ")
print(scores.std())


# In[12]:

#grouping the reviews into positive and negative
reviews.groupby('rating')['review'].count()


# In[13]:

# plotting a graph which counts the number of positive and negative labels
#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[14]:

# splitting the positive and negative review and storing them in separate arrays
def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating'] == 0]
    pos = reviews.loc[Summaries['rating'] == 1]
    return [pos,neg]    
print("Split is being done into positive and negative");

# In[15]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[16]:

# Preprocessing steps

# Using lemmatizer to lemmatizze words
lemmatizer = nltk.WordNetLemmatizer()

# using stop words to remove the words which do not contribute to the sentiment
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    #print(line)
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[17]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[18]:

# combining the positive and negative reviews
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[19]:

#tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[20]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[21]:

# The most common 5000 words
topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))


# In[22]:

#printing the top 200 most common words
word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])
#print(word_his)


# In[23]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[24]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[25]:

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[26]:

lencheck = tr_features.shape


# In[27]:

# Using Naive Bayes classifier to classify the data
#clf =  GaussianNB(priors=None)
#clf = BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=False,class_prior=None)
clf = MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)
tr_features = tr_features.toarray()
clf = clf.fit(tr_features, labels)
num_correct = 0;
newlen = lencheck[0]-1


for ch in range(0,newlen):
    checkPrediction = clf.predict(tr_features[ch])
    if(checkPrediction == [labels[ch]]):
        num_correct = num_correct+1;
print("Number of Correct")
print(num_correct)
acc = num_correct/newlen
print("Accuracy is ")
print(acc*100);
#predicting the output
#tfPredication = clf.predict(tr_features)
#tfAccuracy = metrics.accuracy_score(tfPredication,labels)
#print(tfAccuracy * 100)


# In[ ]:

# printing the metrics
#print(metrics.classification_report(labels, tfPredication))


# In[ ]:

## Testing Dataset

# Reading reviews using pandas library from amazon_baby_test.csv file
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x > 3 else 0)
#print(reviews.head(25))

# calculating the mean of reviews
scores.mean()


# In[ ]:

# Grouping the reviews into positive and negative
reviews.groupby('rating')['review'].count()


# In[ ]:

# plotting a graph which counts the number of positive and negative labels
#reviews.groupby('rating')['review'].count().plot(kind='bar', color= ['r','g'],title='Label Distribution',  figsize = (10,6))


# In[ ]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[ ]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[ ]:

# combining the positive and negative reviews
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[ ]:

# Tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[ ]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[ ]:

# The most common 5002 words
topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))


# In[ ]:

#printing the top 200 most common words
word_his = pd.DataFrame(word_features.most_common(200), columns = ['words','count'])


# In[ ]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[ ]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[ ]:

# Transforming the features using Tfidf transformer
cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[ ]:

#predicting the output
#tePredication = clf.predict(te_features)
#teAccuracy = metrics.accuracy_score(tePredication,labels)
#print(teAccuracy)


# In[ ]:

lencheck = te_features.shape

num_correct = 0;
newlen = lencheck[0]-1


for ch in range(0,newlen):
    checkPrediction = clf.predict(tr_features[ch])
    if(checkPrediction == [labels[ch]]):
        num_correct = num_correct+1;
print("Number of Correct")
print(num_correct)
acc = num_correct/newlen
print("Accuracy is ")
print(acc*100);

# In[ ]:

# printing the metrics
#print(metrics.classification_report(labels, tePredication))


