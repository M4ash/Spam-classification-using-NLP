

# -*- coding: utf-8 -*-
"""
    Group Project: Spam Classification using NLP
    Prepared by:
    Billah Syed Mashkur 1723387
    KM Zubair 1722931
    Tasnim Rafia 1725826
"""

import pandas as pd
import os
os.chdir("/Users/mashkur/Downloads")

dataset = pd.read_csv('Dataset - SPAM text message.csv')

print(dataset)

#data cleaning...
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(dataset['label'])
y=y.iloc[:,1].values #remove first col (ham)

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)



#result analysis
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ax= plt.subplot()
sns.heatmap(confusion, annot=True, ax=ax)

#plotting confusion matrix with labels and colors
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Ham', 'Spam']); ax.yaxis.set_ticklabels(['Ham', 'Spam']);

ax= plt.subplot()
sns.heatmap(confusion/np.sum(confusion), annot=True, fmt='.2%',ax = ax)

#plotting confusion matrix as percentages with labels and colors
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Ham', 'Spam']); ax.yaxis.set_ticklabels(['Ham', 'Spam']);

#plotting a 2D graph between y_test and y_pred
x1 = range(0, y_test.size)
y1 = y_test

x2 = range(0, y_pred.size)
y2 = y_pred

plt.plot(x2, y_pred, label='Predicted')
plt.plot(x1, y_test, label='Actual')

plt.legend()
plt.show()


#scatterplotting
_, ax = plt.subplots()

ax.scatter(x = range(0, y_test.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
ax.scatter(x = range(0, y_pred.size), y=y_pred, c = 'red', label = 'Predicted', alpha = 0.3)

plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('')
plt.legend()
plt.show()





#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)

# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)

# f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

print("Accuracy: ",accuracy*100,"%")
print("Precision: ",precision)
print("Recall: ",recall)
print("F1 Score: ",f1)