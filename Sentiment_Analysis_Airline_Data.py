#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Airline Data

# A classification model was developed using a pre-labeled dataset of positive, negative, and neutral tweets
# given by customers who flew with either American, Delta, Southwest, US Airways, United or Virgin America.
# The data is an open data set obtained from the Figure Eight platform.

import re
from bs4 import BeautifulSoup
# text processing modules
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import sys
# scipy # for statistics
import scipy
# numpy for array, matrix and vector calculations
import numpy as np
# matplotlib for graphs
import pandas as pd
# scikit-learn for machine learning
import sklearn
# Load  specialised libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# model selection
from sklearn import model_selection
# kpi: evaulating the performance of the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score
import timeit
start = timeit.default_timer()
stop = timeit.default_timer() 

# # Load and Review Data
# Check dimensions and shape.  Compute a statistical summary - count, mean, min, max and percentiles.

airline_tweet = pd.read_csv('C:/Users/lou_/Documents/Python/NLP/airline_tweets/Airline-Sentiment-2.csv')

# # Clean Data
# Address missing values and duplicates. Replace missing values or characters with NaN.
# Use the pd is null function to get count of missing values per variable.  

# Count the number of missing values per variable using the pandas isnull() function
# filter on missing data variables
missings = airline_tweet.isnull().sum()[airline_tweet.isnull().sum()!=0]
print('Missing value per variable [5] : \n', missings) 

# Some missing values, but for this model these columns of data are not needed.
# # Select Columns from DF

airline_tweet2 = pd.DataFrame(airline_tweet, columns = [ '_unit_id','airline_sentiment', 'text', 'airline'])
airline_tweet2.head(5)

# Text is cleaned by removal of stopwords, changing text to lower case, removal of bad characters, removal of
# punctuation and HTML decoding,

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):

    """

        text: a string

        return: modified initial string

    """

    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    return text

airline_tweet2['text'] = airline_tweet2['text'].apply(clean_text)

print(airline_tweet2.head(10))

airline_tweet2.info()

# #  Bar Chart - Distribution of Sentiments
# A bar chart is used to investigate the distribution of sentiments.

Sentiment_count=airline_tweet2.groupby('airline_sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['text'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Reviews')
plt.show()

# # Evaluate Models with use of Tf-IDF

X = airline_tweet2.text
y = airline_tweet2.airline_sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# Multinomial NB

senti = ['neutral', 'positive', 'negative']
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

param = {'vect__ngram_range': [(1,1), (1,2), (1,3)],
          'tfidf__use_idf': [True, False],
          'clf__alpha': np.arange(0.1, 0.8, 0.1)
}


grid_clf = GridSearchCV(nb, param, n_jobs = 1, cv=7)
_ = grid_clf.fit(X_train, y_train)
grid_clf.best_params_



multinomial_nb = Pipeline([('vect', CountVectorizer(stop_words=None, ngram_range=(1,2))),
                 ('tfidf', TfidfTransformer(use_idf = True)),
                 ('clf', MultinomialNB(alpha=0.1))])
multinomial_nb.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = multinomial_nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=senti))
print('Time: ', stop - start) 


# Linear Support Vector Machine

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier()),

               ])

parameters = {'vect__ngram_range': [(1,1), (1,2), (1,3)],
              'tfidf__use_idf': [True, False],
              'clf__alpha': np.arange(1e-4, 1e0, 0.2), #1e1, 1e2, 1e3]
              'clf__max_iter': [100], # number of epochs/cycles
              'clf__loss': ['hinge'],
              'clf__penalty': ['l2'],
              'clf__random_state': [42], 
              'clf__tol': [1e-3]
}

grid_clf = GridSearchCV(sgd, parameters, n_jobs = 1, cv=7)
_ = grid_clf.fit(X_train, y_train)
grid_clf.best_params_

sgd = Pipeline([('vect', CountVectorizer(stop_words=None, ngram_range=(1,2))),
                 ('tfidf', TfidfTransformer(use_idf = True)),
                 ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=100, tol=1e-3))])

sgd.fit(X_train, y_train)
                
from sklearn.metrics import classification_report

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=senti))
print('Time: ', stop - start) 

# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LogisticRegression())

               ])

parameters = {'vect__ngram_range': [(1,1), (1,2), (1,3)],
              'tfidf__use_idf': [True, False],
              'clf__n_jobs':[1],
              'clf__C': np.arange(100, 1000, 100)
}

grid_clf = GridSearchCV(logreg, parameters, n_jobs = 1, cv=7)
_ = grid_clf.fit(X_train, y_train)
grid_clf.best_params_

logreg = Pipeline([('vect', CountVectorizer(stop_words=None, ngram_range=(1,3))),
                 ('tfidf', TfidfTransformer(use_idf = True)),
                 ('clf', LogisticRegression(n_jobs=1, C=800)),])

logreg.fit(X_train, y_train)
                
from sklearn.metrics import classification_report

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred,target_names=senti))
print('Time: ', stop - start) 
