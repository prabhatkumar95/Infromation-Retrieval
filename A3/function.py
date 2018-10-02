import os

import inflect
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def preprocess(path):
    file = open(path, encoding='latin')
    d = file.read().lower().split("\n\n")
    d = d[1:]
    d = ''.join(d)
    table = str.maketrans("", "", string.punctuation)
    d = d.translate(table)
    token_m = nltk.RegexpTokenizer(r'\w+')
    t = token_m.tokenize(d)
    stop = set(stopwords.words('english'))
    token_s = []

    for j in t:
        if (j not in stop):
            token_s.append(j)
    t = []
    stemm = PorterStemmer()
    for j in token_s:
        temp = stemm.stem(j)
        t.append(temp)
    return t


def accuracy(X,Y):
    count=0
    for i in range(len(X)):
        if(X[i]==Y[i]):
            count=count+1
    return count/float(len(X))

def tf_preprocess(X):


    for i in X:
        file = open(i, encoding='latin')
        d = file.read().lower().split("\n\n")
        d = d[1:]
        d = ''.join(d)
        table = str.maketrans("", "", string.punctuation)
        d = d.translate(table)
        token_m = nltk.RegexpTokenizer(r'\w+')
        t = token_m.tokenize(d)
        stop = set(stopwords.words('english'))
        token_s = []

        for j in t:
            if (j not in stop):
                token_s.append(j)
        t = []
        stemm = PorterStemmer()
        for j in token_s:
            temp = stemm.stem(j)
            t.append(temp)




