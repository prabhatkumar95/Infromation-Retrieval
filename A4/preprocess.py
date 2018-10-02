import os
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.externals import joblib
import json


def create_idf(X):
    worddict = {}
    tf_dict = {}
    file_dict = {}
    for i in range(0, len(X)):
        file_dict[X[i]] = int(i)
    for j in range(0, len(X)):

        if (j % 100 == 0):
            print(j)
        f = open(X[j], encoding='latin')
        x = f.read().lower()
        table = str.maketrans("", "", string.punctuation)
        x = x.translate(table)
        token_maker = nltk.RegexpTokenizer(r'\w+')
        token = token_maker.tokenize(x)
        stopword = set(stopwords.words('english'))
        token_stop = []
        for i in token:
            if (i not in stopword):
                token_stop.append(i)
        token = []
        stemmer = PorterStemmer()
        for i in token_stop:
            i = stemmer.stem(i)
            token.append(i)

        tokens, token_count = np.unique(token, return_counts=True)

        for i in range(0, len(tokens)):
            if worddict.get(tokens[i]):
                worddict.get(tokens[i]).append(j)
                tf_dict[tokens[i]][file_dict[X[j]]] = token_count[i]

            else:
                worddict[tokens[i]] = [j]
                tf_dict[tokens[i]] = [0] * len(X)
                tf_dict[tokens[i]][int(file_dict[X[j]])] = token_count[i]

    # print("Creating TF-IDF Vector")
    index = {}
    for key in worddict.keys():
        index[key] = len(worddict[key])
    # json.dump(index, open('json_dict', 'w'))
    # print("Phase 1")

    idf_val = {}
    for i in worddict.keys():
        idf_val[i] = np.log(float(len(X)) / int(index.get(i)))
        # temp = tf_dict[i]
        # tf_dict[i] = [float(np.log10(1+x)) * idf_val[i]  for x in temp]
    # print("Saving Files")
    # # joblib.dump(tf_dict, 'tf_dict.sav')
    # json.dump(idf_val, open('idf_val', 'w'))
    # json.dump(file_dict, open('file_dict', 'w'))
    return idf_val,tf_dict,file_dict




def get_token_count(X):
    f = open(X, encoding='latin')
    x = f.read().lower()
    table = str.maketrans("", "", string.punctuation)
    x = x.translate(table)
    token_maker = nltk.RegexpTokenizer(r'\w+')
    token = token_maker.tokenize(x)
    stopword = set(stopwords.words('english'))
    token_stop = []
    for i in token:
        if (i not in stopword):
            token_stop.append(i)
    token = []
    stemmer = PorterStemmer()
    for i in token_stop:
        i = stemmer.stem(i)
        token.append(i)

    tokens, token_count = np.unique(token, return_counts=True)
    return tokens,token_count