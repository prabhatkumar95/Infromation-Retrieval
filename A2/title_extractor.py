import os
import math
import string
import json
import nltk
from nltk.corpus import stopwords
from sklearn.externals import joblib
import inflect

def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def preprocess(query):
    refined_query=[]
    table = str.maketrans("", "", string.punctuation)
    token_maker = nltk.RegexpTokenizer(r'\w+')
    stopword = set(stopwords.words('english'))
    query = query.translate(table)
    digit_engine = inflect.engine()
    token = token_maker.tokenize(query)
    stemmer = nltk.PorterStemmer()
    for i in token:
        if i not in stopword:
            if(IsFloat(i)):
                temp = digit_engine.number_to_words(i)
                temp=stemmer.stem(temp)
                refined_query.append(temp)
            else:
                temp = stemmer.stem(i)
                refined_query.append(temp)
    return refined_query


f = open('index.html','r')
title_lst={}
for i in f.readlines():
    temp = i.split('> ')[1]
    word_count = int(temp.split(" ")[1])
    file_name = temp.split(" ")[0]
    temp = temp.split(" ")[2:]
    temp = " ".join(temp)
    temp = preprocess(temp)
    title_lst[file_name]=[word_count,temp]

json.dump(title_lst,open("title_extract",'w'))


