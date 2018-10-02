import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.externals import joblib

worddict = {}
file_name = []
for path, subdirs, files in os.walk('20_newsgroups'):
    for name in files:
        file_name.append((os.path.join(path, name)))

file_name = file_name[1:]

joblib.dump(file_name, 'file_list.sav')

for j in range(0, len(file_name)):

    f = open(file_name[j], encoding='latin')
    data = f.read().lower().split("\n\n")
    data = data[1:]
    x = ''.join(data)
    table = str.maketrans("", "", string.punctuation)
    x = x.translate(table)
    # print (len(token))
    token = word_tokenize(x)
    stopword = set(stopwords.words('english'))
    token_stop = []

    for i in token:
        if (i not in stopword):
            token_stop.append(i)

    stemmer = PorterStemmer()
    for i in token_stop:
        i = stemmer.stem(i)

    token_stop = set(token_stop)
    # print (len(token_stop))
    for i in token_stop:
        if worddict.get(i):
            worddict.get(i).append(j)
        else:
            worddict[i] = [j]

for key in worddict.keys():
    worddict.get(key).sort()

print(len(worddict.keys()))
import json

index = {}
for key in worddict.keys():
    index[key] = [len(worddict.get(key)), worddict.get(key)]
json.dump(index, open('json_dict', 'w'))