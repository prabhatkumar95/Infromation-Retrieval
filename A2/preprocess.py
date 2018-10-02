import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import math
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict
import inflect
import json

def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



worddict = {}
path = os.path.dirname(os.path.abspath('stories'))
# print (path)
file_name = [file for file in os.listdir('stories')
         if os.path.isfile(os.path.join(path,'stories', file))]
print (file_name)


title_info = json.load(open('title_extract','r'))
joblib.dump(file_name, 'file_list.sav')


file_dict={}
for i in range(0,len(file_name)):
    file_dict[file_name[i]]=int(i)
print(len(file_name))
tf_dict={}
for j in range(0, len(file_name)):

    f = open(os.path.join(path,'stories', file_name[j]), encoding='latin')
    x = f.read().lower()
    table = str.maketrans("", "", string.punctuation)
    x = x.translate(table)
    digit_engine = inflect.engine()
    # print (len(token))
    token_maker = nltk.RegexpTokenizer(r'\w+')

    token = token_maker.tokenize(x)
    stopword = set(stopwords.words('english'))
    token_stop = []

    for i in token:
        if (i not in stopword):
            token_stop.append(i)
    token=[]
    stemmer = PorterStemmer()
    for i in token_stop:
        if(IsFloat(i) and len(i)<15):
            # print(i)
            temp = digit_engine.number_to_words(i)
            temp=stemmer.stem(temp)
            token.append(temp)
        else:
            i = stemmer.stem(i)
            token.append(i)

    tokens, token_count = np.unique(token, return_counts=True)

    title_info_file = title_info[file_name[j]]

    for i in range(0,len(tokens)):
        if worddict.get(tokens[i]):
            worddict.get(tokens[i]).append(j)
            # word = tokens[i]
            # file_number = file_dict[j]
            # print (word,file_number)
            if(tokens[i] in title_info_file[1]):
                tf_dict[tokens[i]][file_dict[file_name[j]]]=token_count[i]+int(math.sqrt(title_info_file[0]))
            else:
                tf_dict[tokens[i]][file_dict[file_name[j]]] = token_count[i]

        else:
            worddict[tokens[i]] = [j]
            tf_dict[tokens[i]]=[0]*len(file_name)
            # print(tf_dict[tokens[i]])
            # print(tf_dict[tokens[i]][0])
            if(tokens[i] in title_info_file[1]):
                tf_dict[tokens[i]][int(file_dict[file_name[j]])]=token_count[i]+int(math.sqrt(title_info_file[0]))
            else:
                tf_dict[tokens[i]][int(file_dict[file_name[j]])] = token_count[i]

#
# for key in worddict.keys():
#     worddict.get(key).sort()

print(len(worddict.keys()))
print(len(tf_dict.keys()))


index = {}
for key in worddict.keys():
    index[key] = [len(worddict.get(key)), worddict.get(key)]
json.dump(index, open('json_dict', 'w'))



idf_val={}
for i in worddict.keys():
    idf_val[i]=math.log10(float(len(file_name))/int(index[i][0]))
    temp=tf_dict[i]
    tf_dict[i]=[float(1+math.log10(x))*idf_val[i] if x>0 else 0 for x in temp]
joblib.dump(tf_dict,'tf_dict.sav')
json.dump(idf_val,open('idf_val','w'))
json.dump(file_dict,open('file_dict','w'))