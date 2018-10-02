import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import math
import numpy as np
from sklearn.externals import joblib
import json
import inflect
from autocorrect import spell





tf_idf = joblib.load('tf_dict.sav')
file_dict = json.load(open('file_dict','r'))
print(len(file_dict.keys()))
file_list=joblib.load('file_list.sav')
# print(query)
# refined_query = []


def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def preprocess(query):
    refined_query=[]
    digit_engine = inflect.engine()
    temp = query.split(" ")
    for i in range(0,len(temp)):
        temp[i]=spell(temp[i])
    query = " ".join(temp)
    table = str.maketrans("", "", string.punctuation)
    token_maker = nltk.RegexpTokenizer(r'\w+')
    stopword = set(stopwords.words('english'))
    query = query.translate(table)
    token = token_maker.tokenize(query)
    stemmer = PorterStemmer()
    for i in token:
        if i not in stopword:
            if (IsFloat(i) and len(i) < 15):
                # print(i)
                temp = digit_engine.number_to_words(i)
                temp = stemmer.stem(temp)
                refined_query.append(temp)
            else:
                i = stemmer.stem(i)
                refined_query.append(i)

    return refined_query

def tf_idf_score(tf_idf_matrix,file_list,query,n):
    result = np.zeros((1, len(file_dict.keys())))
    for i in refined:
        result=result+np.array(tf_idf_matrix[i])


    idx = (-result).argsort()[0][:n]
    # print(idx)
    r_list=[]
    for i in idx:
        r_list.append(file_list[i])
    return r_list

# print(np.unique(refined,return_counts=True))

def cosine_score(tf_idf_matrix,file_list,query,n):
    query_vector = np.zeros((1,len(tf_idf_matrix.keys())))
    token,count = np.unique(query,return_counts=True)
    key_list = list(tf_idf_matrix.keys())

    for i in range(0,len(token)):
        try:
            index = list(key_list).index(token[i])
            query_vector[0][index]=count[i]
        except ValueError:
            continue

    r_list = []

    for i in range(0,len(file_list)):
        file_vect = [item[i] for item in tf_idf_matrix.values()]
        file_vect=np.array(file_vect)
        r_list.append(float(np.dot(query_vector[0].T,file_vect))/(np.linalg.norm(query_vector[0])*np.linalg.norm(file_vect)))

    r_list = np.array(r_list)
    # print(np.max(r_list))
    idx = (-r_list).argsort()[:n]

    # print(idx)
    result = []
    for i in idx:
        result.append(file_list[i])
    return result



results = {}
# def print_and_save()
while(True):

    if (os.path.exists('results.sav')):
        results = joblib.load('results.sav')
        # print(1)
    # print(results)
    query = input("Enter Query : ")
    query=query.lower()
    refined = preprocess(query)
    if(len(refined)==0):
        print("No Result Found")
        continue
    else:
        temp = " ".join(refined)
        # print(temp)
        try :
            temp_result=results[temp]
            # print(1)
            print("TF-IDF : ",temp_result[0])
            print("COSINE : ",temp_result[1])
            results.pop(temp)
            results[temp]=temp_result
            joblib.dump(results,'results.sav')
        except KeyError:
            tf_result = tf_idf_score(tf_idf, file_list, refined, 5)
            cosine_result = cosine_score(tf_idf, file_list, refined, 5)

            if(len(results.keys())<20):
                results[temp] = [tf_result, cosine_result]
                print("TF-IDF : ", tf_result)
                print("COSINE : ", cosine_result)
                joblib.dump(results,'results.sav')
            else:
                results.popitem(last=False)
                results[temp] = [tf_result, cosine_result]
                print("TF-IDF : ", tf_result)
                print("COSINE : ", cosine_result)
                # print(results[temp])
                joblib.dump(results,'results.sav')



