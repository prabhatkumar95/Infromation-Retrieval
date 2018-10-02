import os
import string
from nltk.stem.snowball import PorterStemmer


def queryOR(x, y):
    c = x + y
    c = set(c)
    return list(c)

def queryORMerge(x,y):
    left = 0
    right = 0
    result = []
    loopcount = 0
    while (left < len(x) and right < len(y)):
        loopcount = loopcount + 1
        if (x[left] < y[right]):
            result.append(x[left])
            left = left + 1

        elif (x[left] > y[right]):
            result.append(y[right])
            right = right + 1

        else:
            result.append(x[left])
            left = left + 1
            right = right + 1
    while(left<len(x)):
        result.append(x[left])
        left=left+1
    while(right<len(y)):
        result.append((y[right]))
        right=right+1
    return result, loopcount


import math


def queryAND_Skip(x, y, skip_fact):
    left = 0
    right = 0
    lengthx = len(x)
    lengthy = len(y)
    loopcount=0
    skipx = math.floor(skip_fact *math.sqrt(lengthx))
    if skipx<=1:
        skipx=2
    skipy = math.floor(skip_fact *(math.sqrt(lengthy)))
    if skipy<=1:
        skipy=2
    result = []
    while (left < lengthx and right < lengthy):
        loopcount=loopcount+1
        if (x[left] < y[right]):
            if(left%skipx==0 and (left + skipx) < lengthx and x[left + skipx] < y[right]):
                left = left + skipx
            else:
                left = left + 1
        elif (x[left] > y[right]):
            if(right%skipy==0 and (right + skipy) < lengthy and x[left] > y[right + skipy]):
                right = right + skipy
            else:
                right = right + 1
        else:
            result.append(x[left])
            left = left + 1
            right = right + 1
    return result,loopcount


def queryAND(x, y):
    left = 0
    right = 0
    result = []
    # print (len(x))
    # print (len(y))
    loopcount=0
    while (left < len(x) and right < len(y)):
        loopcount=loopcount+1
        if (x[left] < y[right]):
            left = left + 1
        elif (x[left] > y[right]):
            right = right + 1
        else:
            result.append(x[left])
            left = left + 1
            right = right + 1
    return result,loopcount


def queryNOT(x, u):
    temp = x
    result = []
    temp = set(temp)

    for i in u:
        if i not in temp:
            result.append(i)

    return result


from sklearn.externals import joblib
import json

worddict = json.load(open('json_dict', 'r'))
file_list = joblib.load('file_list.sav')
word_1 = input()
word_2 = input()

table = str.maketrans("", "", string.punctuation)
word_1 = word_1.translate(table)
word_2 = word_2.translate(table)
stemmer = PorterStemmer()
word_1 = stemmer.stem(word_1)
word_2 = stemmer.stem(word_2)

l1 = worddict.get(word_1, [0, []])
l2 = worddict.get(word_2, [0, []])

print(len(l1[1]))
print(len(l2[1]))
print("OR : ", len(queryOR(l1[1], l2[1])))
print("OR : ", len(queryORMerge(l1[1], l2[1])[0]))

a=queryAND(l1[1],l2[1])
print("AND : ", len(a[0])," ",a[1])
b=queryAND_Skip(l1[1],l2[1],1)
print(len(b[0])," ",b[1])




#
import timeit
skipfactor=[0.1,0.3,0.5,0.7,1.0,1.5,3]
exectime = [0.0]*len(skipfactor)
c_loop = [0]*len(skipfactor)
for j in range(0,10):
    start_time = timeit.default_timer()

    for i in range(0,len(skipfactor)):
        start_time = timeit.default_timer()
        c=queryAND_Skip(l1[1],l2[1],skip_fact=skipfactor[i])[1]
        exectime[i]+=exectime[i]+(timeit.default_timer() - start_time)*1000000
        c_loop[i]=c_loop[i]+c
for j in range(0,len(exectime)):
    exectime[j]=exectime[j]/10
    c_loop[j]=c_loop[j]/10
print (exectime)
print (c_loop)
#
#
#
import matplotlib.pyplot as plt
plt.plot(skipfactor, c_loop)
plt.xlabel("Skip Factor")
plt.ylabel("Iterations")
plt.show()
