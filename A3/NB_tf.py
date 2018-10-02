from function import preprocess
import numpy as np
import math


class NB_tf():
    def __init__(self):
        self.prob = {}
        self.unknown = {}
        self.prior={}
        self.idf={}

    def fit(self, X, Y,percent):
        class_sep = {}
        for i in range(0, len(X)):
            temp = Y[i]
            if (temp not in class_sep.keys()):
                class_sep[temp] = []
            class_sep[temp].append(X[i])

        total=set([])
        wordlist=[]
        countlist=[]

        print("Phase 1")

        for i in class_sep.keys():
            temp = list([])
            for j in class_sep[i]:
                prep = preprocess(j)
                temp.extend(prep)
            word,count = np.unique(temp,return_counts=True)
            total=total | set(word)
            wordlist.append(word)
            countlist.append(count)

        print("Phase 2")
        for i in total:
            count=0
            for j in range(0,len(wordlist)):
                if i in wordlist[j]:
                    count=count+1
            if i not in self.idf.keys():
                self.idf[i]=math.log2(len(class_sep.keys())/float(count))
            else:
                print("error")

        print("Phase 3")
        total = set([])
        for i in range(0,len(wordlist)):
            for j in range(0,len(countlist[i])):
                countlist[i][j]=math.log2(1+countlist[i][j])*self.idf[wordlist[i][j]]
            index = np.argsort(countlist[i])
            index=index[::-1]

            for j in range(0,int(len(index)*percent)):
                total.add(wordlist[i][index[j]])

        print("Phase 4")
        for i in class_sep.keys():
            temp = list([])
            for j in class_sep[i]:
                prep = preprocess(j)
                temp.extend(prep)
            word, count = np.unique(temp, return_counts=True)

            self.prob[i] = {}

            word_b = []
            count_b=[]
            for j in range(0,len(word)):
                if word[j] in total:
                    word_b.append(word[j])
                    count_b.append(count[j])

            for j in range(len(word_b)):
                if (word_b[j] not in self.prob[i].keys()):
                    self.prob[i][word_b[j]] = (count_b[j]+1) / float((np.sum(count_b) + len(word_b))+1)
                else:
                    print('Error')
            self.unknown[i] = 1 / float((np.sum(count_b) + len(word_b))+1)

    def predict(self, X):
        result = []
        for i in range(0, len(X)):
            if (i % 50 == 0):
                print('File Number : ', i)
            maxliklihood = float('-inf')
            result_temp = 'medical'
            file_list = preprocess(X[i])
            word, count = np.unique(file_list, return_counts=True)
            for c in self.prob.keys():
                liklihood = 0
                for w in range(0, len(word)):
                    if (word[w] in self.prob[c].keys()):
                        liklihood = liklihood + count[w] * math.log2(self.prob[c][word[w]])
                    else:
                        liklihood = liklihood + count[w] * math.log2(self.unknown[c])
                if (liklihood > maxliklihood):
                    result_temp = c
                    maxliklihood = liklihood
            result.append(result_temp)
        return result
