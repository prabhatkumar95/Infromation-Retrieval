from function import preprocess
import numpy as np
import math


class NB():
    def __init__(self):
        self.prob = {}
        self.unknown = {}
        self.prior = {}
        self.idf = {}

    def fit(self, X, Y):
        class_sep = {}
        for i in range(0, len(X)):
            temp = Y[i]
            if (temp not in class_sep.keys()):
                class_sep[temp] = []
            class_sep[temp].append(X[i])

        for i in class_sep.keys():
            temp = list([])
            for j in class_sep[i]:
                prep = preprocess(j)
                temp.extend(prep)
            word, count = np.unique(temp, return_counts=True)
            self.prob[i] = {}

            for j in range(len(word)):
                if (word[j] not in self.prob[i].keys()):
                    self.prob[i][word[j]] = (count[j] + 1) / float((np.sum(count) + len(word)) + 1)
                else:
                    print('Error')
            self.unknown[i] = 1 / float((np.sum(count) + len(word)) + 1)

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
