import numpy as np
import os
from preprocess import create_idf, get_token_count
import numpy as np


class Roccio():
    def __init__(self):
        self.centeroid = {}
        self.idf = None

    def fit(self, X, Y):
        print("Calculating IDF")
        self.idf = create_idf(X)[0]
        class_sep = {}
        for i in range(0, len(X)):
            temp = Y[i]
            if temp not in class_sep.keys():
                class_sep[temp] = []
            class_sep[temp].append(X[i])

        # initialising centeroid

        for c in class_sep.keys():
            self.centeroid[c] = {}
            for word in self.idf.keys():
                self.centeroid[c][word] = 0
        print("Calculating Centeroid")
        for c in class_sep.keys():
            docs = class_sep[c]
            for doc in docs:
                word, count = get_token_count(doc)
                for w in range(0, len(word)):
                    self.centeroid[c][word[w]] = self.centeroid[c][word[w]] + count[w]

        for c in self.centeroid.keys():
            for w in self.centeroid[c].keys():
                self.centeroid[c][w] = self.centeroid[c][w] / float(len(class_sep[c]))

    def predict(self, X):
        result = []
        for test in range(0, len(X)):
            word, count = get_token_count(X[test])
            refined_word = []
            refined_count = []

            for w in range(0, len(word)):
                if word[w] in self.idf.keys():
                    refined_word.append(word[w])
                    refined_count.append(count[w])
            testvector = np.array(refined_count)

            for i in range(0, len(testvector)):
                testvector[i] = np.log(testvector[i] + 1) * self.idf[refined_word[i]]

            temp_score = float('-inf')
            temp_result = -1

            for c in self.centeroid.keys():
                score = 0
                if len(refined_word) == 0:
                    temp_result = c
                    print("Zero")
                    continue
                else:
                    centeroid_vector = np.zeros(len(refined_word))
                    for w in range(0, len(refined_word)):
                        centeroid_vector[w] = np.log(self.centeroid[c][refined_word[w]] + 1) * self.idf[
                            refined_word[w]]
                    denim = float(
                        np.linalg.norm(testvector) * np.linalg.norm(centeroid_vector))
                    if(denim==0):
                        score=0
                    else:
                        score = float(np.dot(testvector.T, centeroid_vector)) /denim
                    if (score > temp_score):
                        temp_score = score
                        temp_result = c
            result.append(temp_result)
        return result
