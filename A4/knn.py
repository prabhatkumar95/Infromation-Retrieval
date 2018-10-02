from preprocess import get_token_count,create_idf
import numpy as np


class KNN():
    def __init__(self):

        self.idf = None
        self.tf = None
        self.file_list=None
        self.target={}

    def fit(self,X,Y):
        print("Calculating IDF")
        self.idf,self.tf,self.file_list = create_idf(X)

        for doc in range(0,len(X)):
            self.target[X[doc]]=Y[doc]


    def predict(self,X):
        result1=[]
        result3=[]
        result5=[]
        for test in range(0, len(X)):
            score=[]
            label=[]

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

            for v in self.file_list.keys():
                doc_vector = np.zeros(len(refined_word))
                for w in range(0, len(refined_word)):
                    doc_vector[w] = np.log(self.tf[refined_word[w]][int(self.file_list[v])] + 1) * self.idf[
                        refined_word[w]]
                denim = float(
                    np.linalg.norm(testvector) * np.linalg.norm(doc_vector))
                if (denim == 0):
                    score.append(0)
                    label.append(self.target[v])
                else:
                    score.append(float(np.dot(testvector.T, doc_vector)) / denim)
                    label.append(self.target[v])
            temp = np.argsort(score)
            temp1 = temp[-1:]
            temp3 = temp[-3:]
            temp5 = temp[-5:]
            label_f1 = []
            label_f3 = []
            label_f5 = []


            for i in range(0,len(temp1)):
                label_f1.append(label[temp1[i]])
            for i in range(0, len(temp3)):
                label_f3.append(label[temp3[i]])
            for i in range(0, len(temp5)):
                label_f5.append(label[temp5[i]])


            temp1, count1 = np.unique(label_f1, return_counts=True)
            temp3,count3 = np.unique(label_f3,return_counts=True)
            temp5, count5 = np.unique(label_f5, return_counts=True)
            r1 = np.argmax(count1)
            result1.append(temp1[r1])
            r3 = np.argmax(count3)
            result3.append(temp3[r3])
            r5 = np.argmax(count5)
            result5.append(temp5[r5])
        return result1,result3,result5
