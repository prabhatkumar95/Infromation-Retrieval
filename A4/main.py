import os
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess import create_idf
from rocchio import Roccio
from knn import KNN
import matplotlib.pyplot as plt

def accuracy(predicted,truth):
    count=0
    for i in range(0,len(predicted)):
        if(predicted[i]==truth[i]):
            count=count+1
    return count*100/len(predicted)

def plot_conf(predicted,truth,filename,title):
    conf = confusion_matrix(truth, predicted, labels=['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space',
                                                   'talk.politics.misc'])
    norm_conf = (conf.T / conf.sum(axis=1)).T
    df_cm = pd.DataFrame(norm_conf, index=['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space',
                                           'talk.politics.misc'],
                         columns=['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space',
                                  'talk.politics.misc'])
    plt.figure(figsize=(10, 7))
    ax = seaborn.heatmap(df_cm, annot=True)
    ax.set(xlabel='Ground Truth', ylabel='Predicted')
    plt.title(title)
    plt.savefig("plots/"+filename+".png")

#read dataset and add ground truth to labels


file_name = []
label = []
for path, subdirs, files in os.walk('20_newsgroups'):
    for name in files:
        t = path.split('\\')[1]
        label.append(t)
        file_name.append((os.path.join(path, name)))


# print(len(file_name), len(label))
# print(np.unique(label,return_counts=True))
accrr=[]
accr1=[]
accr3=[]
accr5=[]
for testsize in [0.5,0.2,0.1]:
    #split data according to the given ratio
    trainx, testx, trainy, testy = train_test_split(file_name, label, shuffle=True, stratify=label, test_size=testsize)


    model = Roccio()
    model.fit(trainx,trainy)
    predict = model.predict(testx)
    accrr.append(accuracy(predict,testy))
    filename = "Rocchio_"+str(testsize)
    title = "Rocchio Split : "+str((1-testsize)*100)+"-"+str(testsize*100)
    plot_conf(predict,testy,filename,title)



    model = KNN()
    model.fit(trainx,trainy)
    predict1,predict3,predict5 = model.predict(testx)
    accr1.append(accuracy(predict1,testy))
    accr3.append(accuracy(predict3, testy))
    accr5.append(accuracy(predict5, testy))
    filename1 = "KNN_1_" + str(testsize)
    filename3 = "KNN_3_" + str(testsize)
    filename5 = "KNN_5_" + str(testsize)

    title1 = "KNN K=1 Split : " + str((1 - testsize) * 100) + "-" + str(testsize * 100)
    title3 = "KNN K=3 Split : " + str((1 - testsize) * 100) + "-" + str(testsize * 100)
    title5 = "KNN K=5 Split : " + str((1 - testsize) * 100) + "-" + str(testsize * 100)
    plot_conf(predict1, testy, filename1, title1)
    plot_conf(predict3, testy, filename3, title3)
    plot_conf(predict5, testy, filename5, title5)


print(accrr,accr1,accr3,accr5)