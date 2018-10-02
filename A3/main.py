import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn.model_selection import train_test_split
from NB_tf import NB_tf
import numpy as np
from function import accuracy
from sklearn.metrics import confusion_matrix
from NB import NB

def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


file_name = []
label = []
for path, subdirs, files in os.walk('20_newsgroups'):
    for name in files:
        t = path.split('\\')[1]
        label.append(t)
        file_name.append((os.path.join(path, name)))

# print(len(file_name), len(label))
#
# print(np.unique(label))



for i in [0.1,0.3,0.5]:
    print(i,"Started")
    for j in [0.1,0.5,0.8]:
        print(i,j, "Feature Started")

        trainx, testx, trainy, testy = train_test_split(file_name, label, shuffle=True, stratify=label, test_size=i)

        model = NB_tf()
        model.fit(trainx,trainy,j)
        result = model.predict(testx)
        acr=accuracy(result,testy)
        temp="TF_IDF_"+str(j)+"_"+str(i)+" "+str(acr)+"\t"
        f = open('accuracy.txt', mode='a')
        f.write(temp)
        print(temp)
        f.close()
        # print(np.unique(result,return_counts=True))

        #
        # # print(np.unique(testy,return_counts=True))
        conf= confusion_matrix(testy,result,labels=['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'])
        norm_conf= (conf.T/conf.sum(axis=1)).T
        df_cm = pd.DataFrame(norm_conf, index = ['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'],
                      columns = ['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'])
        plt.figure(figsize = (10,7))
        ax=seaborn.heatmap(df_cm, annot=True)
        ax.set(xlabel='Ground Truth', ylabel='Predicted')
        plt.title("TF-IDF/"+str(j)+"/"+str(i))
        plt.savefig("plots/TF_IDF_"+str(j)+"_"+str(i)+".png")

        # plt.show()
        #
        #
        model=NB()
        model.fit(trainx,trainy)
        result = model.predict(testx)
        acr=accuracy(result,testy)
        #
        f = open('accuracy.txt', mode='a')
        #
        f.write("NA_NA_" + str(j) + "_" + str(i) + " " + str(acr) + "\n")
        f.close()
        conf= confusion_matrix(testy,result,labels=['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'])
        norm_conf= (conf.T/conf.sum(axis=1)).T
        df_cm = pd.DataFrame(norm_conf, index = ['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'],
                      columns = ['comp.graphics','rec.sport.hockey','sci.med','sci.space',
        'talk.politics.misc'])
        plt.figure(figsize = (10,7))
        ax=seaborn.heatmap(df_cm, annot=True)
        ax.set(xlabel='Ground Truth', ylabel='Predicted')
        plt.title("NA/NA/"+str(i))
        plt.savefig("plots/NA_NA_"+str(j)+"_"+str(i)+".png")
        # plt.show()
        #
        #
