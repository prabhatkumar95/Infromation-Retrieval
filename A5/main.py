import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import create_adjmatrix,clustering_c,closeness,betweeness

data = np.loadtxt('CA-GrQc.txt',delimiter="\t")
print(data.shape)



mapping,matrix = create_adjmatrix(data)

# matrix_scaled = matrix[0:1000,0:1000]
# print(matrix_scaled[0,:])
degree = np.sum(matrix,axis=0)
# print(degree[0])

# print(np.average(clustering_c(matrix),axis=0))
deg,count = np.unique(degree,return_counts=True)

# print(deg,count)
#
plt.plot(deg,count,label="Count",c='red')
plt.xlabel("Degree")
plt.title("Degree Distribution")
plt.ylabel("Degree Count")
plt.savefig("DegreeCount.png")
plt.show()
# #
#
c_c = clustering_c(matrix)
print(np.average(c_c))
import json
close = closeness(matrix)
betw = betweeness(matrix)
print(np.average(close,axis=0))
json.dump(c_c,open("clustering_c.json",mode='w'))
json.dump(list(close),open("closeness.json",mode='w'))
json.dump(betw,open("betweeness.json",mode='w'))
# betw = json.loads("betweeness.json")
t= list(betw.values())
print(np.average(t))