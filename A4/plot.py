import matplotlib.pyplot as plt

rocchio = [89.56, 90.9, 92.8]
knn1 =  [93.2, 94.7, 96.4]
knn3 = [92.36, 94.2, 95.0]
knn5 = [92.24, 94.7, 95.2]

x= [0.5,0.3,0.1]

plt.plot(x,rocchio,'r',label="Rocchio")
plt.plot(x,knn1,'b',label="KNN K=1")
plt.plot(x,knn3,'brown',label="KNN K=3")
plt.plot(x,knn5,'g',label="KNN K=5")
plt.xlabel("Test Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
