import numpy as np
import matplotlib.pyplot as plt
import seaborn


accuracy=[0.942,0.9553333333333334,0.956]
accuracy_tf_01=[0.762,0.4846666666666667,0.648]
accuracy_tf_05=[0.8416,0.8373333333333334,0.936]
accuracy_tf_08=[0.9508,0.948,0.966]

x = [0.5,0.3,0.1]

plt.plot(x,accuracy,'r',label="Without TF-IDF")
plt.plot(x,accuracy_tf_01,'b',label="10% TF-IDF")
plt.plot(x,accuracy_tf_05,'brown',label="50% TF-IDF")
plt.plot(x,accuracy_tf_08,'g',label="80% TF-IDF")
plt.xlabel("Test Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
