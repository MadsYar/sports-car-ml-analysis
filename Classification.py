from Extract import *
import numpy as np
import matplotlib.pyplot as plt

X_class= X.copy()
X_class = X_class[:,3:] ## attributes used for Classification
Y_class = np.array([Countrydict[cl] for cl in Country]) ## the y value used for Classification
attributeNames = df.columns[range(3,7)] ##names of attribute

i = 0
j = 3
color = ['r','g', 'b','c','m']
print(attributeNames)
plt.title('Sport Cars Classification Problem')
for c in range(len(Countrynames)):
    idx = Y_class == c
    plt.scatter(x=X_class[idx, i],
                y=X_class[idx, j], 
               #c=color[c], 
                s=25, alpha=1,
                label=Countrynames[c])
plt.legend()
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

plt.xticks(rotation=90)

plt.show()


