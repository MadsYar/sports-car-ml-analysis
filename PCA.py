from Extract import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.preprocessing import normalize, scale
import matplotlib.pyplot as plt
X_data=X.copy()
N=len(((X_data[:,1])))
X_PCA=X_data[:,2:]
# Subtract mean value from data
# Y = X_PCA-np.ones((N,1))*X_PCA.mean(axis=0)
Y=normalize(X_PCA)
Y = scale(X_PCA)
print (Countrydict)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
print(X_PCA.mean(axis=0))
print(Y)
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
print(rho)
threshold = 0.9

#Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# print('Ran Exercise 2.1.3')

##############################################
VT = V.T  
Z = np.dot(Y,V)

i = 0
j = 1

CountryNum = np.asarray([Countrydict[value] for value in Country])
print(Country)
# Plot PCA of the data
f = plt.figure()
plt.title('Sports Cars: PCA')
#Z = array(Z)
for c in range(len(Countrynames)):
    # select indices belonging to class c:
    class_mask = CountryNum==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    print(class_mask)
plt.legend(Countrynames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# #Output result to screen
plt.show()

###########################


print(attributeNames[2:])
N,M = X_PCA.shape
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = 0.2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames[2:])
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title(' PCA Component Coefficients')
plt.show()