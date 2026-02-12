from Extract import *
from sklearn import model_selection
import sklearn.linear_model as lm
import sys
sys.path.insert(1, r"C:\Users\musti\OneDrive - Danmarks Tekniske Universitet\Y3S1_ML\02450Toolbox_Python\Tools")
from toolbox_02450 import rlr_validate
import numpy as np
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
# load the data
#class label the Car Make and Country columns
dfmulti= pd.read_csv(filename)
for i in range(rows):
    dfmulti.loc[i,"Car Make"]= Carmakerdict[dfmulti.loc[i,"Car Make"]]
for i in range(rows):
    dfmulti.loc[i,"Country "]= Countrydict[dfmulti.loc[i,"Country "]]
#remove the car maker column
dfmulti=dfmulti.drop('Car Make',axis = 1)

#load the data in x and y
X=dfmulti.values[:,1:]
y=dfmulti.values[:,0].squeeze()

X=np.array(X,dtype = np.float64)
y=np.array(y,dtype = np.float64)
# initialize the attribute 
N, M = X.shape
C = len(Countrynames)
Folds = 10
attributeNames=attributeNames[2:]
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.concatenate((np.array(['Offset']),attributeNames))
M = M+1

##cross validation
Outer_CV = model_selection.KFold(n_splits=Folds,shuffle=False)
inner_CV = 5

# Values of lambda
lambdas = np.power(10.,np.arange(-4,9,0.05))


##
y_pred_MLR=np.zeros((len(y)))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((Folds,1))
Error_test = np.empty((Folds,1))
Error_train_rlr = np.empty((Folds,1))
Error_test_rlr = np.empty((Folds,1))
Error_train_nofeatures = np.empty((Folds,1))
Error_test_nofeatures = np.empty((Folds,1))
w_rlr = np.empty((M,Folds))
mu = np.empty((Folds, M-1))
sigma = np.empty((Folds, M-1))
w_noreg = np.empty((M,Folds))

k=0

for train_index, test_index in Outer_CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    ##NOTE rlr_validate HAS BEEN CHANGED FROM THE GIVEN LIBRARY, now it takes a bool as input to block shuffling
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, inner_CV, shuffle=False)
    print("Lambda in outer fold {0} is {1}".format(k+1,opt_lambda))##print lambda
# Standardize outer fold based on training set, and save the mean and standard

    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    m = lm.LogisticRegression(solver='newton-cg', multi_class='multinomial',  max_iter = 100000, C=1/opt_lambda)
    m.fit(X_train,y_train)
    y_test_est = m.predict(X_test)
   
    test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)
    print("Error rate in outer fold {0} is {1}".format(k+1,test_error_rate))##print error rate
    #m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    y_pred_MLR[test_index]=y_test_est##used for stat file
    # Display the results for the last cross-validation fold

    print('Weights in last fold:')
    if k == 7:
        ##show wieghts in the best fold
        for m in range(M):
            print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,7],2)))
    if k == Folds-1:
        
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')