from Extract import *
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

# Load the X and Y data note y is k-encoded
X=Data_for_class.values[:,:5]
y=Data_for_class.values[:,5:]

# initialize the attribute 
N, M = X.shape
C = len(Countrynames)
Folds = 10

##cross validation
Outer_CV = model_selection.KFold(n_splits=Folds,shuffle=False)
inner_CV = 5

## Initialize parameters
k_range = list(range(1, 51))
param_grid = dict(n_neighbors=k_range)
dist=2
metric = 'minkowski'
metric_params = {}
Max_neighbours = 40
k_range = list(range(1, Max_neighbours))
mu = np.empty((Folds, M-1))
sigma = np.empty((Folds, M-1))
currentfold=1

y_pred_KNN=np.zeros((len(y),len(Countrynames)))#used for stat file
##outer fold

for train_ix, test_ix in Outer_CV.split(X):
    
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]

    # Standardize outer fold based on training set, and save the mean and standard
    mu[currentfold-1, :] = np.mean(X_train[:, 1:], 0)
    sigma[currentfold-1, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[currentfold-1, :] ) / sigma[currentfold-1, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[currentfold-1, :] ) / sigma[currentfold-1, :] 
    # configure the inner cross-validation procedure
    param_grid = dict(n_neighbors=k_range)
    knn = KNeighborsClassifier(n_neighbors=0, p=dist, metric=metric, metric_params=metric_params)
    grid = model_selection.GridSearchCV(knn, param_grid, cv=inner_CV, scoring='accuracy', return_train_score=True)
    a=grid.fit(X_train, y_train)
    print(f" Best score is: {1-a.best_score_} with parameters: {a.best_params_}")
    scores_knn = grid.cv_results_['mean_test_score']


    best_KNN_model=a.best_estimator_
    a.fit(X_train,y_train)
    y_pred_KNN[test_ix,:]=a.predict(X_test)


    currentfold+=1
