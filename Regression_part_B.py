import numpy as np, scipy.stats as st
import pandas as pd
from sklearn import model_selection
import sklearn.linear_model as lm
import torch
import sys
sys.path.insert(1, r"/Users/madsyar/Desktop/DTU/General Engineering/5. semester/Introduction to Machine Learning and Data Mining/02450Toolbox_Python/Tools")
from toolbox_02450 import train_neural_net, draw_neural_net

filename = "/Users/madsyar/Downloads/carz.csv"
data = pd.read_csv(filename)
data_values = data.values

attributeNames = ['Year', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']

y = data['Price (in USD)'].squeeze()

X = data_values[:, [3, 4, 5]]
y = data_values[:, [6]]

X = X.astype(np.float64)
y = y.astype(np.float64)

N, M = X.shape

X = st.zscore(X)

# Parameters for neural network classifier
n_hidden_units = 6     # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
model_parameter = np.array([1, 8, 15])

K_Outer = 10
CV_Outer = model_selection.KFold(n_splits=K_Outer,shuffle=True)

loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

errors = [] # make a list for storing generalizaition error in each loop

lambdas = [np.power(10., i) for i in range(-3, 4)]

for (k_out, (train_index_outer, test_index_outer)) in enumerate(CV_Outer.split(X,y)):
    print('\n Crossvalidation fold (outer): {0}/{1}'.format(k_out+1,K_Outer))
    
    # For the PyTorch neural network
    X_train_outer_network = torch.Tensor(X[train_index_outer,:])
    y_train_outer_network = torch.Tensor(y[train_index_outer])
    X_test_outer_network = torch.Tensor(X[test_index_outer,:])
    y_test_outer_network = torch.Tensor(y[test_index_outer])
    
    # For the linear regression model 
    X_train_outer_regression = X[train_index_outer,:]
    y_train_outer_regression = y[train_index_outer]
    X_test_outer_regression = X[test_index_outer,:]
    y_test_outer_regression = y[test_index_outer]
    
    j = 0
    
    errors_outer = np.zeros([3,10])
    errors_inner = np.zeros([3,10])
                
    K_Inner = 10
    CV_Inner = model_selection.KFold(n_splits=K_Inner,shuffle=True)
    
    for k_in, (train_index_2, test_index_2) in enumerate(CV_Inner.split(X_train_outer_regression,y_train_outer_regression)):
        print('\n Crossvalidation fold (inner): {0}/{1}'.format(k_in+1,K_Inner)) 
        
        # For the PyTorch neural network
        X_train_inner_network = torch.Tensor(X_train_outer_network[train_index_2,:])
        y_train_inner_network = torch.Tensor(y_train_outer_network[train_index_2])
        X_test_inner_network = torch.Tensor(X_train_outer_network[test_index_2,:])
        y_test_inner_network = torch.Tensor(y_train_outer_network[test_index_2])
        
        # For the linear regression model 
        X_train_inner_regression = X_train_outer_regression[train_index_2,:]
        y_train_inner_regression = y_train_outer_regression[train_index_2]
        X_test_inner_regression = X_train_outer_regression[test_index_2,:]
        y_test_inner_regression = y_train_outer_regression[test_index_2]
        
        i = 0
        
        # precompute terms
        Xty = X_train_inner_regression.T @ y_train_inner_regression
        XtX = X_train_inner_regression.T @ X_train_inner_regression
        
        for n_hidden_units in model_parameter:
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
    
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_inner_network,
                                                               y=y_train_inner_network,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test_inner_network)
            
            se = (y_test_est.float()-y_test_inner_network.float())**2 # squared error
            
            mse_inner = (sum(se).type(torch.float)/len(y_test_inner_network)).data.numpy() #mean
            
            errors_inner[i,j] = mse_inner
            i+=1
                
        errors_outer[:,j] = errors_outer[:,j]/ len(y_train_outer_regression)
        errors_inner[:,j] = errors_inner[:,j]/ len(y_train_outer_regression)

        j += 1

    print('\n' + 'The result for the outer crossvalidation fold {}'.format(k_out+1) + ' is:' + '\n')
          
    error_model_outer = np.sum(errors_outer,axis=1)
    lamda = lambdas[np.argmin(error_model_outer)]
    
    m = lm.Ridge(alpha = lamda).fit(X_train_outer_regression, y_train_outer_regression)
    error_lambda = ( np.square(y_test_outer_regression-m.predict(X_test_outer_regression)).sum()/len(y_test_outer_regression))
    print("Linear regression (lambda-value and the error)",lamda,error_lambda)
    
    error_model_inner = np.sum(errors_inner,axis=1)
    lamda = model_parameter[np.argmin(error_model_inner)]
    
    m_2 = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, lamda), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(lamda, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
    
    net_2, final_loss_2, learning_curve_2 = train_neural_net(m_2,
                                                           loss_fn,
                                                           X=X_train_outer_network,
                                                           y=y_train_outer_network,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
    
    y_test_est_2 = net_2(X_test_outer_network)
    
    se_2 = (y_test_est_2.float() - y_test_outer_network.float())**2 # squared error
    mse_2 = (sum(se_2).type(torch.float) / len(y_test_outer_network)).data.numpy() #mean
    print("ANN (h-value and the error)",lamda, mse_2)
    
    m3 = sum(y_train_outer_regression) / len(y_train_outer_regression)
    errorBase = (np.square(y_test_outer_regression - m3).sum() / len(y_test_outer_regression))
    print("BASE (the error)", errorBase)
    print('\n')
    
    z_A = np.abs(y_test_outer_regression - m.predict(X_test_outer_regression) ) ** 2
    z_B = np.abs(y_test_outer_network - y_test_est_2.data.numpy() ) ** 2
    z_B = (z_B).data.numpy()
    z_C = np.abs(y_test_outer_regression - m3) ** 2
    
    alpha = 0.05
    z_1 = z_A - z_B
    z_2 = z_A - z_C
    z_3 = z_B - z_C
    
    z_hat_1 = sum(z_1)/len(y_test_outer_network)
    z_hat_2 = sum(z_2)/len(y_test_outer_network)
    z_hat_3 = sum(z_3)/len(y_test_outer_network)
    
    CI_1 = st.t.interval(1-alpha, len(z_1)-1, loc=np.mean(z_1), scale=st.sem(z_1))
    CI_2 = st.t.interval(1-alpha, len(z_2)-1, loc=np.mean(z_2), scale=st.sem(z_2))
    CI_3 = st.t.interval(1-alpha, len(z_3)-1, loc=np.mean(z_3), scale=st.sem(z_3))
    
    p_1 = 2*st.t.cdf(-np.abs( np.mean(z_1) )/st.sem(z_1), df=len(z_1)-1)  # p-value
    p_2 = 2*st.t.cdf(-np.abs( np.mean(z_2) )/st.sem(z_2), df=len(z_2)-1)  # p-value
    p_3 = 2*st.t.cdf(-np.abs( np.mean(z_3) )/st.sem(z_3), df=len(z_3)-1)  # p-value
    
    print('Z_hat value for comparing ANN and linear regression:                 ',z_hat_1)
    print('Z_hat value for comparing ANN and baseline:                          ',z_hat_3)
    print('Z_hat value for comparing linear regression and baseline:            ',z_hat_2)
    print('\n')
    print('Confidence Interval for comparing ANN and linear regression:         ',CI_1)
    print('Confidence Interval for comparing ANN and baseline:                  ',CI_3)
    print('Confidence Interval for comparing linear regression and baseline:    ',CI_2)
    print('\n')
    print('P value for comparing ANN and linear regression:                     ',p_1)
    print('P value for comparing ANN and baseline:                              ',p_3)   
    print('P value for comparing linear regression and baseline:                ',p_2)