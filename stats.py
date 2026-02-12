#%%no repeats
from Extract import *
from KNNBUTBETTER import y_pred_KNN
from Class import y_pred_MLR
from Baseline import y_pred_Baseline , y_true
import numpy as np
import sys
sys.path.insert(1, r"C:\Users\musti\OneDrive - Danmarks Tekniske Universitet\Y3S1_ML\02450Toolbox_Python\Tools")
from toolbox_02450 import mcnemar

#convert to class labels 
y_pred_KNN = np.argmax(y_pred_KNN, axis=1)+1
y_pred_Baseline = np.argmax(y_pred_Baseline, axis=1)+1
y_true = np.argmax(y_true, axis=1)+1
#%%
##KNN vs Multinomal Logistic Regression
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_pred_KNN, y_pred_MLR, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# %%
##KNN vs Baseline
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_pred_KNN, y_pred_Baseline, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# %%
##Multinomal Logistic Regression vs Baseline
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_pred_MLR, y_pred_Baseline, alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
