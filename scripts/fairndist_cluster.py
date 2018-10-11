from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linprog
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import pandas as pd
from scipy.optimize import SR1
import copy
import pickle
import math
from sklearn.metrics import confusion_matrix
import os
import sys
from clustering import KCons

def to_matrix(g, k):
    """A useful function to turn  a vector into a symmetric matrix"""
    G = np.zeros((k, k))
    pointer = 0
    for i in np.arange(k):
        for j in np.arange(i, k):
            G[i, j] = g[pointer] 
            pointer += 1
        for j in np.arange(0, i):
            G[i, j] = G[j, i]
    return G
	
def distance_sample(sample):
    """compute a n*(n+1) / 2 by k matrix of individual distance"""
    n = sample.shape[0]
    k = sample.shape[1]  
    distances = np.zeros((n, k))
    
    row_count = 0
    for i in np.arange(k):
        for j in np.arange(i):
            distances[row_count, :] = sample[i, :] - sample[j, :]
            row_count += 1
    return distances
	
def malahanobis_distance(x, y, g, k):
    """
    k: features dimension
    g: metric parameters
    """
    G = to_matrix(g, k)
    return np.dot((x - y), np.dot(G, (x - y).transpose()))	


def metric_constraint(sample, outcomes, g, k, i, j):
    distance = malahanobis_distance(sample[i,:], sample[j, :], g, k)
    return distance - (outcomes[i] - outcomes[j]) ** 2
	
	
def best_classifier(cls, lag_mult, trainX, trainY):
    
    """
    lag_mult: dictionary of mulitpliers. For each key, the first entry is 
    an array of multipliers for j<key and the second entry
    is an array of multipliers for j > key
    
    cls: classification methods with a fit attribute 
    that accepts weight
    """
    
    cost1 = np.zeros(trainY.shape[0])
    cost0 = np.zeros(trainY.shape[0])
    for i in np.arange(trainY.shape[0]):
        cost1[i] = 1 - trainY[i] + lag_mult[i, 0]
        cost0[i] = trainY[i] + lag_mult[i, 1]
   
    # create weight vectors
    W = np.absolute(cost0 - cost1)
    Y = (cost0 >= cost1).astype('int32')
    
    # fitting with weights
    cls.fit(trainX, np.ravel(Y), sample_weight=W)
    
    # save classifier as a binary model
    learner_pickled = pickle.dumps(cls)
    
    return learner_pickled
	
   
def best_lag(estimators_list, trainX, trainY, g, delta, epsilon, lag_increment):
    
    """
    Q is a learner algorithm (sum of multiple algorithms)
    
    output: a n by 2 matrix: first column for i = a sum of lagrangian for all constraints (i,j) 
    violated with i < j;
    second column for i: a sum of lagrangian for all constraints (i,j) 
    violated with i > j
    """
    predict = np.zeros(trainX.shape[0])
    for h in estimators_list:
        learner = pickle.loads(h[1])
        predict += learner.predict(trainX)
    
    predict = predict / len(estimators_list)
   
    lag_mult = np.zeros((trainX.shape[0], 2))
    
    # clustering
    def dist_func(x, y):
        return malahanobis_distance(x, y, g, trainX.shape[1] - 2)
		
    clus = KCons(10, trainX, predict, dist_func, estimators_list)
    clus.find_centers()
    print(len(clus.mu))
    xcluster = clus.clusters
    cluster_class = clus.mu_classified
    mask = clus.mask
	
    for i in np.arange(lag_mult.shape[0]):
        if mask[i] == 0:
            if predict[i] > cluster_class[int(xcluster[i])]:
                lag_mult[i, 0] = lag_increment
            elif predict[i] < cluster_class[int(xcluster[i])]:
                lag_mult[i, 1] = lag_increment
		

    return lag_mult
	


def classifier_csc(cls, trainX, trainY, g, delta, lag_increment, niter, epsilon):
    
    lagrangian = np.zeros((trainX.shape[0], 2))
    estimators_list = []
    
    # the classifier may shuffle trainX and trainY but
    # we need to keep track of the original order because 
    # we update Lagrangian
    trainX_copy = copy.deepcopy(trainX)
    trainY_copy = copy.deepcopy(trainY)       
    a = 1
    
    for iter in np.arange(niter):
        lag_increment = lag_increment 
      
        
        learner_pickled = best_classifier(cls, lagrangian, trainX, trainY)
        
        # best learner given multipliers
        estimators_list.append(('classifier{}'.format(iter), learner_pickled))
        
        trainX = trainX_copy
        trainY = trainY_copy
        
        # new lagrangian given voting majority up now
        lag_mult = best_lag(estimators_list, trainX, trainY, g, delta, epsilon, lag_increment)
        
        # number of constraints violated
        print("the number of violated constraints is {}".format(lag_mult[lag_mult != 0].shape)) 
        #lagrangian = (lagrangian * iter + lag_mult) / (iter + 1)
        lagrangian = (lagrangian * iter + lag_mult) / (iter + 1)
                          
    return estimators_list
    


def run_classifier(train, test, feature_list, outcome, protected, niter, size, epsilon):
		
	# distance
	ng = len(feature_list) - 2
	ng = int(ng * (ng + 1) /2)
	g = np.ones(ng) / ng
	
	# standardize data
	for col in feature_list:
		test[col] = test[col] - test[col].mean()
		test[col] = test[col] / test[col].var() ** 0.5
		#range_col = test[col].max() - test[col].min()
		#test[col] = (test[col] - test[col].min()) / range_col
		
	# standardize data
	for col in feature_list:
		#range_col = train[col].max() - train[col].min()
		#train[col] = (train[col] - train[col].min()) / range_col
		train[col] = train[col] - train[col].mean()
		train[col] = train[col] /train[col].var() ** 0.5

	delta = 0.1
	lag_increment = 0.25
	
	
	trainX = np.array(train[feature_list])
	trainY = np.array(train[outcome])
	
	results = pd.DataFrame(index= np.arange(size))
	
	for i in np.arange(size):
		logreg = LogisticRegression(random_state=5)
		clf_list = classifier_csc(logreg, trainX, trainY, g, delta, lag_increment, niter, epsilon)	
		
		predict = np.zeros(test.shape[0])
		for h in clf_list:
			learner = pickle.loads(h[1])
			predict += learner.predict(np.array(test[feature_list]))
		
		predict = predict / niter
		
		predict[predict > 0.5] = 1
		predict[predict <= 0.5] = 0
		test['predict'] = predict
		
		# confusion matrix for protected attributes
		for varname in protected.keys():
			for var in protected[varname]:
				cm = confusion_matrix(np.array(test[test[varname] == var][outcome]), 
                               np.array(test[test[varname] == var].predict))
				cm = cm / cm.sum(axis=1)[:, np.newaxis]
				
				results.loc[i, 'tpr_%s'%var] = cm[0,0]
				results.loc[i, 'tnr_%s'%var] = cm[1, 1]
				results.loc[i, 'dp_%s'%var] = test[test[varname] == var].predict.mean()
				
		# accuracy
		results.loc[i, 'accuracy'] = len(test[test.predict == test[outcome]]) / len(test.predict)
		print(results) 
		epsilon += 0.25
		
	return results
	
if __name__ == '__main__':


	results = test_iter(sys.argv[1], sys.argv[2], 30, 10)
	#results.to_csv('..\\results\\fairness_09262018.csv')
	
	
	
		
	
		