from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linprog
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import pandas as pd
from scipy.optimize import SR1
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import copy
import pickle
import math
from sklearn.metrics import confusion_matrix
import os
import sys

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
    for i in np.arange(trainY.shape[0]):
        cost1[i] = 1 - trainY[i] + lag_mult[i, 1].sum() - lag_mult[i, 0].sum()
    
    cost0 = trainY
    
    # create weight vectors
    W = np.absolute(cost0 - cost1)
    Y = (cost0 >= cost1).astype('int32')
    
    # fitting with weights
    cls.fit(trainX, np.ravel(Y), sample_weight=W)
    
    # save classifier as a binary model
    learner_pickled = pickle.dumps(cls)
    
    return learner_pickled
    
def best_lag(estimators_list, trainX, g, delta, epsilon, lag_increment):
    
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
    
    count_sample = 0
    while count_sample < 10000:

        i = np.random.randint(low=0, high=trainX.shape[0], size=1)
        j = np.random.randint(low=0, high=trainX.shape[0], size=1)
        count_sample += 1
        distance =  np.sqrt(malahanobis_distance(trainX[i, :], trainX[j, :] , g, trainX.shape[1]))
        distanceplus = (predict[i] - predict[j])  - math.exp(epsilon) * distance 
        distanceminus = (predict[j] - predict[i]) - math.exp(epsilon) * distance
            
        if distanceplus > delta:
            if i < j: 
                lag_mult[i, 1] = - lag_increment
                lag_mult[j, 0] = - lag_increment
                lag_mult[i, 0] = lag_increment
                lag_mult[j, 1] = lag_increment
            elif i > j: 
                lag_mult[i, 1] = lag_increment
                lag_mult[j, 0] = lag_increment
                lag_mult[i, 0] = -lag_increment
                lag_mult[j, 1] = -lag_increment
            
        if distanceminus > delta:
            if i < j: 
                lag_mult[i, 1] = lag_increment
                lag_mult[j, 0] = lag_increment
                lag_mult[i, 0] = -lag_increment
                lag_mult[j, 1] = -lag_increment
            elif i > j: 
                lag_mult[i, 1] = -lag_increment
                lag_mult[j, 0] = -lag_increment
                lag_mult[i, 0] = lag_increment
                lag_mult[j, 1] = lag_increment  
                
    return lag_mult


def classifier_csc(cls, trainX, trainY, g, delta, lag_increment, niter, epsilon):
    
    lagrangian = np.zeros((trainX.shape[0], 2))
    estimators_list = []
    
    # the classifier may shuffle trainX and trainY but
    # we need to keep track of the original order because 
    # we update Lagrangian
    trainX_copy = copy.deepcopy(trainX)
    trainY_copy = copy.deepcopy(trainY)                     
    
    for iter in np.arange(niter):
        #lag_increment = lag_increment / (iter + 1)
        
        learner_pickled = best_classifier(cls, lagrangian, trainX, trainY)
        
        # best learner given multipliers
        estimators_list.append(('classifier{}'.format(iter), learner_pickled))
        
        trainX = trainX_copy
        trainY = trainY_copy
        
        # new lagrangina given voting majority up now
        lag_mult = best_lag(estimators_list, trainX, g, delta, epsilon, lag_increment)
        
        # number of constraints violated
        print("the number of violated constraints is {}".format(np.sum((lag_mult != 0).astype('int32'))))
        lagrangian = (lagrangian * iter + lag_mult) / (iter + 1)
        
                          
    return estimators_list
    

def update_metric(sample, outcomes, g, k):
    
    # constraints
    cons = []
    for i in range(sample.shape[0]):
        for j in range(sample.shape[0]):
                cons.append({"type": "ineq", "fun": lambda x: metric_constraint(sample, outcomes, x, k, i, j) })
                
    # impose metric to have nonnegative parameters
    bnds = [(0, None) for i in range(g.shape[0])]
    # optimization
    def obj_fun(x):
        return np.sum((g - x) ** 2)
    
    def jac_obj(x):
        return 2 * (x -g)
    
    xinit = np.ones(int(k * (k + 1) / 2)) / k
    
    res = minimize(obj_fun, jac=jac_obj, bounds=bnds, x0=xinit, constraints=cons, method='SLSQP',
                 )
    
    return res.x

def oracle_fpr(sample):
    
    # confusion matrix
    cm_male = confusion_matrix(np.array(sample[sample.sex == ' Male'].income), 
                               np.array(sample[sample.sex == ' Male'].predict))
    cm_male = cm_male / cm_male.sum(axis=1)[:, np.newaxis]
    
    cm_female = confusion_matrix(np.array(sample[sample.sex == ' Female'].income), 
                                 np.array(sample[sample.sex == ' Female'].predict))
    cm_female = cm_female / cm_female.sum(axis=1)[:, np.newaxis]
    
    if cm_female[1,1] < 0.8 * cm_male[1, 1]:
        return "Unfair"
    else:
        return "Fair"
		

def communicating_oracle(train, cls, feature_list, delta, lag_increment, niter, nsample, epsilon):
    
    # initialize distance
    ng = len(feature_list)
    ng = int(ng * (ng + 1) /2)
    g = np.ones(ng) 
    
    # normalize inputs
    for col in feature_list:
        train[col] = train[col] - train[col].mean()
        train[col] = train[col] /train[col].var() ** 0.5
    
    trainX = np.array(train[feature_list])
    trainY = np.array(train['income'])
   
    
    is_fair ="Unfair"
    iteration = 0
    
    while (is_fair == "Unfair") & (iteration < 10):
        iteration += 1
        
        # run classifier
        clf_list = classifier_csc(cls, trainX, trainY, g, delta, lag_increment, niter, epsilon)
    
        # predict outcomes
        predict = np.zeros(trainX.shape[0])
        for h in clf_list:
            learner = pickle.loads(h[1])
            predict += learner.predict(trainX)
        predict = predict / niter
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        train['predict'] = predict
    
        # oracle decision on fairness
        count_sample = 0
        while count_sample < 1000:
            count_sample += 1
            sample = train.loc[np.random.choice(train.index, nsample, replace=False)]
            is_fair = oracle_fpr(sample)
            if is_fair == 'Unfair':
                break
        
        print( " the oracle says that the classifier is {} after {} draws" .format(is_fair, count_sample))
        
        if is_fair == "Unfair":
            sample_array = np.array(sample[feature_list])
            outcomes = np.array(sample['predict'])
            g = update_metric(sample_array, outcomes, g, trainX.shape[1])
        
    
    return clf_list


def test_oracle_adults(dataname, datatest):
	# load data
	train = pd.read_csv(dataname)
	
	# clean features
	train['workclass'] = train['workclass'].astype('category').cat.codes
	train['education'] = train['education'].astype('category').cat.codes
	train['occupation'] = train['occupation'].astype('category').cat.codes
	train['relationship'] = train['relationship'].astype('category').cat.codes
	train['marital-status'] = train['marital-status'].astype('category').cat.codes
	train['income'] = train['income_bracket'].astype('category').cat.codes
	train['gender'] =  train['sex'].astype('category').cat.codes
	train['srace'] =  train['race'].astype('category').cat.codes
	
	feature_list = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 
		'hours-per-week', 'capital-gain', 'education-num', 'gender']
		
	# standardize data
	for col in feature_list:
		train[col] = train[col] - train[col].mean()
		train[col] = train[col] /train[col].var() ** 0.5
	
	delta = 0.5
	epsilon = 0.2
	lag_increment = 20
	logreg = LogisticRegression()
	niter = 10
	nsample = 100
	clf_list = communicating_oracle(train, logreg, feature_list, delta, lag_increment, niter, nsample, epsilon)
	
	# test data
	test = pd.read_csv(datatest)
	
	# clean features
	test['workclass'] = test['workclass'].astype('category').cat.codes
	test['education'] = test['education'].astype('category').cat.codes
	test['occupation'] = test['occupation'].astype('category').cat.codes
	test['relationship'] = test['relationship'].astype('category').cat.codes
	test['marital-status'] = test['marital-status'].astype('category').cat.codes
	test['income'] = test['income_bracket'].astype('category').cat.codes
	test['gender'] =  test['sex'].astype('category').cat.codes
	test['srace'] =  test['race'].astype('category').cat.codes
	
	# predict outcomes
	predict = np.zeros(test.shape[0])
	for h in clf_list:
		learner = pickle.loads(h[1])
		predict += learner.predict(np.array(test[feature_list]))
	predict = predict / niter
	predict[predict > 0.5] = 1
	predict[predict <= 0.5] = 0
	test['predict'] = predict
	
	# confusion matrix 
	cm_male = confusion_matrix(np.array(test[test.sex == ' Male'].income), 
                               np.array(test[test.sex == ' Male'].predict))
	cm_male = cm_male / cm_male.sum(axis=1)[:, np.newaxis]
    
	cm_female = confusion_matrix(np.array(test[test.sex == ' Female'].income), 
                                 np.array(test[test.sex == ' Female'].predict))
	cm_female = cm_female / cm_female.sum(axis=1)[:, np.newaxis]
	
	
	return cm_male, cm_female
	
if __name__ == '__main__':

	cm_male, cm_female = test_oracle_adults(sys.argv[1],, sys.argv[2])
	print(cm_male)
	print(cm_female)
	
	
		
	
		