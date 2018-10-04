import pandas as pd
import numpy as np
from fairndist_cluster import run_classifier
from fairlearn import moments
from test_fairlearn import run_fairlearn
pd.set_option('display.max_columns', 500)




# train and test data
train = pd.read_csv('..\\data\\adult_income_dataset.csv')
	
# clean features
train['workclass'] = train['workclass'].astype('category').cat.codes
train['education'] = train['education'].astype('category').cat.codes
train['occupation'] = train['occupation'].astype('category').cat.codes
train['relationship'] = train['relationship'].astype('category').cat.codes
train['marital-status'] = train['marital-status'].astype('category').cat.codes
train['income'] = train['income_bracket'].astype('category').cat.codes
train['gender'] =  train['sex'].astype('category').cat.codes
train['srace'] =  train['race'].astype('category').cat.codes

test = pd.read_csv('..\\data\\adult_income_test.csv')
	
# clean features
test['workclass'] = test['workclass'].astype('category').cat.codes
test['education'] = test['education'].astype('category').cat.codes
test['occupation'] = test['occupation'].astype('category').cat.codes
test['relationship'] = test['relationship'].astype('category').cat.codes
test['marital-status'] = test['marital-status'].astype('category').cat.codes
test['income'] = test['income_bracket'].astype('category').cat.codes
test['gender'] =  test['sex'].astype('category').cat.codes
test['srace'] =  test['race'].astype('category').cat.codes

feature_list = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'gender', ]
outcome = 'income'
protected = {'sex': [' Male', ' Female'], 'race':[' Black', ' White']}

# classifier -- aggregate
train['attr'] =  train['srace']
cons = moments.EO()
epsilon = 0.02

results_agg = run_fairlearn(train, test, feature_list, outcome, protected, cons, epsilon, size=1)
results_agg['method'] = 'FR_AGG'

# classifier -- individual
size = 1
niter = 8
epsilon = 0

results_ind = run_classifier(train, test, feature_list, outcome, protected, niter, size, epsilon)
results_ind['method'] = 'FR_IND'

results = pd.concat([results_agg, results_ind])
results.to_csv('..\\results\\fairness_adult_10052018_clusters.csv')