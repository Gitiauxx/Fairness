import pandas as pd
import numpy as np
from test_fairlearn import run_fairlearn
from fairlearn import moments
import audit_tree_conf as ad
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 100)

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

print(test.groupby('gender').size())

feature_list = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 
		'hours-per-week', 'capital-gain', 'education-num']
outcome = 'income'
protected = {'sex': [' Male', ' Female'], 'race':[' Black', ' White']}


# split train and test (70, 30)
np.random.seed(seed=1)
train['attr'] = train['gender']

# logistic regression
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
logreg.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = logreg.predict(np.array(test[feature_list]))
train['predict'] = logreg.predict(np.array(train[feature_list]))

"""
# reduction method
epsilon = 0.01
cons = moments.EO()
results_agg = run_fairlearn(train, test, feature_list, outcome, protected, cons, epsilon, size=1)
results_agg['method'] = 'FR_AGG'
"""

# auditing
score = ad.audit_tree(test, feature_list, 'predict', protected)
print(score)



