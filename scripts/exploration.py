import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from pyemd import emd_samples

# data
train = pd.read_csv('..\\data\\adult_income_dataset.csv')
y_train = np.array(train.income_bracket.astype('category').cat.codes)
train.drop('income_bracket', axis=1, inplace=True)


# features
train['workclass'] = train['workclass'].astype('category').cat.codes
train['education'] = train['education'].astype('category').cat.codes
train['occupation'] = train['occupation'].astype('category').cat.codes
train['relationship'] = train['relationship'].astype('category').cat.codes
train['marital-status'] = train['marital-status'].astype('category').cat.codes

feature_list = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 
		'hours-per-week', 'capital-gain', 'education-num']
train = train[feature_list]

# random forest 
rf = RandomForestClassifier(n_estimators=200)
rf.fit(np.array(train), y_train)

# test data
test = pd.read_csv('..\\data\\adult_income_test.csv')
print(list(set(test.sex)))

# features
test['workclass'] = test['workclass'].astype('category').cat.codes
test['education'] = test['education'].astype('category').cat.codes
test['occupation'] = test['occupation'].astype('category').cat.codes
test['relationship'] = test['relationship'].astype('category').cat.codes
test['marital-status'] = test['marital-status'].astype('category').cat.codes


test1 = test[feature_list]
predicted = rf.predict_proba(np.array(test1))
test['score'] = predicted[:, 1]
test.loc[(test.score < 0.5) , 'predicted'] = 0
test.loc[test.score >= 0.5, 'predicted'] = 1
test.loc[(test.score < 0.5) & (test.sex == ' Female'), 'predicted'] = 0
test.loc[(test.score >= 0.5) & (test.sex == ' Female'), 'predicted'] = 1
test['income_bracket'] = test.income_bracket.astype('category').cat.codes
c = np.array(confusion_matrix(np.array(test.income_bracket), np.array(test.predicted)))
print( c/ c.sum(axis=1)[:, np.newaxis])

c = np.array(confusion_matrix(np.array(test[test.sex == ' Female'].income_bracket), np.array(test[test.sex == ' Female'].predicted)))
print( c/ c.sum(axis=1)[:, np.newaxis])


delta = 0.045
message = 0
beta = 5.5
niter = 500
mae = np.zeros(niter)
for iter in np.arange(niter):
	test_draw = test.loc[np.random.choice(test.index, 100, replace=True), :]
	c = np.array(confusion_matrix(np.array(test_draw.income_bracket), np.array(test_draw.predicted)))

	#distance = 10000
	#while distance > beta:	
	test_draw2 = test.loc[np.random.choice(test.index, 100, replace=True), :]
		#distance = emd_samples(np.array(test_draw2[feature_list]), np.array(test_draw[feature_list]), bins=10)
	
	c1 = np.array(confusion_matrix(np.array(test_draw2.income_bracket), np.array(test_draw2.predicted)))
	c = c/ c.sum(axis=1)[:, np.newaxis]
	c1 = c1/ c1.sum(axis=1)[:, np.newaxis]
	
	
	#if np.abs(c[0,0] - c1[0,0]) > delta:
		#print(emd_samples(np.array(test_draw2[feature_list]), np.array(test_draw[feature_list]), bins=10))
		#message = 1
		#break
	mae[iter] = np.abs(c[1,1] - c1[1,1])

print('the mean distance is {} with std equal to {}'.format(np.mean(mae), np.var(mae) ** (0.5)))


