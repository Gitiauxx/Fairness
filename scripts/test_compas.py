import pandas as pd
import numpy as np
from test_fairlearn import run_fairlearn
from fairlearn import moments
from fairlearn import classred as red
import audit_tree_conf as ad
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)

# load data
data = pd.read_csv('..\\data\\compas-scores-two-years.csv')
data = data[data.race.isin(['Caucasian', 'African-American'])]


# create categorical data for age_cat, sex, race and charge degree

data['gender'] = data.sex.astype('category').cat.codes
data['age_cat'] = data.age_cat.astype('category').cat.codes
data['charge_degree'] = data.c_charge_degree.astype('category').cat.codes
data['crace'] = data.race.astype('category').cat.codes
data['is_violent_recid'] = data.is_violent_recid.astype('category').cat.codes
data['juv_fel_count'] = data.juv_fel_count.astype('category').cat.codes
data['count_race']  = data['priors_count'] * data['crace']

print(data.groupby('crace').priors_count.describe())

feature_list = ['age_cat',  'priors_count', 'juv_fel_count', 'is_violent_recid']
for var in feature_list:
	data = data[~np.isnan(data[var])]

outcome = 'two_year_recid'
protected = {'race': ['Caucasian', 'African-American']}

# split train and test (70, 30)
np.random.seed(seed=1)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# classifier -- aggregate
train['attr'] =  train['crace']

# logistic regression
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
dct.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = dct.predict(np.array(test[feature_list]))

# auditing learner
feature_audit = ['age_cat', 'priors_count', 'juv_fel_count', 'is_violent_recid']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
print(score)

feature_audit = ['age_cat']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
print(score)
feature_audit = ['age_cat',  'priors_count']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
print(score)
feature_audit = ['age_cat',  'priors_count', 'juv_fel_count']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
print(score)



# reduction method
epsilon = 0.01
constraint = moments.EO()
trainX = train[feature_list]
trainY = train[outcome]
trainA = train['attr'] 
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
res_tuple = red.expgrad(trainX, trainA, trainY, dct,
							cons=constraint, eps=epsilon)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]
test['predict'] = np.array(best_classifier(np.array(test[feature_list])))
test.loc[test.predict < 0.5, 'predict'] = 0
test.loc[test.predict > 0.5, 'predict'] = 1

# auditing learner
feature_audit = ['age_cat', 'priors_count', 'juv_fel_count', 'is_violent_recid']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
#print(unfair_treatment[unfair_treatment.sex == 'Caucasian'][feature_audit + ['predict', outcome]].describe())
#print(unfair_treatment[unfair_treatment.race == 'African-American'][feature_audit + ['predict', outcome]].describe())
print(score)



