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
data = pd.read_csv('..\\data\\admissions_bar.csv')
data['lsat'] = data.lsat.apply(lambda x: x.replace(' ', ''))
data['ugpa'] = data.ugpa.apply(lambda x: x.replace(' ', ''))
data['race'] = data.race.apply(lambda x: x.replace(' ', ''))
data['fam_inc'] = data.fam_inc.apply(lambda x: x.replace(' ', ''))
data = data[data.lsat != '']
data = data[data.ugpa != '']
data = data[data.race != '']
data = data[data.fam_inc != '']
data.loc[data.pass_bar == ' ', 'pass_bar'] = '0'
data = data[data.pass_bar.isin(['1', '0'])]
data = data[data.gender.isin(['male', 'female'])]

print(list(set(data.fam_inc)))


# create categorical data for age_cat, sex, race and charge degree
data['lsat'] = data.lsat.astype(float)
data['ugpa'] = data.ugpa.astype(float)
data['fam_inc'] = data.fam_inc.astype(float)
data['gender'] = data.sex.astype('category').cat.codes
data['race'] = data.race.astype('category').cat.codes
data['dropout'] = data.dropout.astype('category').cat.codes
data['cluster'] = data.cluster.astype('category').cat.codes
data['pass_bar'] = data.pass_bar.astype('category').cat.codes

data = data[data.race1.isin(['white', 'black'])]
data['attr'] =  data['race']

feature_list = [ 'ugpa', 'cluster', 'gender', 'lsat', 'fam_inc', 'race']
for v in list(data.columns):
    data = data[~data[v].isnull()]
outcome = 'pass_bar'
protected = {'race1': ['white', 'black']}

# split train and test (70, 30)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# logistic regression
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
dct.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = dct.predict(np.array(test[feature_list]))

# auditing learner
feature_audit = [ 'ugpa', 'cluster', 'lsat', 'fam_inc']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
#print(unfair_treatment[unfair_treatment.race == 'Caucasian'][feature_audit + ['predict', outcome]].describe())
#print(unfair_treatment[unfair_treatment.race == 'African-American'][feature_audit + ['predict', outcome]].describe())
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
feature_audit = [ 'ugpa', 'cluster', 'lsat', 'fam_inc']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
#print(unfair_treatment[unfair_treatment.sex == 'Caucasian'][feature_audit + ['predict', outcome]].describe())
#print(unfair_treatment[unfair_treatment.race == 'African-American'][feature_audit + ['predict', outcome]].describe())
print(score)