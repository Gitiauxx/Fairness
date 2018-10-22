import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import audit_tree_conf as ad
from test_fairlearn import run_fairlearn
from fairlearn import moments

pd.set_option('display.max_columns', 100)

# create fake data
N = 400000
np.random.seed(seed=5)
data = pd.DataFrame(index=np.arange(N))
data['protected'] = np.random.choice([0, 1], len(data))
data['x1'] = np.random.normal(size=len(data)) + data.protected
data['x2'] = np.random.normal(size=len(data))
data['x3'] = np.random.normal(size=len(data))
data['x2'] = 0.5 * data.protected + 0.5 *data.x3 
data['y'] =  data['x1'] + data['x2'] - 0.5 * data['protected']
data['outcome'] = (data.y >= 0).astype('int32')


feature_list = ['x1', 'x2', 'protected']
outcome = 'outcome'
data['attr'] = '0'
data.loc[data.protected == 1, 'attr'] = '1'
protected = {'attr':['0', '1']}

# split train and test (70, 30)
np.random.seed(seed=5)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)), replace=False)]
test = data.drop(train.index)

# classifier using a logisitic model
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
logreg.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = logreg.predict_proba(np.array(test[feature_list]))[:, 0]
train['predict'] = logreg.predict_proba(np.array(train[feature_list]))[:, 0]

print(len(test[test['predict'] == test[outcome]]) / len(test))

# auditing confusion tree
feature_audit = [ 'x1'] 
score, learner, unfair_treatment = ad.audit_tree_attr(test, feature_audit, 'predict', protected)
print(score)
print(unfair_treatment[unfair_treatment.attr == '0'][feature_audit + ['predict']].describe())
print(unfair_treatment[unfair_treatment.attr == '1'][feature_audit + ['predict']].describe())




