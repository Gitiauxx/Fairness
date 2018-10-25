import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import audit_tree as ad
from test_fairlearn import run_fairlearn
from fairlearn import moments

pd.set_option('display.max_columns', 100)

# create fake data
N = 10000
np.random.seed(seed=1)
data = pd.DataFrame(index=np.arange(N))
data['protected'] = np.random.choice([0, 1], len(data))
data['x1'] = np.random.normal(size=len(data)) + 0.0 * data.protected
data['x2'] = np.random.normal(size=len(data))
data['x3'] = np.random.normal(size=len(data))
data['x2'] = -1. * data.protected + 0.5 *data.x3 
data['y'] =  data['x1'] + data['x3']
data['outcome'] = (data.y >= 0).astype('int32')


feature_list = ['x1']
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
dct.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = dct.predict(np.array(test[feature_list]))

# add some individual unfairness
for size in np.arange(10):
    ind = np.random.choice(test.index, int(size/10 * len(test)), replace=True)
    test_add = test.loc[ind]
    test_add['protected'] = 1- test_add['protected']
    test_add['predict'] = 1
    data = pd.concat([test_add, test], axis=0)

    data.set_index(np.arange(len(data)), inplace=True)


    # auditing confusion tree
    feature_audit = ['x1'] 
    score = ad.audit_tree(data, feature_audit, 'predict', protected)
    print(score)
#print(unfair_treatment[unfair_treatment.attr == '0'][feature_audit + ['predict']].describe())
#print(unfair_treatment[unfair_treatment.attr == '1'][feature_audit + ['predict']].describe())





