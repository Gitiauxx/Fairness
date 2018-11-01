import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import audit_tree_conf as ad
from test_fairlearn import run_fairlearn
from fairlearn import moments
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)

# create fake data
N = 400000
np.random.seed(seed=1)
data = pd.DataFrame(index=np.arange(N))
data['protected'] = np.random.choice([0, 1], len(data))
data['x1'] = np.random.normal(size=len(data)) + 0.25 * data.protected
data['x2'] = np.random.normal(size=len(data))
data['x3'] = np.random.normal(size=len(data), scale=0.5)
data['x2'] = - 0.25 * data.protected + data.x2 
data['y'] =  data['x1'] + data['x2'] + data['x3']
data['outcome'] = (data.y >= 0).astype('int32')
print(data.groupby('protected').outcome.size())

feature_list = ['x1', 'x2']
outcome = 'outcome'
data['attr'] = '0'
data.loc[data.protected == 1, 'attr'] = '1'
protected = {'attr':['0', '1']}

# split train and test (70, 30)
np.random.seed(seed=3)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)), replace=False)]
test = data.drop(train.index)

# classifier using a logisitic model
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
logreg.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
test['predict'] = logreg.predict(np.array(test[feature_list]))

# add some individual unfairness
results = pd.DataFrame()
for size in np.arange(10):
    np.random.seed(seed=10)
    ind = np.random.choice(test.index, int(size/10 * len(test)), replace=False)
    test_add = test.copy()
    #test_add['protected'] = 1- test_add['protected']
    placebo = test_add.copy()
    test_add = test_add.loc[ind]
    test_add = test_add[test_add.protected ==0]
    test_add['predict'] = 1- test_add['predict']
    data = pd.concat([test_add, test])

    #test_add['protected'] = 1- test_add['protected']
    #test_add['r'] = np.random.choice([0, 1], len(test_add))
    #test_add[ 'predict'] = test_add[ 'r'] * (1 - test_add['predict']) + \
               #          (1-test_add['r']) * test_add['predict']
    #test_add.drop('r', axis=1, inplace=True)
    #data = pd.concat([test_add, test], axis=0)
   

    data.set_index(np.arange(len(data)), inplace=True)
    #placebo.set_index(np.arange(len(placebo)), inplace=True)

    # auditing confusion tree
    feature_audit = ['x1'] 
    score, _ = ad.audit_tree(data, feature_audit, 'predict', protected, seed=1)
    results.loc[size/10, 'Delta_Unfair'] = score['attr']
    score, _ = ad.audit_tree(placebo, feature_audit, 'predict', protected, seed=1)
    results.loc[size/10, 'Delta_Fair'] = score['attr']
results.index.name = 'gamma'
print(results)
results.to_csv("..\\results\\synth_exp2_dct.csv")

plt.plot()

#print(unfair_treatment[unfair_treatment.attr == '0'][feature_audit + ['predict']].describe())
#print(unfair_treatment[unfair_treatment.attr == '1'][feature_audit + ['predict']].describe())





