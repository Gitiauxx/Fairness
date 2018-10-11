import pandas as pd
import numpy as np
from test_fairlearn import run_fairlearn
from fairlearn import moments
from sklearn.linear_model import LogisticRegression
import bd_tree as bd
pd.set_option('display.max_columns', 100)

# create fake data
N = 8000
np.random.seed(seed=10)
data = pd.DataFrame(index=np.arange(N))
data['protected'] = np.random.choice([0, 1], len(data))
data['x1'] = np.random.normal(size=len(data))
data['x2'] = np.random.normal(size=len(data))
data['x2'] = (0.2 * np.random.normal(size=len(data)) + 1) * data.protected + data.x2
data['x3'] = np.random.normal(size=len(data))
data['x3'] = data.x2 * data.x3 + 2 * data.protected

data['y'] = 2*data.protected + data['x2'] + data['x3'] + data['x1']
#+  data['x2'] + data['protected'] + data['x3']
data['outcome'] = (data.y >= 0).astype('int32')


feature_list = ['x1',  'x2', 'x3']
outcome = 'outcome'
data['attr'] = '0'
data.loc[data.protected == 1, 'attr'] = '1'
protected = {'attr':['0', '1']}

# split train and test (70, 30)
np.random.seed(seed=4)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)), replace=False)]
test = data.drop(train.index)
print(len(train))
print(len(test[test.protected == 1]))
print(len(test[test.protected == 0]))
	
# reduction method
epsilon = 10
cons = moments.EO()
results_agg = run_fairlearn(train, test, feature_list, outcome, protected, cons, epsilon, size=1)
results_agg['method'] = 'FR_AGG'


