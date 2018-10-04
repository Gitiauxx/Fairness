import pandas as pd
import numpy as np
from fairndist_cluster import run_classifier
from fairlearn import moments
from test_fairlearn import run_fairlearn
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

feature_list = ['age_cat',  'priors_count', 'juv_fel_count', 'is_violent_recid', 'gender',
			 'crace']
for var in feature_list:
	data = data[~np.isnan(data[var])]

outcome = 'two_year_recid'
protected = {'race': ['Caucasian', 'African-American'], 'sex': ['Male', 'Female']}

# split train and test (70, 30)
np.random.seed(seed=4)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# classifier -- aggregate
train['attr'] =  train['crace']
cons = moments.EO()
epsilon = 0.02

results_agg = run_fairlearn(train, test, feature_list, outcome, protected, cons, epsilon, size=1)
results_agg['method'] = 'FR_AGG'

# classifier -- individual
size = 1
niter = 20
epsilon = 0

results_ind = run_classifier(train, test, feature_list, outcome, protected, niter, size, epsilon)
results_ind['method'] = 'FR_IND'

results = pd.concat([results_agg, results_ind])
results.to_csv('..\\results\\fairness_compas_10032018_cluster.csv')
