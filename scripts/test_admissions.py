import pandas as pd
import numpy as np
from test_fairndist_bis import run_classifier
from fairlearn import moments
from test_fairlearn import run_fairlearn
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


feature_list = [ 'ugpa', 'cluster', 'gender', 'lsat', 'fam_inc', 'race']
outcome = 'pass_bar'
protected = {'race1': ['white', 'black', 'hisp']}

# split train and test (70, 30)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# classifier -- aggregate
train['attr'] =  train['race']
cons = moments.EO()
epsilon = 0.02

results_agg = run_fairlearn(train, test, feature_list, outcome, protected, cons, epsilon, size=1)
results_agg['method'] = 'FR_AGG'

# classifier -- individual
size = 1
niter = 5
epsilon = 0

results_ind = run_classifier(train, test, feature_list, outcome, protected, niter, size, epsilon)
results_ind['method'] = 'FR_IND'

results = pd.concat([results_agg, results_ind])
results.to_csv('..\\results\\fairness_admissions_10032018.csv')