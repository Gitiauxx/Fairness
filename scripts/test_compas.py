import pandas as pd
import numpy as np
from test_fairndist import run_classifier


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

feature_list = ['age_cat', 'charge_degree', 'is_violent_recid', 'juv_fel_count', 
			'priors_count', 'crace', 'gender']
outcome = 'two_year_recid'
protected = {'race': ['Caucasian', 'African-American']}

# split train and test (70, 30)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# classifier
size = 1
niter = 20
epsilon = 0

results = run_classifier(train, test, feature_list, outcome, protected, niter, size, epsilon)

