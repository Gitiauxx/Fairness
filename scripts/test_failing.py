import pandas as pd
from test_fairndist import run_classifier
import numpy as np


data = pd.read_csv('..\\data\\preprocessed_students_fl.csv')
data['iid'] = data['iid'] - data['iid'].min()
		
# remove nan
data = data[~np.isnan(data.grade_previous)]

# add gender and race data
data['gender'] = data.SEX.astype('category').cat.codes
data['srace'] = data.race.astype('category').cat.codes
data.loc[data.SEX == 'M', 'sex'] = ' Male'
data.loc[data.SEX == 'F', 'sex'] = ' Female'
		
# put in the test set the last term for each student
data['last_term'] = data.groupby('sid').termnum.transform("max")
data['first_term'] = data.groupby('sid').termnum.transform("min")
data['nterm'] = data['last_term'] - data['first_term'] + 1
		
# test set 
test = data[data.termnum >= data.last_term]
train = data[data.termnum < data.last_term]
 

features_list = ['has_failed', 'in_stem', 'grade_previous', 'instructor_avg',
				'gender', 'srace']
outcome = 'failing'
				
# classifier
size = 1
niter = 20
epsilon = 0

results = run_classifier(train, test, features_list, outcome, niter, size, epsilon)
results.to_csv('..\\results\\fairness_09262018_failing.csv')