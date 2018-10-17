import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from fairlearn import moments
from fairlearn import classred as red
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

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

feature_list = ['age_cat', 'charge_degree', 'gender',
			 'crace']
for var in feature_list:
	data = data[~np.isnan(data[var])]
	data[var] = (data[var] - data[var].mean()) / data[var].var() ** 0.5

outcome = 'two_year_recid'
protected = {'race': ['Caucasian', 'African-American'], 'sex': ['Male', 'Female']}

# split train and test (70, 30)
np.random.seed(seed=3)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

train['attr'] =  train['crace']
cons = moments.EO()
epsilon = 0.2

trainX = train[feature_list]
trainY = train[outcome]
trainA = train['attr']

# gaussian mixture model -- clustering
batch_size = 500
mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
                      n_init=10, max_no_improvement=20, verbose=0, 
					  random_state=1, 
					  reassignment_ratio=0.1)
#gmm = GaussianMixture(n_components=80)

trainXA = np.array(trainX)
mbk.fit(trainXA)
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)

#gmm.fit(trainXA[:, :-2])
#train['cluster'] = gmm.predict(trainXA[:, :-2])
train['cluster'] = pairwise_distances_argmin(trainXA, mbk_means_cluster_centers)

#gmm.fit(trainXA)
#train['cluster2'] = gmm.predict(trainXA)

mbk = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
                      n_init=10, max_no_improvement=20, verbose=0, 
					  random_state=1,
					  reassignment_ratio=0.1)

mbk.fit(trainXA[:, :-2])
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
train['cluster2'] = pairwise_distances_argmin(trainXA[:, :-2], mbk_means_cluster_centers)

# bias detecting from fairlearn
learner = LogisticRegression()

res_tuple = red.expgrad(trainX, trainA, trainY, learner,
                                cons=cons, eps=epsilon)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]

predicted = best_classifier(trainX)
train['predicted'] = predicted

bias = train.groupby('cluster').predicted.mean()
bias  = bias * ( 1 - bias)
print(bias)

bias = train.groupby('cluster')[outcome].mean()
bias  = bias * ( 1 - bias)
print(bias)

bias = train.groupby('cluster2').predicted.mean()
bias  = bias * ( 1 - bias)
print(bias)

