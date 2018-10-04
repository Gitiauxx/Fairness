from fairlearn import classred as red
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def run_fairlearn(train, test, feature_list, outcome, protected, constraint, epsilon, size=1):

	# standardize data
	for col in feature_list:
		range_col = test[col].max() - test[col].min()
		test[col] = (test[col] - test[col].min()) / range_col
		
	# standardize data
	for col in feature_list:
		range_col = train[col].max() - train[col].min()
		train[col] = (train[col] - train[col].min()) / range_col
		
	# logsitic learner
	learner = LogisticRegression()
	
	
	# result on test data
	trainX = train[feature_list]
	trainY = train[outcome]
	trainA = train['attr']
	
	results = pd.DataFrame(index= np.arange(size))
	
	for i in np.arange(size):
		
		res_tuple = red.expgrad(trainX, trainA, trainY, learner,
                                cons=constraint, eps=epsilon)
		res = res_tuple._asdict()
		best_classifier = res["best_classifier"]
	
		predict = best_classifier(np.array(test[feature_list]))
		predict[predict > 0.5] = 1
		predict[predict <= 0.5] = 0
		test['predict'] = np.array(predict)
		
		
		# confusion matrix for protected attributes
		for varname in protected.keys():
			for var in protected[varname]:
				cm = confusion_matrix(np.array(test[test[varname] == var][outcome]), 
                               np.array(test[test[varname] == var].predict))
				cm = cm / cm.sum(axis=1)[:, np.newaxis]
				
				results.loc[i, 'tpr_%s'%var] = cm[0,0]
				results.loc[i, 'tnr_%s'%var] = cm[1, 1]
				results.loc[i, 'dp_%s'%var] = test[test[varname] == var].predict.mean()
				
		# accuracy
		results.loc[i, 'accuracy'] = len(test[test.predict == test[outcome]]) / len(test.predict)
		print(results) 
	
	return results
	
	