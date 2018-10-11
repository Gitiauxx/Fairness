from fairlearn import classred as red
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import bd_tree_alt2 as bd


def run_fairlearn(train, test, feature_list, outcome, protected, constraint, epsilon, size=1):

	# standardize data
	#for col in feature_list:
		#range_col = test[col].max() - test[col].min()
		#test[col] = (test[col] - test[col].min()) / range_col
		
	# standardize data
	#for col in feature_list:
		#range_col = train[col].max() - train[col].min()
		#train[col] = (train[col] - train[col].min()) / range_col
	
	
	# result on test data
	trainX = train[feature_list]
	trainY = train[outcome]
	trainA = train['attr']
	
	results = pd.DataFrame(index= np.arange(size))
	
	for i in np.arange(size):
		
		# logsitic learner
		learner = LogisticRegression()
		
		res_tuple = red.expgrad(trainX, trainA, trainY, learner,
                                cons=constraint, eps=epsilon)
		res = res_tuple._asdict()
		best_classifier = res["best_classifier"]
		#learner.fit(trainX, trainY.ravel())
	
		predict = best_classifier(np.array(test[feature_list]))
		#predict = learner.predict(np.array(test[feature_list]))
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
		
		
		#construction of fairness tree
		for varname in protected.keys():
			var_list = protected[varname]
			test_left = test[test[varname] == var_list[0]][feature_list + ['predict', varname]]
			test_right = test[test[varname] == var_list[1]][feature_list + ['predict', varname]]
			fairness_tree = bd.build_tree(test_left, test_right, 20, 1, 'predict', varname)
			
			
			smax = bd.get_scores(fairness_tree[0], 1, 1)
			results.loc[i, 'score_%s' %varname] = smax
			
			#left = bd.get_group(fairness_tree[0], smax)
			#right = bd.get_group(fairness_tree[1], smax)
			
			
			#test_left['score_%s' %varname] = test_left[feature_list].apply(lambda row: bd.predict_score(fairness_tree[0], row), axis=1)
			#test_right['score_%s' %varname] = test_right[feature_list].apply(lambda row: bd.predict_score(fairness_tree[0], row), axis=1)
			
			#left = bd.get_group(fairness_tree[0])
			#right = bd.get_group(fairness_tree[1])
			
			#print(test_left.loc[[i for sublist in left for i in sublist]].describe())
			#print(test_right.loc[[i for sublist in right for i in sublist]].describe())
			
	
	print(results) 
	
	return results
	
	
