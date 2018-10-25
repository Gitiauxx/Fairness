import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import pickle
from sklearn.metrics import accuracy_score


def fit(train, features, outcome, learner):
    learner.fit(np.array(train[features]), np.array(train[outcome]).ravel())
    return learner

def predict(learner_list, test, features, outcome):
    df_list = []
    for learner in learner_list:
        clf = pickle.loads(learner)
        test["predict_%s" %(clf.name)],  test["confidence_%s" %(clf.name)]= clf.predict(test[features], return_std=True)
        df_list.append(test[["predict_%s" %(clf.name), "confidence_%s" %(clf.name)]])
    return pd.concat(df_list, axis=1)

def audit_tree(d, features, outcome, protected):
    
    kernel = DotProduct() + WhiteKernel()

    # create learners
    for varname, attribute_list in protected.items():
        # remove protected from features
        features_audit = [feat for feat in features if feat != varname]
        learner_list = []

        # split  (70/30)
        train = d.loc[np.random.choice(d.index, int(0.7 * len(d)), replace=False)]
        test = d.drop(train.index)

        for attribute in attribute_list:
            data = train[train[varname] == attribute]
            learner = GaussianProcessRegressor(kernel=kernel, random_state=0)
            #learner = DecisionTreeClassifier()
            learner.name = attribute
            learner = fit(data, features_audit, outcome, learner)
            
            learner_list.append(pickle.dumps(learner))
           
        score = np.inf
        for attribute in attribute_list:
            data = test[test[varname] == attribute]
            results = predict(learner_list, data, features_audit, outcome)
            
            #results_confusion = results[["predict_%s"%var for var in attribute_list]].mean(axis=1)
            results_confusion = np.array(results[["predict_%s"%var for var in attribute_list]])
            results_variance = np.ones(len(results_confusion))
            results_confusion[results_confusion < 0.5] = 0
            results_confusion[results_confusion >= 0.5] = 1
            print(accuracy_score(np.array(data[outcome]), results_confusion[:, 0]))
            print(accuracy_score(np.array(data[outcome]), results_confusion[:, 1]))
            results_difference = np.absolute(results_confusion[:, 0] - results_confusion[:, 1])

            for var in attribute_list:
                results_variance = results_variance * np.array(results["confidence_%s"%var])
           
            if (((results_difference / results_variance).sum()) * results_variance.sum()) < score:
                score =((results_difference / results_variance).sum()) * results_variance.sum()
        
        return score
