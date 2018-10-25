import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score
import copy

def create_confusion(test, protected, outcome):
    count = 1
    for varname, attribute_list in protected.items():
        for attribute in attribute_list:
            test.loc[test[varname] == attribute, 'conf_predict'] = count * test[outcome] + \
                            (1- count) * (1 - test[outcome])
            count -= 1
    #test['conf_predict'] = test.predict * test.conf_predict
    return test


def add_leakages(data, protected):
     # split  (70/30)
    train = d.loc[np.random.choice(d.index, int(0.7 * len(d)), replace=False)]
    test = d.drop(train.index)




def audit_tree(data, features, outcome, protected, seed=1):

    results = {}

    for varname, attribute_list in protected.items():

        # remove protected from features
        features_audit = [feat for feat in features if feat != varname]

        # confuse the outcome
        pro = {varname: attribute_list}
        d = create_confusion(data, pro, outcome)

        # split  (70/30)
        train = d.loc[np.random.choice(d.index, int(0.7 * len(d)), replace=False)]
        test = d.drop(train.index)
        
        # test and train data
        testX = np.array(test[features_audit])
        testY = np.array(test['conf_predict'].ravel())
        trainX = np.array(train[features_audit])
        trainY = np.array(train['conf_predict'].ravel())

        # fit confused prediction
        learner = LogisticRegression()
        learner = learner.fit(trainX, trainY)
        predicted = learner.predict(testX)
        results[varname] = accuracy_score(testY, predicted)

    return results, learner

def audit_tree_attr(data, features, outcome, protected, seed=1):
    data = copy.deepcopy(data)
    results = {}

    for varname, attribute_list in protected.items():

        # remove protected from features
        features_audit = [feat for feat in features if feat != varname]

        # confuse the outcome
        data = create_confusion(data, protected, outcome)

        # split  (70/30)
        train = data.loc[np.random.choice(data.index, int(0.7 * len(data)), replace=False)]
        test = data.drop(train.index)
        
        # test and train data
        testX = np.array(test[features_audit])
        testY = np.array(test[varname].astype('category').cat.codes.ravel())
        trainX = np.array(train[features_audit])
        trainY = np.array(train[varname].astype('category').cat.codes.ravel().ravel())

        # predict attr without labels
        learner = LogisticRegression()
        learner = learner.fit(trainX, trainY)
        predicted = learner.predict(testX)
        score1 = accuracy_score(testY, predicted)
        print(score1)

        # predict attr with labels
        features_audit.append(outcome)
        testX = np.array(test[features_audit])
        trainX = np.array(train[features_audit])
    
        learner = LogisticRegression()
        learner = learner.fit(trainX, trainY)
        predicted2 = learner.predict(testX)
        print(predicted2.mean())
        score2 = accuracy_score(testY, predicted2)
        print(score2)
        results[varname] = score2 - score1

        # identify unfair treatmenty
        test['pred1'] = predicted
        test['pred2'] = predicted2
       
        unfair_treatment = test[test.pred1 != test.pred2]

    return results, learner, unfair_treatment
    
    