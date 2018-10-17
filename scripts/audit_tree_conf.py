import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

def create_confusion(test, protected, outcome):
    count = 1
    for varname, attribute_list in protected.items():
        for attribute in attribute_list:
            test.loc[test[varname] == attribute, 'conf_predict'] = count * test[outcome] + \
                            (1- count) * (1 - test[outcome])
            count -= 1

    return test


def fit(train, features, outcome, learner):
    learner.fit(np.array(train[features]), np.array(train[outcome]).ravel())
    return learner

def predict(learner_list, test, features, outcome):
    df_list = []
    for learner in learner_list:
        clf = pickle.loads(learner)
        test["predict_%s" %(clf.name)] = clf.predict(test[features])
        df_list.append(test[["predict_%s" %(clf.name)]])
    return pd.concat(df_list, axis=1)

def audit_tree(data, features, outcome, protected, seed=1):

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
        testY = np.array(test['conf_predict'].ravel())
        trainX = np.array(train[features_audit])
        trainY = np.array(train['conf_predict'].ravel())

        # fit confused prediction
        learner = DecisionTreeClassifier()
        learner = learner.fit(trainX, trainY)
        predicted = learner.predict(testX)
        results[varname] = accuracy_score(testY, predicted)

    return results
    