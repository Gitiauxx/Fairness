import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score


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

def audit_tree(train, test, features, outcome, protected):
    
    # create learners
    for varname, attribute_list in protected.items():
        # remove protected from features
        features_audit = [feat for feat in features if feat != varname]
        learner_list = []

        for attribute in attribute_list:
            data = train[train[varname] == attribute]
            learner = DecisionTreeClassifier()
            learner.name = attribute
            learner = fit(data, features_audit, outcome, learner)
            print(accuracy_score(np.array(test[outcome]).ravel(), 
                                learner.predict(np.array(test[features_audit]))))
            learner_list.append(pickle.dumps(learner))
        
        
        score = 0
        for attribute in attribute_list:
            data = test[test[varname] == attribute]
            results = predict(learner_list, data, features_audit, outcome)
            
            results_confusion = results.mean(axis=1)
            score += ((results_confusion !=0) & (results_confusion !=1)).astype('int32').sum()
        
        return score
