import os,time
from os.path import dirname,join,exists,isdir

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.externals import joblib

import Utils as utils


def train_rf(tag,X_train,y_train,**mla_args):

    #Create RandomForest classifier
    clf = RandomForestClassifier(n_estimators=mla_args['n_estimators'],\
                                 criterion=mla_args['criterion'],\
                                 max_features=mla_args['max_features'],\
                                 max_depth=mla_args['max_depth'],\
                                 max_leaf_nodes=mla_args['max_leaf_nodes'],\
                                 min_samples_split=mla_args['min_samples_split'],\
                                 min_samples_leaf=mla_args['min_samples_leaf'],\
                                 n_jobs=mla_args['n_jobs'])

    print("\nTraining forest...")
    TrainingStartTime = time.time()
    clf.fit(X_train,y_train)
    TrainingEndTime = time.time()
    print("Training is done! (elapsed time for training RandomForestClassifier w/ %d samples: %s)" %\
          (len(X_train),utils.Timer(TrainingEndTime-TrainingStartTime)))

    utils.PickleDump(tag,clf,mla_args['mla_selection'])
    
    return clf


def test_rf(clf,X_test):

    print("\nClassifying samples...")
    EvaluationStartTime = time.time()
    result = clf.predict_proba(X_test)
    EvaluationEndTime = time.time()
    print("Classification is done! (elapsed time for classifying %d samples: %s)" %\
          (len(X_test),utils.Timer(EvaluationEndTime-EvaluationStartTime)))

    return result
    
    
def classification_summary(clf,result,X_test,y_test):

    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_test[i] == result[i]:
            tp += 1.
        elif y_test[i] == 0 and y_test[i] == result[i]:
            tn += 1.
        elif y_test[i] == 1 and result[i] == 0:
            fn += 1.
        else:
            fp += 1.

    print("\nClassification Results:")
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp,tn,fp,fn))
    test_score = clf.score(X_test, y_test)
    print("Mean accuracy score on test data: %f" % test_score)
 

def diagnosing_trained_forest(clf,features,**mla_args):

    if mla_args['estimators'] is True:
        print("\nEstimators (Trees):")
        print(clf.estimators_)

    if mla_args['classes'] is True:
        print("\nClasses and Number of Classes:")
        print(clf.classes_)
        print(clf.n_classes_)

    if mla_args['feature_importance'] is True:
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature Ranking:")
        for f in range(len(features)):
            print("%d. %s (%f)" % (f+1, features[indices[f]], importances[indices[f]]))
            
            
def train_svm(tag,X_train,y_train,**mla_args):

    #Create SupportVectorMachines classifier
    clf = svm.SVC(C=mla_args['C'],\
                  kernel=mla_args['kernel'],\
                  degree=mla_args['degree'],\
                  gamma=mla_args['gamma'],\
                  coef0=mla_args['coef0'],\
                  probability=mla_args['probability'],\
                  tol=mla_args['tol'],\
                  max_iter=mla_args['max_iter'],\
                  random_state=mla_args['random_state'])

    print("\nTraining support vector machines...")
    TrainingStartTime = time.time()
    clf.fit(X_train,y_train)
    TrainingEndTime = time.time()
    print("Training is done! (elapsed time for training SupportVectorMachinesClassifier w/ %d samples: %s)" %\
          (len(X_train),utils.Timer(TrainingEndTime-TrainingStartTime)))

    utils.PickleDump(tag,clf,mla_args['mla_selection'])
    
    return clf


def test_svm(clf,X_test):

    print("\nClassifying samples...")
    EvaluationStartTime = time.time()
    result = clf.predict_proba(X_test)
    EvaluationEndTime = time.time()
    print("Classification is done! (elapsed time for classifying %d samples: %s)" %\
          (len(X_test),utils.Timer(EvaluationEndTime-EvaluationStartTime)))

    return result
