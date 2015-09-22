import os,time
from os.path import dirname,join,exists,isdir

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def training(X_train,y_train,\
             val_n_estimators,\
             val_criterion,\
             val_max_features,\
             val_max_depth,\
             val_max_leaf_nodes,\
             val_min_samples_split,\
             val_min_samples_leaf,\
             val_n_jobs,\
             pklfile):

    #Create RandomForest classifier
    rfc = RandomForestClassifier(n_estimators=val_n_estimators,\
                                 criterion=val_criterion,\
                                 max_features=val_max_features,\
                                 max_depth=val_max_depth,\
                                 max_leaf_nodes=val_max_leaf_nodes,\
                                 min_samples_split=val_min_samples_split,\
                                 min_samples_leaf=val_min_samples_leaf,\
                                 n_jobs=val_n_jobs)

    print("\nTraining forest...")
    TrainingStartTime = time.time()
    rfc.fit(X_train,y_train)
    TrainingEndTime = time.time()
    print("Training is done! (elapsed time for training RandomForestClassifier w/ %d samples: %f sec)" %\
          (len(X_train),TrainingEndTime-TrainingStartTime))

    joblib.dump(rfc, pklfile)
    print("Trained forest is saved in %s" % pklfile)

    return rfc


def classification(rfc,X_test):

    print("\nClassifying samples...")
    EvaluationStartTime = time.time()
    result = rfc.predict_proba(X_test)
    EvaluationEndTime = time.time()
    print("Classification is done! (elapsed time for classifying %d samples: %f sec)" %\
          (len(X_test),EvaluationEndTime-EvaluationStartTime))

    return result


def classification_summary(rfc,result,X_test,y_test):

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
    test_score = rfc.score(X_test, y_test)
    print("Mean accuracy score on test data: %f" % test_score)
    
    
def diagnosing_trained_forest(rfc,estimators,classes,feature_importance,features):

    if estimators is True:
        print("\nEstimators (Trees):")
        print(rfc.estimators_)

    if classes is True:
        print("\nClasses and Number of Classes:")
        print(rfc.classes_)
        print(rfc.n_classes_)

    if feature_importance is True:
        importances = rfc.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nFeature Ranking:")
        for f in range(len(features)):
            print("%d. %s (%f)" % (f+1, features[indices[f]], importances[indices[f]]))
