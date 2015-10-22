import os,time
from os.path import dirname,join,exists,isdir

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.externals import joblib

import Utils as utils


class MLA:

    def __init__(self, tag, dataset, **kwargs):
        self.tag = tag
        self.X_train = dataset.X_train
        self.y_train = dataset.y_train
        self.X_test = dataset.X_test
        self.y_test = dataset.y_test
        self.kwargs = kwargs

        
    def train(self):

        self.clf = Train(self.X_train,self.y_train,**self.kwargs)

        return self.clf

    
    def test(self, clf):
        
        self.result = Test(clf,self.X_test)

        return self.result


def InitClassifier(**kwargs):

    if 'param_search_file' in kwargs:
        mla_args = utils.TrimArgs_ps(**kwargs)
    else:
        mla_args = utils.TrimArgs(**kwargs)

    if kwargs['mla_selection'] == 'rf':
        clf = RandomForestClassifier(**mla_args)
    elif kwargs['mla_selection'] == 'svm':
        clf = svm.SVC(**mla_args)

    return clf


def Train(X_train,y_train,**kwargs):

    clf = InitClassifier(**kwargs)
    TrainingStartTime = time.time()

    if kwargs['mla_selection'] == 'rf':
        print("\nTraining random forest w/ training samples...")
    elif kwargs['mla_selection'] == 'svm':
        print("\nTraining support vector machine w/ training samples...")

    clf.fit(X_train,y_train)

    TrainingEndTime = time.time()

    if kwargs['mla_selection'] == 'rf':
        print("Training is done! (elapsed time for training random forest w/ %d samples: %s)" %\
              (len(X_train),utils.Timer(TrainingEndTime-TrainingStartTime)))
    elif kwargs['mla_selection'] == 'svm':
        print("Training is done! (elapsed time for training support vector machine w/ %d samples: %s)" %\
              (len(X_train),utils.Timer(TrainingEndTime-TrainingStartTime)))

    tag = utils.Tag(**kwargs)
    utils.PickleDump(tag,clf,kwargs['mla_selection'])

    return clf


def Test(clf,X_test):

    EvaluationStartTime = time.time()
    
    print("\nEvaluating test samples...")

    result = clf.predict_proba(X_test)

    EvaluationEndTime = time.time()
    
    print("Evaluation is done! (elapsed time for evaluating %d test samples: %s)" %\
          (len(X_test),utils.Timer(EvaluationEndTime-EvaluationStartTime)))

    return result
        

def DiagnosingTrainedForest(clf,features,**args):

    if args['estimators'] is True:
        print("\nEstimators (Trees):")
        print(clf.estimators_)

    if args['classes'] is True:
        print("\nClasses and Number of Classes:")
        print(clf.classes_)
        print(clf.n_classes_)

    if args['feature_importance'] is True:
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Ranking:")
        for f in range(len(features)):
            print("%d. %s (%f)" % (f+1, features[indices[f]], importances[indices[f]]))
