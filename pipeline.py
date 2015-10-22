#!/usr/bin/python

import os
from os.path import exists,join,isdir
import argparse

import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

import LoadData as ld
import Classifier as CLF
import Plot as plt
import MLA
import Utils as utils


def parse_command_line():

    usage = """for more details, visit http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    
    """
    
    parser = argparse.ArgumentParser(description=usage)

    #----- Load Data -----#
    parser.add_argument("--data-file",action="store",type=str)
    parser.add_argument("--feature-file",action="store",type=str)
    parser.add_argument("--class-cut",action="store",type=float)
    parser.add_argument("--training-part",action="store",type=str,default="first")
    parser.add_argument("--training-size",action="store",type=float,default=0.5)
    parser.add_argument("--scaler",action="store",type=str,default=None)
    
    subparsers = parser.add_subparsers(dest='mla_selection')
    
    #----- Training Parameters for RF -----#
    group_rf = subparsers.add_parser("rf")
    group_rf.add_argument("--pkl-file",action="store",default=None,type=str)
    group_rf.add_argument("--n-estimators",action="store",type=int,default=100,\
                          help="default=100")
    group_rf.add_argument("--criterion",action="store",type=str,default="gini",\
                          help="default=gini, available options: gini or entropy")
    group_rf.add_argument("--max-features",action="store",default="auto",\
                          help="default=auto, If int, then consider max_features features at each split. If float between 0. and 1., then max_features is a persentage and int(max_features * n_features) features are considered at each split")
    group_rf.add_argument("--max-depth",default=None,\
                          help="default=None, available values: integer or None")
    group_rf.add_argument("--max-leaf-nodes",default=None,\
                          help="default=None, available values: integer or None")
    group_rf.add_argument("--min-samples-split",default=2,type=int)
    group_rf.add_argument("--min-samples-leaf",default=1,type=int)
    group_rf.add_argument("--n-jobs",action="store",type=int,default=1)
    #----- Diagnostic for RF -----#
    group_rf.add_argument("--estimators",default=False,action="store_true",help="list of DecisionTreeClassifier")
    group_rf.add_argument("--classes",default=False,action="store_true")
    group_rf.add_argument("--feature-importance",default=False,action="store_true")

    #----- Training Parameters for SVM -----#
    group_svm = subparsers.add_parser("svm")
    group_svm.add_argument("--pkl-file",action="store",default=None,type=str)
    group_svm.add_argument("--C",action="store",default=1.,type=float)
    group_svm.add_argument("--kernel",action="store",default="rbf",type=str,\
                           help="Options: linear, poly, rbf, sigmoid, precomputed")
    group_svm.add_argument("--degree",action="store",default=3,type=int,\
                           help="Degree of the polynomial kernel function. Ignored by all other kernels.")
    group_svm.add_argument("--gamma",action="store",default=0.,type=float,\
                           help="Kernel coefficient for rbf, poly and sigmoid. If gamma is 0. then 1/n_features will be used instead.")
    group_svm.add_argument("--coef0",action="store",default=0.,type=float,\
                           help="Independent term in kernel function. It is only significant in poly and sigmoid.")
    group_svm.add_argument("--probability",action="store_true",default=True)
    group_svm.add_argument("--tol",action="store",default=1e-3,type=float)
    group_svm.add_argument("--max-iter",action="store",default=10000,type=int,\
                           help="Hard limit on iterations within solver, or -1 for no limit.")
    group_svm.add_argument("--random-state",action="store",default=None)
    
    args = vars(parser.parse_args())

    return args


def training_go_or_stop(dataset,pklfiledir,**args):

    tag = utils.Tag(**args)
    pklfilename = 'trained'+tag+'.pkl'
    pklfile = join(pklfiledir,pklfilename)

    if args['mla_selection'] == 'rf':
        mla_name = 'RF'
    elif args['mla_selection'] == 'svm':
        mla_name = 'SVM'
        
    if os.path.exists(pklfile):
        print "\nYou already have a trained result which is trained with the same parameters!"
        answer = utils.QueryYesNo("Do you want to train %s again?" % mla_name)
        if answer == True:
            ml = MLA.MLA(tag, dataset, **args)
            clf = ml.train()
        elif answer == False:
            ml = MLA.MLA(tag, dataset, **args)
            clf = joblib.load(pklfile)
            print "Trained result is loaded from %s" % (pklfile)
            print clf
    else:
        ml = MLA.MLA(tag, dataset, **args)
        clf = ml.train()

    return ml,clf


def evaluation(ml,clf):

    classified_result_proba = ml.test(clf)
    
    return classified_result_proba


def performance_test(ml,clf,result,features,**args):

    plt.draw_roc(ml.y_test,result,ml.tag)

    if args['mla_selection'] == 'rf' and\
       (args['estimators'] is True or\
       args['classes'] is True or\
       args['feature_importance'] is True):
        print("\n=== Diagnostic(s) of Trained RandomForest ===")
        MLA.DiagnosingTrainedForest(clf,features,**args)
    else:
        pass

    
def evaluation_go_or_stop(ml,clf,samples,**args):

    figfile = 'figure/plot_roc_lin'+ml.tag+'.png'
    
    if os.path.exists(figfile):
        print "\nYou already have results of the performance test. "
        answer = utils.QueryYesNo("Do you want to update previous results?")
        if answer == True:
            result = evaluation(ml,clf)
            performance_test(ml, clf, result, samples.features, **args)
        elif answer == False:
            pass
    else:
        result = evaluation(ml,clf)
        performance_test(ml, clf, result, samples.features, **args)

        
def main():

    args = parse_command_line()
    
    samples = ld.LoadData(**args)
    dataset = ld.DataGeneration(samples,**args)    

    # Creation of a directory for trained pickle file
    pklfiledir = 'trained_pickle'
    if not os.path.isdir(pklfiledir):
        os.mkdir(pklfiledir)
    else:
        pass
    
    if args['pkl_file'] is not None:
        tag = args['pkl_file'].replace('%s/trained' % pklfiledir,'').replace('.pkl','')
        ml = MLA.MLA(tag, dataset, **args)
        clf = joblib.load(args['pkl_file'])
        print "\nTrained forest is loaded from %s" % (args['pkl_file'])
        print clf        
    else:
        ml,clf = training_go_or_stop(dataset,pklfiledir,**args)

    evaluation_go_or_stop(ml,clf,samples,**args)


if __name__=='__main__':

    main()
