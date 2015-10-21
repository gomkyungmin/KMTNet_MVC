#!/usr/bin/python

import os
from os.path import exists,join,isdir
import argparse

import numpy as np
from sklearn.externals import joblib

import LoadData as ld
import Classifier as CLF
import Plot as plt
import MLA


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


def load_data(data_file,feature_file,class_cut):
    
    samples = ld.load_data(data_file,feature_file,class_cut)
    features = samples.features

    return samples, features

    
def data_generation(samples,training_part,training_size):

    X = samples.data
    y = samples.target

    train_samples = len(samples.data)*training_size

    if training_part == 'first':
        X_train = X[:train_samples]
        X_test = X[train_samples:]
        y_train = y[:train_samples]
        y_test = y[train_samples:]
    elif training_part == 'second':
        X_train = X[train_samples:]
        X_test = X[:train_samples]
        y_train = y[train_samples:]
        y_test = y[:train_samples]
    
    return X_train,y_train,X_test,y_test


def performance_test(ml,clf,features,**mla_args):

    classified_result_proba = ml.test(clf)
    
    figdir = 'figure'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    else:
        pass

    plt.draw_roc(ml.y_test,classified_result_proba,figdir,ml.tag)

    if mla_args['mla_selection'] == 'rf' and\
       (mla_args['estimators'] is True or\
       mla_args['classes'] is True or\
       mla_args['feature_importance'] is True):
        print("\n=== Diagnostic(s) of Trained RandomForest ===")
        CLF.diagnosing_trained_forest(clf,features,**mla_args)
    else:
        pass


def main():

    args = parse_command_line()

    data_file = args['data_file']
    feature_file = args['feature_file']
    class_cut =args['class_cut']
    training_size = args['training_size']
    training_part = args['training_part']

    mla_args = args.copy()
    del mla_args['data_file'], mla_args['feature_file'],\
        mla_args['class_cut'], mla_args['training_size'],\
        mla_args['training_part']
    
    samples, features = load_data(data_file,feature_file,class_cut)
    X_train, y_train, X_test, y_test\
        = data_generation(samples,training_part,training_size)

    pklfiledir = 'trained_pickle'
    if not os.path.isdir(pklfiledir):
        os.mkdir(pklfiledir)
    else:
        pass
    
    if args['pkl_file'] is not None:
        tag = args['pkl_file'].replace('trained_pickle/trained_%s' % mla,'')
        tag = tag.replace('.pkl','')
        ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)

        clf = joblib.load(args['pkl_file'])
        print("\nTrained forest is loaded from %s" % (args['pkl_file']))
        print clf
        
    elif mla_args['mla_selection'] == 'rf':
        tag = '_'+str(mla_args['mla_selection'])\
              +'_trees'+str(mla_args['n_estimators'])\
              +'_Mf'+str(mla_args['max_features'])\
              +'_Md'+str(mla_args['max_depth'])\
              +'_Mln'+str(mla_args['max_leaf_nodes'])\
              +'_mss'+str(mla_args['min_samples_split'])\
              +'_msl'+str(mla_args['min_samples_leaf'])\
              +'_'+str(mla_args['criterion'])\
              +'_'+str(training_part)+str(training_size)
    
        pklfilename = 'trained'+tag+'.pkl'
        pklfile = join(pklfiledir,pklfilename)

        if os.path.exists(pklfile):
            print("\nYou have a support vector machine trained w/ the same parameters!")

            clf = joblib.load(pklfile)
            print("Trained forest is loaded from %s" % (pklfile))
            print clf

            tag = pklfile.replace('trained_pickle/trained_%s' %\
                                  mla_args['mla_selection'],'')
            tag = tag.replace('.pkl','')
            ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)

        else:
            ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)
            clf = ml.train()

    elif mla_args['mla_selection'] == 'svm':
        tag = '_'+str(mla_args['mla_selection'])\
              +'_C'+str(mla_args['C'])\
              +'_k'+str(mla_args['kernel'])\
              +'_d'+str(mla_args['degree'])\
              +'_g'+str(mla_args['gamma'])\
              +'_c'+str(mla_args['coef0'])\
              +'_t'+str(mla_args['tol'])\
              +'_Mi'+str(mla_args['max_iter'])\
              +'_r'+str(mla_args['random_state'])\
              +'_'+str(training_part)+str(training_size)
    
        pklfilename = 'trained'+tag+'.pkl'
        pklfile = join(pklfiledir,pklfilename)

        if os.path.exists(pklfile):
            print("\nYou have a support vector machine trained w/ the same parameters!")

            clf = joblib.load(pklfile)
            print("Trained support vector machine is loaded from %s" % (pklfile))
            print clf

            tag = pklfile.replace('trained_pickle/trained_%s' %\
                                  mla_args['mla_selection'],'')
            tag = tag.replace('.pkl','')
            ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)

        else:
            ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)
            clf = ml.train()

    performance_test(ml,clf,features,**mla_args)


if __name__=='__main__':

    main()
