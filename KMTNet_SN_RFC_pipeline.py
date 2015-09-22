#!/usr/bin/python

import os,time
from os.path import dirname,join,exists,isdir
import argparse

import numpy as np
from sklearn.externals import joblib

import LoadData as ld
import RFClassifier as RFC
import Plot as plt


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
    #----- Training Parameters -----#
    parser.add_argument("--n-estimators",action="store",type=int,default=100,\
                        help="default=100")
    parser.add_argument("--criterion",action="store",type=str,default="gini",\
                        help="default=gini, available options: gini or entropy")
    parser.add_argument("--max-features",action="store",default="auto",\
                        help="default=auto, If int, then consider max_features features at each split. If float between 0. and 1., then max_features is a persentage and int(max_features * n_features) features are considered at each split")
    parser.add_argument("--max-depth",default=None,\
                        help="default=None, available values: integer or None")
    parser.add_argument("--max-leaf-nodes",default=None,\
                        help="default=None, available values: integer or None")
    parser.add_argument("--min-samples-split",default=2,type=int)
    parser.add_argument("--min-samples-leaf",default=1,type=int)
    parser.add_argument("--n-jobs",action="store",type=int,default=1)
    #----- Diagnostic -----#
    parser.add_argument("--estimators",default=False,action="store_true",help="list of DecisionTreeClassifier")
    parser.add_argument("--classes",default=False,action="store_true")
    parser.add_argument("--feature-importance",default=False,action="store_true")

    args = parser.parse_args()
    
    return args
    

def load_data():
    
    samples = ld.load_data(args.data_file,args.feature_file,args.class_cut)
    features = samples.features

    return samples, features

    
def data_generation(samples):

    X = samples.data
    y = samples.target

    train_samples = len(samples.data)*args.training_size

    if args.training_part == 'first':
        X_train = X[:train_samples]
        X_test = X[train_samples:]
        y_train = y[:train_samples]
        y_test = y[train_samples:]
    elif args.training_part == 'second':
        X_train = X[train_samples:]
        X_test = X[:train_samples]
        y_train = y[train_samples:]
        y_test = y[:train_samples]

    
    return X_train,y_train,X_test,y_test
    
    
if __name__=='__main__':

    args = parse_command_line()

    val_n_estimators = args.n_estimators
    val_criterion = args.criterion
    val_max_features = args.max_features
    val_max_depth = args.max_depth
    val_max_leaf_nodes = args.max_leaf_nodes
    val_min_samples_split = args.min_samples_split
    val_min_samples_leaf = args.min_samples_leaf
    val_n_jobs = args.n_jobs

    samples, features = load_data()
    X_train,y_train,X_test,y_test=data_generation(samples)

    pklfiledir = 'trained_forest'
    if not os.path.isdir(pklfiledir):
        os.mkdir(pklfiledir)
    else:
        pass

    tag = '_trees'+str(val_n_estimators)+'_Mf'+str(val_max_features)\
          +'_Md'+str(val_max_depth)+'_Mln'+str(val_max_leaf_nodes)+'_mss'\
          +str(val_min_samples_split)+'_msl'+str(val_min_samples_leaf)\
          +'_'+val_criterion+'_'+str(args.training_part)+str(args.training_size)
    
    pklfilename = 'trained_forest'+tag+'.pkl'
    pklfile = join(pklfiledir,pklfilename)

    if os.path.exists(pklfile):
        rfc = joblib.load(pklfile)
        print("\nTrained forest is loaded from %s" % (pklfile))
        print rfc

    else:
        rfc = RFC.training(X_train,y_train,\
                           val_n_estimators,\
                           val_criterion,\
                           val_max_features,\
                           val_max_depth,\
                           val_max_leaf_nodes,\
                           val_min_samples_split,\
                           val_min_samples_leaf,\
                           val_n_jobs,\
                           pklfile)

    classified_result_proba = RFC.classification(rfc,X_test)
    
    if args.estimators is True or args.classes is True or args.feature_importance is True:
        print("\n=== Diagnostic(s) of Trained RandomForest ===")
        RFC.diagnosing_trained_forest(rfc,\
                                      args.estimators,\
                                      args.classes,\
                                      args.feature_importance,\
                                      features)
    else:
        pass

    figdir = 'figure'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    else:
        pass
    
    plt.draw_roc(y_test, classified_result_proba, figdir, tag)
