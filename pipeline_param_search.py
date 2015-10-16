#!/usr/bin/python

import argparse

import numpy as np

import LoadData as ld
import ParamSearch as ps
import MLA
import pipeline


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
    
    #----- Parameter Space Search for RF -----#
    group_rf = subparsers.add_parser("rf")
    # group_rf.add_argument("--param-search",default=False,action="store_true",\
    #                     help="switch for searching optimal training parameters")
    group_rf.add_argument("--param-search-file",action="store",type=str)
    #----- Training Parameters -----#
    group_rf.add_argument("--max-leaf-nodes",default=None,\
                        help="default=None, available values: integer or None")
    group_rf.add_argument("--min-samples-leaf",default=1,type=int)
    group_rf.add_argument("--n-jobs",action="store",type=int,default=1)
    #----- Diagnostic -----#
    group_rf.add_argument("--estimators",default=False,action="store_true",\
                        help="list of DecisionTreeClassifier")
    group_rf.add_argument("--classes",default=False,action="store_true")
    group_rf.add_argument("--feature-importance",default=False,action="store_true")

    #----- Parameter Space Search for SVM -----#
    group_svm = subparsers.add_parser("svm")
    group_svm.add_argument("--param-search-file",aciton="store",default=None,type=str)
    
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


if __name__=='__main__':

    args = parse_command_line()

    data_file = args['data_file']
    feature_file = args['feature_file']
    class_cut = args['class_cut']
    training_size = args['training_size']
    training_part = args['training_part']

    mla_args = args.copy()
    del mla_args['data_file'], mla_args['feature_file'],\
        mla_args['class_cut'], mla_args['training_size'],\
        mla_args['training_part']
    
    samples, features = load_data(data_file,feature_file,class_cut)
    X_train, y_train, X_test, y_test = data_generation(samples,training_part,training_size)

    params = ps.read_param(args['param_search_file'])

    if mla_args['mla_selection'] == 'rf':
        best_combination = ps.find_optimal(params,X_train,y_train,X_test,y_test)

        mla_args['n_estimators'] = best_combination[0]
        mla_args['criterion'] = best_combination[1]
        mla_args['max_features'] = best_combination[2]
        mla_args['min_samples_split'] = best_combination[3]
        mla_args['max_depth'] = best_combination[4]

        tag = '_'+str(mla)\
              +'_C'+str(mla_args['C'])\
              +'_k'+str(mla_args['kernel'])\
              +'_d'+str(mla_args['degree'])\
              +'_g'+str(mla_args['gamma'])\
              +'_c'+str(mla_args['coef0'])\
              +'_t'+str(mla_args['tol'])\
              +'_Mi'+str(mla_args['max_iter'])\
              +'_r'+str(mla_args['random_state'])\
              +'_'+str(training_part)+str(training_size)

        ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)
        clf = ml.train()
        
    if mla_args['mla_selection'] == 'svm':
        best_combination = ps.find_optimal(params,X_train,y_train,X_test,y_test)

        # mla_args['...'] = best_combination[0]
        # mla_args['...'] = best_combination[1]
        # mla_args['...'] = best_combination[2]
        # mla_args['...'] = best_combination[3]
        # mla_args['...'] = best_combination[4]

        tag = '_'+str(mla)\
              +'_C'+str(mla_args['C'])\
              +'_k'+str(mla_args['kernel'])\
              +'_d'+str(mla_args['degree'])\
              +'_g'+str(mla_args['gamma'])\
              +'_c'+str(mla_args['coef0'])\
              +'_t'+str(mla_args['tol'])\
              +'_Mi'+str(mla_args['max_iter'])\
              +'_r'+str(mla_args['random_state'])\
              +'_'+str(training_part)+str(training_size)

        ml = MLA.MLA(tag,X_train,y_train,X_test,y_test,**mla_args)
        clf = ml.train()
        
    pipeline.performance_test(ml,clf,features,**mla_args)
