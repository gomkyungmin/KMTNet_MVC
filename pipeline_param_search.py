#!/usr/bin/python

import argparse

import numpy as np

import LoadData as ld
import ParamSearch as ps
import MLA
import pipeline
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
    #parser.add_argument("--scaler",action="store",type=str,default="Standard")

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
    group_svm.add_argument("--param-search-file",action="store",default=None,type=str)
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


def main():

    args = parse_command_line()
    
    samples, features = load_data(args['data_file'],args['feature_file'],args['class_cut'])

    params = ps.read_param(args['param_search_file'])

    ps.find_optimal(params,samples,**args)
    

if __name__=='__main__':

    main()
    
