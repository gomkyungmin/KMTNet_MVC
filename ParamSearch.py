import time
import numpy as np
import itertools
from StringIO import StringIO

from sklearn.metrics import roc_curve,auc
from sklearn.externals import joblib

import Utils as utils
import MLA
import Plot as plt
import LoadData as ld


class ParamSearch:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.combinations = read_param(kwargs['param_search_file'])
        self.samples = ld.LoadData(**kwargs)
        
        if kwargs['parallel'] is False:
            find_optimal(self.combinations,self.samples,**self.kwargs)
        elif kwargs['parallel'] is True:
            find_optimal_parallel(self.combinations,self.samples,**self.kwargs)
            
        

def read_param(param_file):

    params_base = np.genfromtxt(param_file, delimiter='', dtype=None)
    params_label = params_base[:,0]
    params_val = params_base[:,1]

    params_val_sub = []
    for ele in params_val:
        d = np.genfromtxt(StringIO(ele), delimiter=',', dtype=None)
        params_val_sub.append(d.tolist())

    combinations = list(itertools.product(*params_val_sub))

    return combinations


def save_optimal_combination(aucs,combinations,clf_name):

    indices = np.argsort(aucs)[::-1]
    ordered_auc = []
    ordered_comb = []
    
    for i in range(len(aucs)):
        ordered_auc.append(aucs[indices[i]])
        ordered_comb.append(combinations[indices[i]])
    
    arr_comb = np.array(ordered_comb)
    arr_auc = np.array(ordered_auc)
    arr_out = np.append(arr_comb,\
                        arr_auc.reshape(len(arr_auc),1),1)

    rank_file = 'combination_ranking_%s.txt' % str(clf_name)
    if clf_name == 'rf':
        header_label = 'scaler n_estimator criterion max_features min_samples_split max_depth auc'
    elif clf_name == 'svm':
        header_label = 'scaler C auc'
    footer_label = 'This result is recorded at '+utils.WhatTimeIsItNow()
    np.savetxt(rank_file,arr_out,header=header_label,footer=footer_label,fmt="%s")
    print("\nRank of combinations of parameters saved in %s" % rank_file)


def init_classifier(combination,**args):

    if args['mla_selection'] == 'rf':

        if combination[0] == 'None':
            combination[0] = None
        if combination[5] == 'None':
            combination[5] = None

        args['n_estimators'] = combination[1]
        args['criterion'] = combination[2]
        args['max_features'] = combination[3]
        args['min_samples_split'] = combination[4]
        args['max_depth'] = combination[5]
        
    if args['mla_selection'] == 'svm':

        args['C'] = combination[1]

    clf = MLA.InitClassifier(**args)
    
    return clf

    
def find_optimal(combinations,samples,**args):

    aucs = []
    
    best_auc = 0.
    best_combination = []
    best_clf = 0.
    best_result = 0.
    
    SearchStartTime = time.time()
    print "\nSearching optimal training parameters (based on auc value) for %d combinations..."\
        % (len(combinations))
    
    for c in combinations:
        c = list(c)

        clf = init_classifier(c,**args)

        args['scaler'] = c[0]
        dataset = ld.DataGeneration(samples,**args)
        print c,
        clf.fit(dataset.X_train,dataset.y_train)
        result_proba = clf.predict_proba(dataset.X_test)

        fpr, tpr, threshold = roc_curve(dataset.y_test, result_proba[:,1])
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        print roc_auc
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_combination = c
            best_clf = clf
            best_result = result_proba
        else:
            best_auc = best_auc
            best_combination = best_combination
            best_clf = best_clf
            best_result = best_result

        del args['scaler']

    if args['mla_selection'] == 'rf':
        args['scaler'] = best_combination[0]
        args['n_estimators'] = best_combination[1]
        args['criterion'] = best_combination[2]
        args['max_features'] = best_combination[3]
        args['min_samples_split'] = best_combination[4]
        args['max_depth'] = best_combination[5]

    elif args['mla_selection'] == 'svm':
        args['scaler'] = best_combination[0]
        args['C'] = best_combination[1]
            
    SearchEndTime = time.time()
    print("Searching optimal parameter is done! (elapsed time: %s)"\
          % (utils.Timer(SearchEndTime-SearchStartTime)))
    
    print "The best auc=%f w/" % best_auc,
    print best_combination,
    if args['mla_selection'] == 'rf':
        print("(c.f. [scaler, n_estimators, criterion, max_features, min_samples_split, max_depth])")
    elif args['mla_selection'] == 'svm':
        print("(c.f. [scaler, C])")

    tag = utils.Tag(**args)
    utils.PickleDump(tag,best_clf,args['mla_selection'])
    plt.draw_roc(dataset.y_test,best_result,tag)

    save_optimal_combination(aucs,combinations,args['mla_selection'])


def find_optimal_parallel(combinations,samples,**args):

    print "\nSorry, parallel computing part is not implemented yet. :( \n"
    
    # aucs = []
    
    # best_auc = 0.
    # best_combination = []
    # best_clf = 0.
    # best_result = 0.
    
    # SearchStartTime = time.time()
    # print "\nSearching optimal training parameters (based on auc value) for %d combinations..."\
    #     % (len(combinations))

    # N.B. following for-loop should be modified to be compatible to any parallel programming
    # for c in combinations:
    #     c = list(c)

    #     clf = init_classifier(c,**args)

    #     args['scaler'] = c[0]
    #     dataset = ld.DataGeneration(samples,**args)
    #     print c,
    #     clf.fit(dataset.X_train,dataset.y_train)
    #     result_proba = clf.predict_proba(dataset.X_test)

    #     fpr, tpr, threshold = roc_curve(dataset.y_test, result_proba[:,1])
    #     roc_auc = auc(fpr,tpr)
    #     aucs.append(roc_auc)
    #     print roc_auc
        
    #     if roc_auc > best_auc:
    #         best_auc = roc_auc
    #         best_combination = c
    #         best_clf = clf
    #         best_result = result_proba
    #     else:
    #         best_auc = best_auc
    #         best_combination = best_combination
    #         best_clf = best_clf
    #         best_result = best_result

    #     del args['scaler']

    # # N.B. following parts may be usable for parallel computing too.
    # if args['mla_selection'] == 'rf':
    #     args['scaler'] = best_combination[0]
    #     args['n_estimators'] = best_combination[1]
    #     args['criterion'] = best_combination[2]
    #     args['max_features'] = best_combination[3]
    #     args['min_samples_split'] = best_combination[4]
    #     args['max_depth'] = best_combination[5]

    # elif args['mla_selection'] == 'svm':
    #     args['scaler'] = best_combination[0]
    #     args['C'] = best_combination[1]
            
    # SearchEndTime = time.time()
    # print("Searching optimal parameter is done! (elapsed time: %s)"\
    #       % (utils.Timer(SearchEndTime-SearchStartTime)))
    
    # print "The best auc=%f w/" % best_auc,
    # print best_combination,
    # if args['mla_selection'] == 'rf':
    #     print("(c.f. [scaler, n_estimators, criterion, max_features, min_samples_split, max_depth])")
    # elif args['mla_selection'] == 'svm':
    #     print("(c.f. [scaler, C])")

    # tag = utils.Tag(**args)
    # utils.PickleDump(tag,best_clf,args['mla_selection'])
    # plt.draw_roc(dataset.y_test,best_result,tag)

    # save_optimal_combination(aucs,combinations,args['mla_selection'])
