import time
import numpy as np
import itertools
from StringIO import StringIO

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve,auc
from sklearn.externals import joblib

import Utils as utils
import Plot as plt
import pipeline


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
    np.savetxt(rank_file,arr_out,header=header_label,fmt="%s")
    print("\nRank of combinations of parameters saved in %s" % rank_file)

    
def find_optimal(combinations,samples,**args):

    aucs = []
    
    best_auc = 0.
    best_combination = []
    best_clf = 0.
    best_result = 0.
    
    SearchStartTime = time.time()
    print("\nSearching optimal training parameters (based on auc value)...")
    for c in combinations:
        c = list(c)
        print c,
        
        X_train, y_train, X_test, y_test\
            = pipeline.data_generation(samples,\
                                       args['training_part'],\
                                       args['training_size'],\
                                       c[0])

        if args['mla_selection'] == 'rf':
            
            if c[5] == 'None':
                c[5] = None
            
            clf = RandomForestClassifier(n_estimators=c[1],\
                                         criterion=c[2],\
                                         max_features=c[3],\
                                         min_samples_split=c[4],\
                                         max_depth=c[5],\
                                         max_leaf_nodes=args['max_leaf_nodes'],\
                                         min_samples_leaf=args['min_samples_leaf'],\
                                         n_jobs=args['n_jobs'])

        if args['mla_selection'] == 'svm':

            clf = svm.SVC(C=c[1],\
                          kernel=args['kernel'],\
                          degree=args['degree'],\
                          gamma=args['gamma'],\
                          coef0=args['coef0'],\
                          probability=args['probability'],\
                          tol=args['tol'],\
                          max_iter=args['max_iter'],\
                          random_state=args['random_state'])
            
        clf.fit(X_train,y_train)
        result_proba = clf.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, result_proba[:,1])
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
        
    tag = utils.Tag(**args)
    utils.PickleDump(tag,best_clf,args['mla_selection'])
    plt.draw_roc(y_test,best_result,tag)
    
    SearchEndTime = time.time()
    print("The best auc=%f w/" % best_auc),
    print(best_combination),
    if args['mla_selection'] == 'rf':
        print("(c.f. [scaler, n_estimators, criterion, max_features, min_samples_split, max_depth])")
    elif args['mla_selection'] == 'svm':
        print("(c.f. [scaler, C])")
    print("Searching optimal parameter is done! (elapsed time for %d combinations: %s)" % (len(combinations),utils.Timer(SearchEndTime-SearchStartTime)))

    save_optimal_combination(aucs,combinations,args['mla_selection'])
    
    #return best_combination, best_clf
