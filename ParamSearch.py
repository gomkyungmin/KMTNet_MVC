import time
import numpy as np
import itertools
from StringIO import StringIO

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc

import Utils as utils


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


def find_optimal(combinations,X_train,y_train,X_test,y_test):

    aucs = []
    
    best_auc = 0.
    best_combination = []

    SearchStartTime = time.time()
    print("\nSearching optimal training parameters (based on auc value)...")
    for c in combinations:
        c = list(c)
        if c[4] == 'None':
            c[4] = None
            
        rfc = RandomForestClassifier(n_estimators=c[0],\
                                     criterion=c[1],\
                                     max_features=c[2],\
                                     min_samples_split=c[3],\
                                     max_depth=c[4])

        rfc.fit(X_train,y_train)
        result_proba = rfc.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, result_proba[:,1])
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_combination = c
        else:
            best_auc = best_auc
            best_combination = best_combination

    SearchEndTime = time.time()
    print("The best auc=%f w/" % best_auc),
    print(best_combination),
    print("(c.f. [n_estimators, criterion, max_features, min_samples_split, max_depth])")
    print("Searching optimal parameter is done! (elapsed time for %d combinations: %s)" % (len(combinations),utils.Timer(SearchEndTime-SearchStartTime)))

    save_optimal_combination(aucs,combinations)
    
    return best_combination


def save_optimal_combination(aucs,combinations):

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

    rank_file = 'combination_ranking.txt'
    header_label = 'n_estimator criterion max_features min_samples_split max_depth auc'
    np.savetxt(rank_file,arr_out,header=header_label,fmt="%s")
    print("\nRank of combinations of parameters saved in %s" % rank_file)
