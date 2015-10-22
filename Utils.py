import time, sys
from os.path import exists
from os.path import join

from sklearn.externals import joblib


def Timer(duration):

    hrs, rem = divmod(duration, 3600)
    mins, sec = divmod(rem, 60)
    output = "{:0>2}:{:0>2}:{:05.2f}".format(int(hrs),int(mins),sec)

    return output


def Tag(**args):

    if args['mla_selection'] == 'rf':
        tag = '_'+str(args['mla_selection'])\
              +'_trees'+str(args['n_estimators'])\
              +'_Mf'+str(args['max_features'])\
              +'_Md'+str(args['max_depth'])\
              +'_Mln'+str(args['max_leaf_nodes'])\
              +'_mss'+str(args['min_samples_split'])\
              +'_msl'+str(args['min_samples_leaf'])\
              +'_'+str(args['criterion'])\
              +'_'+str(args['training_part'])+str(args['training_size'])\
              +'_scaler'+str(args['scaler'])

    elif args['mla_selection'] == 'svm':
        if args['max_iter'] == -1:
            args['max_iter'] = 'None'
            
        tag = '_'+str(args['mla_selection'])\
              +'_C'+str(args['C'])\
              +'_k'+str(args['kernel'])\
              +'_d'+str(args['degree'])\
              +'_g'+str(args['gamma'])\
              +'_c'+str(args['coef0'])\
              +'_t'+str(args['tol'])\
              +'_Mi'+str(args['max_iter'])\
              +'_r'+str(args['random_state'])\
              +'_'+str(args['training_part'])+str(args['training_size'])\
              +'_scaler'+str(args['scaler'])

    return tag


def PickleDump(tag, clf, mla):

    pklfiledir = 'trained_pickle'
    pklfilename = 'trained'+tag+'.pkl'
    pklfile = join(pklfiledir,pklfilename)

    joblib.dump(clf, pklfile)
    print "Trained %s is saved in %s" % (mla, pklfile)
    print clf


def TrimArgs(**args):

    mla_args = args.copy()

    del mla_args['data_file'], mla_args['feature_file'],\
        mla_args['class_cut'], mla_args['training_size'],\
        mla_args['training_part'], mla_args['mla_selection'],\
        mla_args['scaler'], mla_args['pkl_file']
        
    if args['mla_selection'] == 'rf':

        del mla_args['feature_importance'], mla_args['classes'], mla_args['estimators'] 


    return mla_args


def TrimArgs_ps(**args):

    mla_args = args.copy()

    del mla_args['data_file'], mla_args['feature_file'],\
        mla_args['class_cut'], mla_args['training_size'],\
        mla_args['training_part'], mla_args['mla_selection'],\
        mla_args['param_search_file']
    
    if args['mla_selection'] == 'rf':

        del mla_args['feature_importance'], mla_args['classes'], mla_args['estimators'] 


    return mla_args


def QueryYesNo(question):

    valid = {"yes": True, "y": True,\
             "no": False, "n": False}
    
    prompt = " [y/n] "
    
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if choice is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

