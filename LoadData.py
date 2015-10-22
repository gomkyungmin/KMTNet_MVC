from os.path import dirname
from os.path import join

import numpy as np
from astropy.io import fits

from sklearn import preprocessing


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
    exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def LoadData(**args):

    module_path = dirname(__file__)
    base_dir = module_path
    print "Data is loaded from %s" % (join(base_dir,args['data_file']))

    hdulist = fits.open(join(base_dir,args['data_file']))
    catalog_data = hdulist[2].data

    classes_original = catalog_data['CLASS_STAR']
    classes_filtered = classes_original >= args['class_cut']
    target = classes_filtered.astype(np.int)

    features = np.genfromtxt(join(base_dir, args['feature_file']), delimiter=',', dtype=str)

    print "# of data: %d, # of features: %d" % (len(catalog_data),len(features))
    print "features:"
    print features
    
    for j,feature in enumerate(features):
        if j == 0:
            flat_data = catalog_data[feature].reshape((len(catalog_data),1))
        else:
            flat_data = np.append(flat_data,catalog_data[feature].reshape((len(catalog_data),1)),1)

    
    return Bunch(features=features,\
                 data=flat_data,\
                 target=target)
    

def DataScaler(X_train, X_test, scaler):

    if scaler == 'Standard':
        scaler = preprocessing.StandardScaler()
    elif scaler == 'MinMax':
        scaler = preprocessing.MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
                            

def DataGeneration(samples,**args):

    X = samples.data
    y = samples.target

    train_samples = len(samples.data)*args['training_size']

    if args['training_part'] == 'first':
        X_train = X[:train_samples]
        X_test = X[train_samples:]
        y_train = y[:train_samples]
        y_test = y[train_samples:]
    elif args['training_part'] == 'second':
        X_train = X[train_samples:]
        X_test = X[:train_samples]
        y_train = y[train_samples:]
        y_test = y[:train_samples]

    dataset = Bunch(X_train=X_train,\
                    X_test=X_test,\
                    y_train=y_train,\
                    y_test=y_test)

    # Preprocessing (Scaling) for X_train and X_test
    if args['scaler'] is not None:
        if 'param_search_file' in args:
            pass
        else:
            print "\nA scaler, %s, is applied in data generation." % args['scaler']
        dataset.X_train, dataset.X_test\
            = DataScaler(dataset.X_train, dataset.X_test, args['scaler'])
    else:
        if 'param_search_file' in args:
            pass
        else:
            print "\nNo scaler is applied in data generation."
            
    return dataset
