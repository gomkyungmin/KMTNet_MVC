from os.path import dirname
from os.path import join

import numpy as np
from astropy.io import fits

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
    exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_data(datafile,features,cls_cut):

    module_path = dirname(__file__)
    base_dir = module_path
    print "Data is loaded from %s" % (join(base_dir,datafile))

    hdulist = fits.open(join(base_dir,datafile))
    catalog_data = hdulist[2].data

    classes_original = catalog_data['CLASS_STAR']
    classes_filtered = classes_original >= cls_cut
    target = classes_filtered.astype(np.int)

    features = np.genfromtxt(join(base_dir, features), delimiter=',', dtype=str)

    print("# of data: %d, # of features: %d" % (len(catalog_data),len(features)))
    print("features:")
    print(features)
    
    for j,feature in enumerate(features):
        if j == 0:
            flat_data = catalog_data[feature].reshape((len(catalog_data),1))
        else:
            flat_data = np.append(flat_data,catalog_data[feature].reshape((len(catalog_data),1)),1)

    
    return Bunch(features=features,\
                 data=flat_data,\
                 target=target)
    
