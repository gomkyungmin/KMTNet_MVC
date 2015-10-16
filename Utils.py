import time
from os.path import join
from sklearn.externals import joblib


def Timer(duration):

    hrs, rem = divmod(duration, 3600)
    mins, sec = divmod(rem, 60)
    output = "{:0>2}:{:0>2}:{:05.2f}".format(int(hrs),int(mins),sec)

    return output


def PickleDump(tag, clf, mla):

    pklfiledir = 'trained_pickle'
    pklfilename = 'trained'+tag+'.pkl'
    pklfile = join(pklfiledir,pklfilename)

    joblib.dump(clf, pklfile)
    print("Trained %s is saved in %s" % (mla, pklfile))
