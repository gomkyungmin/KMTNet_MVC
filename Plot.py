import os
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc


def draw_roc(y_test, result_proba, tag):

    figdir = 'figure'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    else:
        pass
    
    fpr, tpr, threshold = roc_curve(y_test, result_proba[:,1])
    roc_auc = auc(fpr, tpr)

    x_min = 1./len(y_test)

    random_guess_x = np.linspace(x_min,1,num=len(y_test),endpoint=True)
    random_guess_y = random_guess_x

    print "\nPlotting figures..."

    plt.clf()
    plt.grid(True,which='both')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr,tpr)
    plt.plot(random_guess_x,random_guess_y,'--',color='0.5',label='random guess')
    plt.text(0.45, 0.9, 'auc=%f' % roc_auc)
    plt.legend(loc='lower right')
    plt.savefig(figdir+'/plot_roc_lin'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_roc_lin'+tag+'.png',figdir)
    
    plt.clf()
    plt.grid(True,which='both')
    plt.xlim([x_min,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.semilogx(fpr, tpr)
    plt.semilogx(random_guess_x,random_guess_y,'--',color='0.5',label='random guess')
    plt.text(6e-3, 0.4, 'auc=%f' % roc_auc)
    plt.legend(loc='upper left')
    plt.savefig(figdir+'/plot_roc_xlog'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_roc_xlog'+tag+'.png',figdir)
