from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import Classifier as CLF


class MLA:

    def __init__(self, tag, X_train, y_train, X_test, y_test, **kwargs):
        self.tag = tag
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kwargs = kwargs
        self.mla = self.kwargs['mla_selection']

        
    def train(self):
                
        if self.mla == 'rf':
            self.train = CLF.train_rf(self.tag,self.X_train,self.y_train,**self.kwargs)
            
        if self.mla == 'svm':
            self.train = CLF.train_svm(self.tag,self.X_train,self.y_train,**self.kwargs)
            
        return self.train

    
    def test(self,clf):
        
        if self.mla == 'rf':
            self.clf_result = CLF.test_rf(clf,self.X_test)

        if self.mla == 'svm':
            self.clf_result = CLF.test_svm(clf,self.X_test)

        return self.clf_result
