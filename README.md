# KMTNet_MVC (ver. 0.2)

This package is built up based on python2.7 (2.7.10)

* Required packages

   numpy: related to preparation of training and testing samples.
   
   astropy: related to read out data catalog formatted in fits.
   
   scikit-learn: related to training and evaluation of given samples. Also, this package is required for the performance test.

* Description

   pipeline.py: multivariate classification pipeline (available MLAs: Random Forest, SVM)
   
   pipeline_param_search.py: multivariate classification pipeline w/ finding optimal parameter set for selected features (available MLAs: Random Forest, SVM)
   
* Required file(s)

   feature.txt: including names of features to be used.
   param_space_xx.txt: including parameters of features to be tested for the best combination (only for pipeline_param_search.py)


# KMTNet_MVC (ver. 0.1)

This package is built up based on python2.7 (2.7.10)

* Required packages

   numpy: related to preparation of training and testing samples.
   
   astropy: related to read out data catalog formatted in fits.
   
   scikit-learn: related to training and evaluation of given samples. Also, this package is required for the performance test.

* Required file(s)

   feature.txt: including names of features to be used.
