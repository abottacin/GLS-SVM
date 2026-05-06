# GLS-SVM
A MATLAB library, stored in the directory *GLS-SVM_library*, was developed for implementing a variant of the least squares support vector machines, called Generalized Least Squares Support Vector Machines (GLS-SVM), capable of accounting for the variance-covariance matrix of the target variable during training. 

The functions of the library allows to: 
* Tune the hyperparameters of the selected kernel function through a k-fold cross-validation coupled with a Bayesian optimizer;
* Train a GLS-SVM model with chosen/tuned hyperparameters;
* Perform predictions and evaluate the associated variance-covariance matrix;
* Estimate the bias affecting the predictions to construct reliable confidence intervals.

The library was developed by using MATLAB R2023b, using also the following toolboxes: Statistics and Machine Learning Toolbox and Parallel Computing Toolbox.

Two case studies, stored in the directory *Case studies
