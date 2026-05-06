# GLS-SVM
A MATLAB library, stored in the directory *GLS-SVM_library*, was developed for implementing a variant of the least squares support vector machines, called Generalized Least Squares Support Vector Machines (GLS-SVM), that is a kernel-based model capable of accounting for the variance-covariance matrix of the target variable during training. 

The functions of the library allows to: 
* Tune the hyperparameters of the selected kernel function through a k-fold cross-validation coupled with a Bayesian optimizer;
* Train a GLS-SVM model with chosen/tuned hyperparameters;
* Perform predictions and evaluate the associated variance-covariance matrix;
* Estimate the bias affecting the predictions to construct reliable confidence intervals.

The library was developed by using MATLAB R2023b, using also the following toolboxes: Statistics and Machine Learning Toolbox and Parallel Computing Toolbox.

Two examples, stored in the directory *Case_studies*, are reported for showing the potential of GLS-SVM in comparison of other kernel-based methods.
One of the case study is based on synthetic data, whereas the other is a real world example based on the calibration of a mass flow controller. 
The GLS-SVM is compared with LS-SVM and with Gaussian Processes, the latter trained using the python library scikit-learn 1.7.2. The training datasets can be found 
in the *Data* subfolders related to each case study.  
