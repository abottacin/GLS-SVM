The JSON configuration file is used to set the main parameters of the simulation, which are loaded in MATLAB by the script "input_config.m".
The scripts should be executed following the prefix label:
	1) s1_GLS_SVM_training.m --> train of LS-SVM and GLS-SVM
	2) s2_GP_training.py --> train of homoscedastic and heteroscedastic Gaussian Processes
	3) s3_Results_analysis.m --> comparison among the models

The description of the simulation parameters are reported here below by following the same structure of the JSON file. 

DATASET CONSTRUCTION
	
	- test_function: select one of the test functions defined a priori. Available test functions: sinc, sine, lin+sine.
	- sampling_type: decide between regular or irregular sampling
	- n_train: number of training data.
	- n_pred: the number of predictions to be performed.
	- n_samples: number of datasets sampled from the covariance matrix characterising the response variable.
	- index_sample: index used to select one of the dataset sampled from the multivariate distribution of response data.

	NOISE 
		- min_sigma: minimum uncertainty associated with the response data.
		- max_sigma: maximum uncertainty associated with the response data.
		- noise_type: kind of heteroscedasticity characterising the target variable. 
	          Available types: constant, linear, U-shapes, cU-shaped and exponential.

	CORRELATION
		- length_scale: correlation length expressed in number of data
		- corr_kernel: type of correlation considered to construct the covariance matrix. 
		  Choose between: independent, linear, exponential, RBF, power-law.


TRAINING PARAMETERS

	- svm_type: type of training, "f" for function estimation, "c" for classification. 
	- svm_kernel_type: type of kernel used during training. Choose between RBF_kernel and poly_kernel. 
	- n_train_iteration: number of consecutive training to be performed for finding the minimum of the loss function. 
	- k_fold: number of folds for cross-validation.
	
	BAYESIAN OPTIMIZATION

		- n_obj_eval: number of evaluations of the objective function during GLS-SVM training.
		- n_seed_points: number of initial evaluation points.
		- exploration_ration: positive value associated to the propensity to explore the space of solutions.
		- parallel_flag: boolean flag used to toggle the MATLAB Parallel Computing Toolbox.
		- verbose: value equal to 0 and 1 used to suppress or not the output of the optimization.  

PLOT

	- kf: coverage factor considered for plotting uncertainties and evaluating the comparability index.
	- linewidth_value: value determining the width of lines in plots.
	- colors: list of colors associated with each model, expressed as hexadecimal code.



