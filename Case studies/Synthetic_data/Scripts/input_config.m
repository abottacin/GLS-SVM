
%% Input configuration
raw = fileread("input_config.json");
cfg = jsondecode(raw);

% Dataset config
test_function  = cfg.dataset.test_function;
sampling_type  = cfg.dataset.sampling_type;
n              = cfg.dataset.n_train;
n_pred              = cfg.dataset.n_pred;
n_samples      = cfg.dataset.n_samples;
index_sample     = cfg.dataset.index_sample;
min_sigma      = cfg.dataset.noise.min_sigma;
max_sigma      = cfg.dataset.noise.max_sigma;
noise_type     = cfg.dataset.noise.noise_type;
length_scale   = cfg.dataset.correlation.length_scale;
corr_kernel    = cfg.dataset.correlation.corr_kernel;

% SVM config
test_config.svm_type             = cfg.svm_config.svm_type;
test_config.svm_kernel_type      = cfg.svm_config.svm_kernel_type;
test_config.n_train_iteration    = cfg.svm_config.n_train_iteration;
test_config.k_fold               = cfg.svm_config.k_fold;
test_config.n_obj_eval           = cfg.svm_config.bayesian_optimization.n_obj_eval;
test_config.n_seed_points        = cfg.svm_config.bayesian_optimization.n_seed_points;
test_config.exploration_ratio    = cfg.svm_config.bayesian_optimization.exploration_ratio;
test_config.parallel_flag        = cfg.svm_config.bayesian_optimization.parallel_flag;
test_config.verbose              = cfg.svm_config.bayesian_optimization.verbose;

% Plot parameters
kf                 = cfg.plot.kf;
linewidth_value    = cfg.plot.linewidth_value;
color_ls           = cfg.plot.colors.ls;
color_gls          = cfg.plot.colors.gls;
color_gp_hom       = cfg.plot.colors.gp_homo;
color_gp_het       = cfg.plot.colors.gp_hetero;
