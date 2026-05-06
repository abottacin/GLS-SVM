
% Input configuration
raw = fileread("input_config.json");
cfg = jsondecode(raw);

% Dataset config
np              = cfg.dataset.n_pred;

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