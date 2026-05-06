clear 
close all
clc

%% Add library to path

targetFolder = "..\..\..\GLS-SVM_library";
addpath(targetFolder);

%% Input configuration

run("input_config.m")

%% Assembling training dataset

switch test_function
    case "sinc"
        x_train_lim = [-2.2,2.2];
        x_plot = linspace(x_train_lim(1),x_train_lim(2),n_pred)';
        f = @(x) sinc(x);
    case "sine"
        x_train_lim = [0,2*pi];
        x_plot = linspace(x_train_lim(1),x_train_lim(2),n_pred)';
        f = @(x) sin(x);
    case "lin+sine"
        x_train_lim = [-5,5];
        x_plot = linspace(x_train_lim(1),x_train_lim(2),n_pred)';
        f = @(x) 2*(x/10+sin(4*x/10)+sin(13*x/10));
end

switch sampling_type
    case "regular"
        x = linspace(x_train_lim(1),x_train_lim(2),n)';
    case "irregular"
        rng("default") % For reproducibility
        x = sort(datasample(x_plot(x_plot >= x_train_lim(1) & x_plot <= x_train_lim(2)),n,'Replace',false));
end

mean_vec  = f(x); % true y

switch noise_type
    case "constant"
        sigma = (max_sigma + min_sigma)/2*ones(n,1);
    case "linear"
        sigma = linspace(min_sigma, max_sigma, n)';    
    case "U-shaped"
        sigma = min_sigma + (max_sigma - min_sigma)/x(end)^2 * x.^2;
    case "cU-shaped"
        sigma = max_sigma - (max_sigma - min_sigma)/x(end)^2 * x.^2;
    case "exponential"
        A = (max_sigma - min_sigma +1)/(exp(x(end)-x(1))-1);
        sigma = (min_sigma - A) + A * exp(x-x(1));
end

% Build correlation matrix (Toeplitz, unit diagonal)
lags    = 0:n-1;

switch corr_kernel
    case "independent"
        rho_row = [1,zeros(1,n-1)]; 
    case "exponential"
        rho_row = exp(-lags / length_scale);  
    case "RBF"
        rho_row = exp(-lags.^2 / (2 * length_scale^2));
    case "linear"
        rho_row = max(0, 1 - lags / length_scale);
    case "power-law"
        p = 2;
        rho_row = sigma^2 ./ (1 + lags / length_scale).^p;
end

R       = toeplitz(rho_row); % pure correlation matrix


% Build heteroskedastic covariance matrix: C = diag(sigma) * R * diag(sigma)
S          = diag(sigma);
cov_matrix = S * R * S;

% Sample from multivariate normal

rng("default") % For reproducibility

samples   = mvnrnd(mean_vec, cov_matrix, n_samples);

fig_name = "../Results/Correlation_matrix" + noise_type + "_rho_" + corr_kernel;

% Heatmap of the defined correlation matrix
figure(1); clf, hold on, box on
set(gca,'FontSize',16)
imagesc(R);
colorbar; axis square;
xlabel('Index'); ylabel('Index');
xlim([0+0.5, n+0.5]), ylim([0+0.5,n+0.5])

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")

%% Training and testing datasets

% Training x already defined

% Training y chosen from the population of samples:
y = samples(index_sample,:)'; 

% Dataset visualization
figure(2), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,950,450])
errorbar(x,y,kf*sigma,'ko',"MarkerFaceColor","k","MarkerSize",3,"LineWidth",0.02,"CapSize",3);
plot(x_plot,f(x_plot),'r','LineWidth',linewidth_value+0.2);
xlabel("x"), ylabel("y")
legend("Data","m(x)",'Location','best')
ylim([-5,7])

fig_name = "../Results/Training_dataset_noise_" + noise_type + "_rho_" + corr_kernel;

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")

%% LS-SVM training (LS-SVMlab 1.8 open source library https://www.esat.kuleuven.be/sista/lssvmlab/)

rng("default") % For reproducibility

% Initialize the structure storing the best hyperparameters for LS-SVM
ls.cost = inf;
ls.gam = 0;
ls.model_pars = [];

% Train LS-SVM for the chosen number of times and select the best model
for j = 1:test_config.n_train_iteration

    model_ls = initlssvm(x, y, test_config.svm_type, [], [], test_config.svm_kernel_type,'original');
    model_ls = tunelssvm(model_ls,'simplex','crossvalidatelssvm',{test_config.k_fold,'mse'});
    model_ls  = trainlssvm(model_ls);

    if model_ls.cost < ls.cost
        ls.cost = model_ls.cost;
        ls.gam = model_ls.gam;
        ls.model_pars = model_ls.kernel_pars;
    end

end

% Train the overall best LS-SVM model
model_ls = initlssvm(x, y, test_config.svm_type, ls.gam, ls.model_pars, test_config.svm_kernel_type,'original');
model_ls  = trainlssvm(model_ls);

% LS-SVM predictions over training data
[y_train_ls, Vy_train_ls] = lssvm_pred_unbiased(model_ls, x);
uy_train_ls = sqrt(diag(Vy_train_ls));

% LS-SVM predictions over test data 
[y_ls, Vy_ls] = lssvm_pred_unbiased(model_ls, x_plot);
uy_ls = sqrt(diag(Vy_ls));

%%  GLS-SVM trainng

rng("default") % For reproducibility

% Tuning of the hyperparameters
[gls.kernel_pars, gls.cost, gls.cv] = glssvm_bayesopt(x, y, cov_matrix, test_config);

% Train the Final GLS-SVM Model with Tuned Parameters
model_gls = glssvm_training(x, y, cov_matrix, gls.kernel_pars, test_config.svm_kernel_type);

% GLS-SVM predictions over training data
[y_train_gls, Vy_train_gls] = glssvm_unbiased_pred(model_gls, x);
uy_train_gls = sqrt(diag(Vy_train_gls));

% GLS-SVM predictions over test data
[y_gls, Vy_gls] = glssvm_unbiased_pred(model_gls, x_plot);
uy_gls = sqrt(diag(Vy_gls));

%% Plot the predictions with confidence bands

figure(3), clf, hold on, box on, grid on
set(gca,"FontSize",14)

% input data
p0 = errorbar(x,y,kf*sigma,'k.',"MarkerSize",8,"LineWidth",0.02,"CapSize",0);
xlabel("x"), ylabel("y")

% LS-SVM
p2 = plot(x_plot,y_ls,'LineWidth',linewidth_value,'Color',color_ls);
plot(x_plot,y_ls + kf*uy_ls,'--','LineWidth',linewidth_value,'Color',color_ls)
plot(x_plot,y_ls - kf*uy_ls,'--','LineWidth',linewidth_value,'Color',color_ls)

% GLS-SVM
p3 = plot(x_plot,y_gls,'LineWidth',linewidth_value,'Color',color_gls);
plot(x_plot,y_gls + kf*uy_gls,'--','LineWidth',linewidth_value,'Color',color_gls)
plot(x_plot,y_gls - kf*uy_gls,'--','LineWidth',linewidth_value,'Color',color_gls)

% Reference function
p1 = plot(x_plot,f(x_plot),'r--','LineWidth',linewidth_value);
legend([p0,p1,p2,p3],["Data","f(x)","LS-SVM","GLS-SVM"])

%% Save simulation data and results

file_data_train = "../Data/synthetic_data_train.csv";
file_data_pred = "../Data/synthetic_data_predictions.csv";
file_cov = "../Data/synthetic_cov_matrix.csv";

tb_train = table();
tb_train.x = x; tb_train.y = y; tb_train.uy = sigma; 
tb_train.y_ls = y_train_ls; tb_train.uy_ls = uy_train_ls; 
tb_train.y_gls = y_train_gls; tb_train.uy_gls = uy_train_gls; 

tb_pred = table();
tb_pred.x = x_plot; tb_pred.y = f(x_plot); 
tb_pred.y_ls = y_ls; tb_pred.uy_ls = uy_ls; 
tb_pred.y_gls = y_gls; tb_pred.uy_gls = uy_gls; 

writetable(tb_train, file_data_train)
writetable(tb_pred, file_data_pred)
writematrix(cov_matrix, file_cov)

