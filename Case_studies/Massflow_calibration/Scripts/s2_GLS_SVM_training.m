clear 
close all
clc

%% Add library to path

targetFolder = "..\..\..\GLS-SVM_library";
addpath(targetFolder);

%% Input configuration

run("input_config.m")

%% Assembling training dataset

rng("default") % For reproducibility

tb = readtable("..\Data\massflow_data.csv"); % calibration dataset
x = tb.x; % independent variable
y = tb.y; % response variable

n = length(x); % number of training data

x_plot = linspace(x(1), x(end), np)'; % test data

% Import the measurement covariance matrix
cov_matrix = importdata("..\Data\massflow_cov_matrix.csv");
uy = sqrt(diag(cov_matrix)); % Uncertainties

% Calibration curve
b = [ -39.96659414; 5.305832526; 0.6103696675; 0.01195172999; -0.0001248937568]; % regression coefficients
cov_mat_b = importdata("..\Data\GLS_coeffs_cov_matrix.csv"); % covariance matrix of regression coefficients
m_cal = @(b,x) b(1)*x.^(-1) + b(2)*x.^(-0.5) + b(3) + b(4)*x.^0.5 + b(5)*x; % calibration curve (GLS fit)
y_cal = m_cal(b,x_plot); % response variables evaluated with the GLS fit at the test points

% Derivatives with respect to regression coefficients used to propagate
% cov_mat_b:
dy_cal = @(x) [(x.^(-1))'; (x.^(-0.5))'; ones(1,length(x)); (x.^(0.5))'; (x.^(1))']; 

% Covariance matrix between the GLS predictions 
cov_mat_cal = dy_cal(x_plot)'*cov_mat_b*dy_cal(x_plot);
uy_cal = sqrt(diag(cov_mat_cal)); % uncertainties of the GLS predictions

% Heatmap of the measurement covariance matrix
figure(1);
imagesc(cov_matrix);
colorbar; axis square;
title('Heteroskedastic Covariance Matrix');
xlabel('Index'); ylabel('Index');

% Plot of the calibration dataset
figure(2), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,950,500])
p0 = plot(x_plot,y_cal,'r','LineWidth',linewidth_value);
p1 = errorbar(x,y,kf*uy,'ko',"MarkerFaceColor","k","MarkerSize",3,"LineWidth",0.02);
xlabel("{\it q_{N}} /(cm^3 min^{-1})"), ylabel("\it q_R/q_N")
legend([p1,p0],["Calibration points","Fit"],'Location','best')
xlim([0,1550])

fig_name = "../Results/Calibration_dataset";
saveas(gcf,fig_name,'epsc')
saveas(gcf,fig_name + ".png")

%% LS-SVM training 

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

% LS-SVM predictions
[y_ls, Vy_ls] = lssvm_pred(model_ls, x_plot);
uy_ls = sqrt(diag(Vy_ls));

%%  GLS-SVM training 

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

%% Gaussian Processes

% Import results obtained with python script

tb_gp = readtable("..\Results\GP_predictions.csv");

tb_gp_train = readtable("..\Results\GP_over_training_data.csv");

%% Uncertainties over training dataset

cov_mat_y_cal = dy_cal(x)'*cov_mat_b*dy_cal(x);
tb.y_gls = m_cal(b,x);
tb.y_glssvm = y_train_gls;
tb.y_gp_hom = tb_gp_train.y_gp_hom;
tb.y_gp_het = tb_gp_train.y_gp_het;
tb.uy_gls = sqrt(diag(cov_mat_y_cal));
tb.uy_glssvm = uy_train_gls;
tb.uy_gp_hom = tb_gp_train.uy_gp_hom;
tb.uy_gp_het = tb_gp_train.uy_gp_het;

%% Plots 

xconf = [x_plot;x_plot(end:-1:1)];
yconf_gls = [y_gls + kf*uy_gls; y_gls(end:-1:1) - kf*uy_gls(end:-1:1)];
yconf_gp_hom = [tb_gp.y_gp_hom + kf*tb_gp.uy_gp_hom; tb_gp.y_gp_hom(end:-1:1) - kf*tb_gp.uy_gp_hom(end:-1:1)];
yconf_gp_het = [tb_gp.y_gp_het + kf*tb_gp.uy_gp_het; tb_gp.y_gp_het(end:-1:1) - kf*tb_gp.uy_gp_het(end:-1:1)];
yconf_cal = [y_cal + kf*uy_cal; y_cal(end:-1:1) - kf*uy_cal(end:-1:1)];

% Predictions 
figure(3), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,700,500])
p1 = plot(x_plot,y_cal,'r-','LineWidth',linewidth_value);
p2 = plot(x_plot,y_ls,'--','LineWidth',linewidth_value,'Color',color_ls);
p3 = plot(x_plot,y_gls,'-','LineWidth',linewidth_value,'Color','#608b27');
p5 = plot(x_plot,tb_gp.y_gp_het ,'-','LineWidth',linewidth_value,'Color',color_gp_het);
p4 = plot(x_plot,tb_gp.y_gp_hom ,'--','LineWidth',linewidth_value,'Color',color_gp_hom);
p0 = errorbar(x,y,kf*uy,'ko',"MarkerFaceColor","k","MarkerSize",3,"LineWidth",0.02);
xlabel("{\it q_{N}} /(cm^3 min^{-1})"), ylabel("\it q_R/q_N")
legend([p0, p1, p2, p3, p4, p5],["Calibration points","GLS","LS-SVM","GLS-SVM","GP homo.","GP heter."],'Location','best')
xlim([0,1550]), ylim([0.84,1])

fig_name = "../Results/GLS_SVM_calibration";
saveas(gcf,fig_name,'epsc')
saveas(gcf,fig_name + ".png")

% Zoom of predictions with confidence bands
figure(4), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,700,500])
c1 = fill(xconf,yconf_gls,'red');
c1.EdgeColor = 'none';
c1.FaceColor = color_gls;
c1.FaceAlpha = 0.3;
c2 = fill(xconf,yconf_gp_hom,'red');
c2.EdgeColor = 'none';
c2.FaceColor = color_gp_hom;
c2.FaceAlpha = 0.3;
c3 = fill(xconf,yconf_gp_het,'red');
c3.EdgeColor = 'none';
c3.FaceColor = color_gp_het;
c3.FaceAlpha = 0.3;
c4 = fill(xconf,yconf_cal,'red');
c4.EdgeColor = 'none';
c4.FaceAlpha = 0.3;
p2 = plot(x_plot,y_ls,'--','LineWidth',linewidth_value,'Color',color_ls);
p3 = plot(x_plot,y_gls,'-','LineWidth',linewidth_value,'Color','#608b27');
p4 = plot(x_plot,tb_gp.y_gp_hom ,'--','LineWidth',linewidth_value,'Color',color_gp_hom);
p5 = plot(x_plot,tb_gp.y_gp_het ,'-','LineWidth',linewidth_value,'Color',color_gp_het);
p1 = plot(x_plot,y_cal,'r-','LineWidth',linewidth_value);
p0 = errorbar(x,y,kf*uy,'ko',"MarkerFaceColor","k","MarkerSize",3,"LineWidth",0.02);
xlabel("{\it q_{N}} /(cm^3 min^{-1})"), ylabel("\it q_R/q_N")
legend([p0, p1, p2, p3, p4, p5],["Calibration points","GLS","LS-SVM","GLS-SVM","GP homo.","GP heter."],'Location','best')
xlim([320,920]), box on

fig_name = "../Results/GLS_SVM_calibration_zoom";
saveas(gcf,fig_name,'epsc')
saveas(gcf,fig_name + ".png")

% Evaluation of the comparability index
E_gls = abs(y_gls - y_cal)./(kf*sqrt(uy_gls.^2 + uy_cal.^2));
E_gp_hom = abs(tb_gp.y_gp_hom - y_cal)./(kf*sqrt(tb_gp.uy_gp_hom.^2 + uy_cal.^2));
E_gp_het = abs(tb_gp.y_gp_het - y_cal)./(kf*sqrt(tb_gp.uy_gp_hom.^2 + uy_cal.^2));

% Comparability index
figure(5), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,700,500])
plot(x_plot,E_gls,'LineWidth',linewidth_value,'Color',color_gls);
plot(x_plot,E_gp_hom,'LineWidth',linewidth_value,'Color',color_gp_hom);
plot(x_plot,E_gp_het,'LineWidth',linewidth_value,'Color',color_gp_het);
xlabel("{\it q_N} /(cm^3 min^{-1})"),ylabel("{\it C}_{index}")
legend("GLS-SVM","GP homo.","GP heter.")
xlim([0,1550])

fig_name = "../Results/Comparability_MFC";

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")

