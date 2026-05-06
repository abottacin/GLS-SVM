clear 
close all
clc

tb_train = readtable("../Data/synthetic_data_train.csv");
tb_res = readtable("../Results/synthetic_data_results.csv");

run("input_config.m")

% Model predictions
figure(1), clf, hold on, box on
set(gca,'Fontsize',16)
set(gcf,'Position',[500,500,950,450])
p2 = plot(tb_res.x,tb_res.y_ls,'-','LineWidth',linewidth_value,'Color',color_ls); % LS-SVM
p3 = plot(tb_res.x,tb_res.y_gls,'-','LineWidth',linewidth_value,'Color',color_gls); % GLS-SVM
p4 = plot(tb_res.x,tb_res.y_gp_hom,'-','LineWidth',linewidth_value,'Color',color_gp_hom); % GP homoscedastic
p5 = plot(tb_res.x,tb_res.y_gp_het,'-','LineWidth',linewidth_value,'Color',color_gp_het); % GP heteroscedastic
p0 = plot(tb_train.x,tb_train.y,'ko',"MarkerFaceColor","k","MarkerSize",3,"LineWidth",0.02); % input data
p1 = plot(tb_res.x,tb_res.y,'r-','LineWidth',linewidth_value + 0.3); % Reference function
xlabel("\it x"), ylabel("\it y")
legend([p0,p1,p2,p3,p4,p5],["Data","m(x)","LS-SVM","GLS-SVM","GP homo.","GP heter."],'Location','northwest')
ylim([-5,7])

fig_name = "../Results/Mean_predictions_" + noise_type + "_rho_" + corr_kernel;

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")

% Residuals
figure(2), clf, hold on, box on
set(gca,"FontSize",16)
plot(tb_res.x,tb_res.y_ls - tb_res.y,'LineWidth',linewidth_value,'Color',color_ls)
plot(tb_res.x,tb_res.y_gls - tb_res.y ,'LineWidth',linewidth_value,'Color',color_gls)
plot(tb_res.x,tb_res.y_gp_hom - tb_res.y,'LineWidth',linewidth_value,'Color',color_gp_hom)
plot(tb_res.x,tb_res.y_gp_het - tb_res.y,'LineWidth',linewidth_value,'Color',color_gp_het)
xlabel("\it x"),ylabel("\it m_p - m")
legend(["LS-SVM","GLS-SVM","GP homo","GP hetero"],'location','best')

% Prediction uncertainties
figure(3), clf, hold on, box on
set(gca,"FontSize",16)
set(gcf,'Position',[500,500,950,450])
p3 = plot(tb_res.x,tb_res.uy_gp_hom,'LineWidth',linewidth_value,'Color',color_gp_hom);
p0 = plot(tb_train.x,tb_train.uy,'r-',"MarkerFaceColor",'r','LineWidth',linewidth_value,'MarkerSize',1.5);
p1 = plot(tb_res.x,tb_res.uy_ls,'LineWidth',linewidth_value,'Color',color_ls);
p2 = plot(tb_res.x,tb_res.uy_gls,'LineWidth',linewidth_value,'Color',color_gls);
p4 = plot(tb_res.x,tb_res.uy_gp_het,'LineWidth',linewidth_value,'Color',color_gp_het);
xlabel("\it x"),ylabel("{\it u}({\itm_p})");
legend([p0,p1,p2,p3,p4],["Data","LS-SVM","GLS-SVM","GP homo.","GP heter."],'location','best')
% legend([p0,p1,p2,p3],["LS-SVM","GLS-SVM","GP homo","GP hetero"],'location','best')

fig_name = "../Results/Prediction_uncertainties_" + noise_type + "_rho_" + corr_kernel;

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")


%  Evaluate the comparability index considering the chosen coverage factor
E_ls = abs(tb_res.y_ls - tb_res.y)./(kf*tb_res.uy_ls);
E_gls = abs(tb_res.y_gls - tb_res.y)./(kf*tb_res.uy_gls);
E_gp_hom = abs(tb_res.y_gp_hom - tb_res.y)./(kf*tb_res.uy_gp_hom);
E_gp_het = abs(tb_res.y_gp_het - tb_res.y)./(kf*tb_res.uy_gp_het);

% Comparability index
figure(4), clf, hold on, box on
set(gca,"FontSize",16)
set(gcf,'Position',[500,500,950,450])
plot(tb_res.x,E_ls,'LineWidth',linewidth_value,'Color',color_ls)
plot(tb_res.x,E_gls ,'LineWidth',linewidth_value,'Color',color_gls)
% plot(tb_res.x,E_gp_hom,'LineWidth',linewidth_value,'Color',color_gp_hom)
plot(tb_res.x,E_gp_het,'LineWidth',linewidth_value,'Color',color_gp_het)
xlabel("\it x"),ylabel("{\it C}_{index}")
legend(["LS-SVM","GLS-SVM","GP hetero."],'location','northeast')

fig_name = "../Results/Comparability_" + noise_type + "_rho_" + corr_kernel;

saveas(gcf,fig_name, 'epsc')
saveas(gcf,fig_name + ".png")


