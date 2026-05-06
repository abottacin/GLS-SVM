function [ypred_unbiased, Vypred_unbiased] = lssvm_pred_unbiased(model,xpred)
% lssvm_pred_unbiased is used to make unbiased predictions with a trained LS-SVM model
% 
%   Description
%       ypred_unbiased, Vypred_unbiased] = lssvm_pred_unbiased(model, xpred) make unbiased 
%       predictions for new xpred input data using a trained LS-SVM. This functions is based 
%       on LS-SVMlib1.8 open-source library (https://www.esat.kuleuven.be/sista/lssvmlab/).
% 
%   Inputs
%       model : LS-SVM model trained with the LS-SVMlib1.8 library
%       xpred : input vector of new d-dimensional testing data (Npred x d)
% 
%   Outputs
%       ypred_unbiased  : vector of unbiased predictions (Npred x 1)
%       Vypred_unbiased : covariance matrix associated with the unbiased predictions (Npred x Npred) 

[~,Vy_input,~] = predlssvm(model,xpred,0.05,'pointwise');

model.Vy_input = Vy_input;

% Uncertainty quantification

N = size(model.xtrain,1);

Npred = size(xpred,1);

I = eye(N);

Ktrain = kernel_matrix(model.xtrain, model.kernel_type, model.kernel_pars); % kernel matrix over training data

Kpred = kernel_matrix(xpred, model.kernel_type, model.kernel_pars,model.xtrain); % kernel matrix over training data

Otrain = Ktrain + eye(N)/model.gam; % Omega matrix

eta = Otrain\ones(N,1); % auxiliary variable

s_aux = ones(1,N)*eta; % auxiliary variable

Oinv = inv(Otrain); % Matrix inversion

Z = Oinv - (eta*eta'/s_aux); % alpha = Z*y

q = eta/s_aux; % b = q'*y

Htrain = Ktrain*Z + ones(N,1)*q';

Hpred = Kpred*Z + ones(Npred,1)*q';

Hpred_c = Hpred*(2*I - Htrain);

ypred_unbiased = Hpred_c*model.ytrain;

Vypred_unbiased = Hpred_c*Vy_input*Hpred_c';


end