function [ypred, Vypred] = lssvm_pred(model, xpred)
% lssvm_pred is used to perform predictions with a trained LS-SVM model
% 
%   Description
%       [ypred, Vypred] = lssvm_pred(model, xpred) make predictions for new xpred
%       input data using a trained LS-SVM. This functions is based on
%       LS-SVMlib1.8 open-source library (https://www.esat.kuleuven.be/sista/lssvmlab/)
% 
%   Inputs
%       model : LS-SVM model trained with the LS-SVMlib1.8 library
%       xpred : input vector of new d-dimensional testing data (Npred x d)
% 
%   Outputs
%       ypred  : vector of predictions (Npred x 1)
%       Vypred : covariance matrix associated with the predictions (Npred x Npred) 

% Perform predictions 
ypred = simlssvm(model,xpred); 

% Estimation of the input covariance matrix associated with the response
% variable
[~,Vy_input,~] = predlssvm(model,xpred,0.05,'pointwise');
model.Vy_input = Vy_input;


% Uncertainty propagation

N = size(model.xtrain,1);

Npred = size(xpred,1);

Ktrain = kernel_matrix(model.xtrain, model.kernel_type, model.kernel_pars); % kernel matrix over training data

Kpred = kernel_matrix(xpred, model.kernel_type, model.kernel_pars,model.xtrain); % kernel matrix over training data

Otrain = Ktrain + eye(N)/model.gam; % Omega matrix

eta = Otrain\ones(N,1); % auxiliary variable

s_aux = ones(1,N)*eta; % auxiliary variable

Oinv = inv(Otrain); % Matrix inversion

Z = Oinv - (eta*eta'/s_aux); % alpha = Z*y

q = eta/s_aux; % b = q'*y

Hpred = Kpred*Z + ones(Npred,1)*q';

Vypred = Hpred*Vy_input*Hpred';


end