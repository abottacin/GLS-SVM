function [ypred, Vypred] = glssvm_pred(model, Xpred)
% glssvm_pred is used to make predictions with a trained GLS-SVM model
% 
%   Description
%       [ypred, Vypred] = glssvm_unbiased_pred(model, xpred) make predictions 
%       for new xpred input data using a trained GLS-SVM. 
% 
%   Inputs
%       model : GLS-SVM model trained with the function glssvm_training.m
%       Xpred : input matrix of new d-dimensional testing data (Npred x d)
% 
%   Outputs
%       ypred  : vector of predictions (Npred x 1)
%       Vypred : covariance matrix associated with the predictions (Npred x Npred)

    % Prediction 
    Kpred = K_matrix(Xpred, model.kernel_type, model.kernel_pars,model.X); % kernel matrix over training data
    
    ypred = Kpred*model.a + model.b;
    
    % Uncertainty propagation
    N = size(model.X,1);
    
    Npred = size(Xpred,1);
    
    eta = model.Otrain\ones(N,1); % auxiliary variable
    
    s_aux = ones(1,N)*eta; % auxiliary variable
    
    Oinv = inv(model.Otrain); % Matrix inversion
    
    Z = Oinv - (eta*eta'/s_aux); % alpha = Z*y
    
    q = eta/s_aux; % b = q'*y
    
    Hpred = Kpred*Z + ones(Npred,1)*q';
    
    Vypred = Hpred*model.Vy*Hpred';

end