function avg_gmse = gmse_cv_rbf2(A2, sig2, X, y, Uy, kernel_type, cv)
% gmse_cv_rbf is the cross-validation function for a (Const)^2*RBF kernel
% 
%   Description
%       avg_general_rmse = gmse_cv_rbf(sig2, X, y, Uy, kernel_type, cv) is
%       the k-fold cross-validation function for an RBF kernel. 
%  
%   Inputs
%       X            : matrix of training data corresponding to the d-dimensional independent variable (N x d)
%       y            : vector of data corresponding to the response variable (N x 1)
%       Uy           : measurement covariance matrix associated with y (N x N)
%       kernel_type  : chosen kernel
%       cv           : cross-validation partition
%                   
%   Outputs
%       avg_gmse     : average of the GMSE value across the folds

rng("default") % For reproducibility

k_fold = cv.NumTestSets;

fold_gmse = zeros(k_fold, 1);
    
    for fold = 1:k_fold
        % Get training and validation indices for the current fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        X_train_fold = X(train_idx, :);
        Y_train_fold = y(train_idx);
        Uy_train_fold = Uy(train_idx,train_idx);

        N_fold = size(X_train_fold,1);
        
        X_val_fold = X(val_idx, :);
        Y_val_fold = y(val_idx);

        N_val = size(X_val_fold,1);

        Uy_val_fold = Uy(val_idx,val_idx);
              
        % Train GLS-SVM model for this fold
        
        K_train_fold = K_matrix(X_train_fold, kernel_type, [A2, sig2]); % kernel matrix over training data
        
        O_fold = K_train_fold + Uy_train_fold; % Omega matrix

        nu_fold = O_fold\Y_train_fold; % auxiliary variable

        eta_fold = O_fold\ones(N_fold,1); % auxiliary variable

        s_aux = ones(1,N_fold)*eta_fold; % auxiliary variable

        b = eta_fold'*Y_train_fold./s_aux; % bias

        a = nu_fold - eta_fold*b; % alpha
        
        K_val_fold = K_matrix(X_train_fold, kernel_type, [A2, sig2],X_val_fold); % kernel matrix over validation data
        
        Y_pred_val_fold = K_val_fold'*a + b; % Predictions

        Uy_cross = Uy_val_fold;

        try
            % Use matrix left division for X \ y which is equivalent to inv(X)*y
            % This is generally more numerically stable than explicit inv()
             gmse = 1/N_val*(Y_val_fold - Y_pred_val_fold)'*(Uy_cross\(Y_val_fold - Y_pred_val_fold));
        catch 
            % Handle potential singularity (e.g., if Sigma_val_fold is ill-conditioned)
            warning('Covariance matrix for fold %d is singular or ill-conditioned. Using pseudo-inverse.', fold);
            gmse = 1/N_val*(Y_val_fold - Y_pred_val_fold)' * pinv(Uy_cross) * (Y_val_fold - Y_pred_val_fold);
        end

        fold_gmse(fold) = gmse; % Minimize NLL

    end
    
    % Average GMSE across all folds 
    avg_gmse = mean(fold_gmse);
end
