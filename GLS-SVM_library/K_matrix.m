function K = K_matrix(Xtrain,kernel_type, kernel_pars,Xt)
% K_matrix constructs the positive (semi-)definite and kernel matrix.
% 
%   Description
%       K = K_matrix(Xtrain,kernel_type, kernel_pars,Xt) constructs the kernel values for
%       all test data points in the rows of Xt, relative to the points of X. If Xt is not 
%       specified, the kernel matrix over the training data is constructed. 
%
%   Inputs
%       X      : N x d matrix with the inputs of the training data
%       kernel : Kernel type (by default 'RBF_kernel')
%       sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%       Xt(*)  : Nt x d matrix with the inputs of the test data
%   
%   Outputs
%       K  : N x N (N x Nt) kernel matrix


[nb_data,d] = size(Xtrain);

if strcmp(kernel_type,'RBF_kernel')
    if nargin<4
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        K = XXh+XXh'-2*(Xtrain*Xtrain');
        K = exp(-K./(2*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        K = XXh1+XXh2' - 2*Xtrain*Xt';
        K = exp(-K./(2*kernel_pars(1)));
    end

elseif strcmp(kernel_type,'RBF2_kernel')
      if nargin<4
            XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
            K = XXh+XXh'-2*(Xtrain*Xtrain');
            K = kernel_pars(1)*exp(-K./(2*kernel_pars(2)));
      else
            XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
            XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
            K = XXh1+XXh2' - 2*Xtrain*Xt';
            K = kernel_pars(1)*exp(-K./(2*kernel_pars(2)));
      end

elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4
        K = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        K = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end

    
end