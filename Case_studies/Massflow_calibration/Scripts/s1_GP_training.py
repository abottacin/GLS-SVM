# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:20:52 2026

@author: a.bottacin
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
import matplotlib.pyplot as plt
import json

# =============================================================================
# Define polynomial kernel as for GLS-SVM (not built-in in scikit-learn)
# =============================================================================

class PolynomialKernel(Kernel):
    """
    Polynomial kernel: k(x_i, x_j) = (x_i^T x_j + t)^d

    Both t (bias) and d (degree) are treated as optimizable hyperparameters.

    Parameters
    ----------
    t : float, offset/bias term (default=1.0)
    d : float, degree of the polynomial (default=2.0)
    t_bounds : pair of floats, optimization bounds for t
    d_bounds : pair of floats, optimization bounds for d
    """
    def __init__(self, t=1.0, d=2.0, t_bounds=(1e-5, 1e5), d_bounds=(1e-5, 5.0)):
        self.t = t
        self.d = d
        self.t_bounds = t_bounds
        self.d_bounds = d_bounds

    @property
    def hyperparameter_t(self):
        return Hyperparameter("t", "numeric", self.t_bounds)

    @property
    def hyperparameter_d(self):
        return Hyperparameter("d", "numeric", self.d_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        base = X @ Y.T + self.t   # (x_i^T x_j + t), shape (n, m)
        K = base ** self.d         # kernel matrix

        if eval_gradient:
            if Y is not X:
                raise ValueError("Gradient can only be evaluated when Y is None (Y=X).")

            # dK/dt = d * (base)^(d-1)
            dK_dt = self.d * base ** (self.d - 1)               # shape (n, n)

            # dK/dd = (base)^d * log(base)
            dK_dd = base ** self.d * np.log(np.clip(base, 1e-10, None))  # shape (n, n)

            # scikit-learn expects gradients in log-space: dK/d(log theta) = theta * dK/d(theta)
            dK_dt_log = self.t * dK_dt                          # chain rule for log(t)
            dK_dd_log = self.d * dK_dd                          # chain rule for log(d)

            return K, np.dstack([dK_dt_log, dK_dd_log])         # shape (n, n, 2)

        return K

    def diag(self, X):
        """Return the diagonal of the kernel matrix k(x, x)."""
        return np.array([(x @ x + self.t) ** self.d for x in X])

    def is_stationary(self):
        return False
    
# =============================================================================
# Input configuration
# =============================================================================

with open("input_config.json") as f:
    cfg = json.load(f)    

n_pred = cfg["dataset"]["n_pred"]

# =============================================================================
# Import data from mass flow calibration
# =============================================================================

file_data_train = "../Data/massflow_data.csv";
file_cov = "../Data/massflow_cov_matrix.csv";

df_train = pd.read_csv(file_data_train)
cov_mat = pd.read_csv(file_cov,header=None).values

x_train, y_train = df_train["x"].values.reshape(-1,1), df_train["y"].values

x_plot = np.linspace(x_train[0], x_train[-1],n_pred).reshape(-1,1)

# Define the known variance for each target point
var_y = np.diag(cov_mat)

# =============================================================================
# Gaussian Processes
# =============================================================================

# Define the Kernel
kernel = PolynomialKernel(t=1.0, d=2.0)

# Initialize and train the GP
gpr_hom = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr_het = GaussianProcessRegressor(kernel=kernel, alpha=var_y, n_restarts_optimizer=10,normalize_y=True)

# Fit the model
gpr_hom.fit(x_train, y_train)
gpr_het.fit(x_train, y_train)

# Predictions over training dataset
y_train_gp_hom, uy_train_gp_hom = gpr_hom.predict(x_train, return_std=True)
y_train_gp_het, uy_train_gp_het = gpr_het.predict(x_train, return_std=True)

# Prediction over test dataset
y_gp_hom, uy_gp_hom = gpr_hom.predict(x_plot, return_std=True)
y_gp_het, uy_gp_het = gpr_het.predict(x_plot, return_std=True)

# =============================================================================
# Plot of the predictions with confidence bands
# =============================================================================

plt.errorbar(
    x_train,
    y_train,
    2*np.sqrt(var_y),
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)

plt.plot(x_plot, y_gp_hom, label="GP homo")
plt.plot(x_plot, y_gp_het, label="GP hetero")
plt.fill_between(
    x_plot.ravel(),
    y_gp_hom - 2 * uy_gp_hom,
    y_gp_hom + 2 * uy_gp_hom,
    alpha=0.5,
    label=r"GP homo band",
)
plt.fill_between(
    x_plot.ravel(),
    y_gp_het - 2 * uy_gp_het,
    y_gp_het + 2 * uy_gp_het,
    color="tab:orange",
    alpha=0.5,
    label=r"GP hetero band",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on a noisy dataset")


# =============================================================================
# Save the prediction and uncertainties in the dataframe
# =============================================================================

df_GP_pred = pd.DataFrame()

df_GP_pred["y_gp_hom"]  = y_gp_hom
df_GP_pred["uy_gp_hom"] = uy_gp_hom
df_GP_pred["y_gp_het"]  = y_gp_het
df_GP_pred["uy_gp_het"] = uy_gp_het

file_data_res = "../Results/GP_predictions.csv";
df_GP_pred.to_csv(file_data_res, index=False)

df_GP_train = pd.DataFrame()

df_GP_train["y_gp_hom"]  = y_train_gp_hom
df_GP_train["uy_gp_hom"] = uy_train_gp_hom
df_GP_train["y_gp_het"]  = y_train_gp_het
df_GP_train["uy_gp_het"] = uy_train_gp_het

file_data_res = "../Results/GP_over_training_data.csv";
df_GP_train.to_csv(file_data_res, index=False)