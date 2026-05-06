# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:22:51 2026

@author: a.bottacin
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import json

# =============================================================================
# Input configuration
# =============================================================================

with open("input_config.json") as f:
    cfg = json.load(f)

# Dataset
test_function  = cfg["dataset"]["test_function"]
sampling_type  = cfg["dataset"]["sampling_type"]
n              = cfg["dataset"]["n_train"]
n_samples      = cfg["dataset"]["n_samples"]
index_sample   = cfg["dataset"]["index_sample"]
min_sigma      = cfg["dataset"]["noise"]["min_sigma"]
max_sigma      = cfg["dataset"]["noise"]["max_sigma"]
noise_type     = cfg["dataset"]["noise"]["noise_type"]
length_scale   = cfg["dataset"]["correlation"]["length_scale"]
corr_kernel    = cfg["dataset"]["correlation"]["corr_kernel"]

# Plot parameters
kf               = cfg["plot"]["kf"]

# =============================================================================
# Import data defined by MATLAB script s1_GLS_SVM_training.m
# =============================================================================

file_data_train = "../Data/synthetic_data_train.csv";
file_data_pred = "../Data/synthetic_data_predictions.csv";
file_cov = "../Data/synthetic_cov_matrix.csv";

df_train = pd.read_csv(file_data_train)
df_pred = pd.read_csv(file_data_pred)
cov_mat = pd.read_csv(file_cov,header=None).values

x_train, y_train = df_train["x"].values.reshape(-1,1), df_train["y"].values
x_plot = df_pred["x"].values.reshape(-1,1)

# Define the known variance for each target point
var_y = np.diag(cov_mat)

# =============================================================================
# Gaussian Processes
# =============================================================================

# Define the kernel
kernel = RBF(length_scale=1e1, length_scale_bounds=(1e-3, 1e3)) 

# Initialize the GPs
gpr_hom = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
gpr_het = GaussianProcessRegressor(kernel=kernel, alpha=var_y, n_restarts_optimizer=15)

# Fit the models
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
    np.sqrt(var_y),
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
    y_gp_hom - kf * uy_gp_hom,
    y_gp_hom + kf * uy_gp_hom,
    alpha=0.5,
    label=r"GP homo band",
)
plt.fill_between(
    x_plot.ravel(),
    y_gp_het - kf * uy_gp_het,
    y_gp_het + kf * uy_gp_het,
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

df_pred["y_gp_hom"]  = y_gp_hom
df_pred["uy_gp_hom"] = uy_gp_hom
df_pred["y_gp_het"]  = y_gp_het
df_pred["uy_gp_het"] = uy_gp_het

file_data_res = "../Results/synthetic_data_results.csv";
df_pred.to_csv(file_data_res, index=False)

df_train["y_gp_hom"]  = y_train_gp_hom
df_train["uy_gp_hom"] = uy_train_gp_hom
df_train["y_gp_het"]  = y_train_gp_het
df_train["uy_gp_het"] = uy_train_gp_het

df_train.to_csv(file_data_train, index=False)