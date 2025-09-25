# Bayesian VAR and TVP-VAR Models in MATLAB for Macroeconomic Analysis

This repository contains MATLAB scripts for implementing and analyzing two types of Vector Autoregression (VAR) models used in macroeconomic analysis: a standard Bayesian VAR (BVAR) and a Time-Varying Parameter VAR (TVP-VAR). The models are estimated using Markov Chain Monte Carlo (MCMC) methods, specifically the Gibbs sampler.

The analysis is performed on a dataset of macroeconomic variables, including GDP, the Federal Funds Rate (R), and Inflation (Inf).

## Models Implemented

### 1. Constant-Parameter Bayesian VAR (C-VAR)
-   This is a standard VAR model where the coefficients ($\beta$) and the variance-covariance matrix ($\Sigma$) are assumed to be constant over time.
-   The model is estimated using a Gibbs sampler with conjugate priors: an Inverse-Wishart prior for the covariance matrix and a Normal prior for the coefficients.
-   **Key tasks**:
    -   Performs ADF tests for stationarity and plots autocorrelation functions.
    -   Estimates the posterior distributions of the model parameters ($\beta$ and $\Sigma$).
    -   Calculates posterior means, variances, and credible intervals.

### 2. Time-Varying Parameter VAR (TVP-VAR)
-   This is an advanced VAR model that allows the model's parameters ($\beta_t$, $\Sigma_t$, and $Q_t$) to evolve over time, capturing structural changes in the economy.
-   The coefficients ($\beta_t$) are assumed to follow a random walk, allowing them to drift over time.
-   The model is also estimated using a Gibbs sampler.
-   **Key tasks**:
    -   Estimates the time-varying posterior distributions of all parameters.
    -   Generates **Impulse Response Functions (IRFs)** at different points in time (e.g., pre- and post-financial crisis) to analyze how the economy's response to shocks has changed.
    -   Visualizes the IRFs of GDP, FFR, and Inflation to various shocks.

## How to Use This Project

1.  **Prepare Your Data**:
    -   Ensure your data is in a `.csv` or `.xlsx` file. The scripts provided use `data.csv` and `data1.xlsx`.
    -   The data should be organized with time periods as rows and variables as columns (e.g., GDP, Rate, Inflation).
    -   Update the file paths inside the MATLAB scripts (`C_VAR_analysis.m` and `TVP_VAR_analysis.m`) to point to your data files.

2.  **Run the Analysis**:
    -   Open either `C_VAR_analysis.m` or `TVP_VAR_analysis.m` in MATLAB.
    -   Run the script by clicking the "Run" button or pressing F5.
    -   The script will execute the MCMC simulation, which may take several minutes depending on the number of simulations (`Nsim`).

3.  **View the Output**:
    -   **Console Output**: The scripts will print the posterior means and variances of the estimated parameters to the MATLAB command window.
    -   **Plots**: The scripts will generate several figures, including autocorrelation plots for the C-VAR and detailed impulse response function plots for the TVP-VAR.

## Required Software

-   **MATLAB**: The scripts are written in MATLAB and require a standard installation.
-   **MATLAB Toolboxes**: No special toolboxes are required for the core functionality, as functions like `iwishrnd` and `gamrnd` are part of the Statistics and Machine Learning Toolbox, which is standard in many installations.
