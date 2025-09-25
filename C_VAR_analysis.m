% ===================================================================
% PROJECT: Constant-Parameter Bayesian VAR (C-VAR) Model
% DESCRIPTION: This script estimates a standard BVAR model using a
%              Gibbs sampler to analyze macroeconomic data.
% ===================================================================

%% 1. Workspace Cleanup
clear; close all; clc; delete *asv;

%% 2. Load and Visualize Data
% Load raw data from CSV
data_raw = importdata('D:\Desktop\data.csv');
data = data_raw.data;
time_vector1 = (2000:0.0803:2021.12)';

% Plot the raw data series
figure('color', 'w');
plot(time_vector1, data, 'linewidth', 3);
xlim([1999 2022]);
title('Raw Macroeconomic Time Series');

%% 3. Stationarity and Autocorrelation Analysis
% Perform Augmented Dickey-Fuller (ADF) tests for stationarity
ans_gdp_level = adftest(data(:,1)); % GDP
ans_r_level = adftest(data(:,2));   % Rate
ans_inf_level = adftest(data(:,3)); % Inflation

% Data is non-stationary, so take first differences
dgdp = diff(data(:,1));
dinf = diff(data(:,3));
ans_gdp_diff = adftest(dgdp);
ans_inf_diff = adftest(dinf);

% Calculate autocorrelation and partial autocorrelation
gdp_acf = autocorr(data(:,1));
r_acf = autocorr(data(:,2));
inf_acf = autocorr(data(:,3));
gdp_pacf = parcorr(data(:,1));
r_pacf = parcorr(data(:,2));
inf_pacf = parcorr(data(:,3));

% Plot ACF and PACF
figure('color', 'w');
subplot(2,3,1); stem(gdp_acf); title('GDP ACF');
subplot(2,3,2); stem(r_acf); title('Rate ACF');
subplot(2,3,3); stem(inf_acf); title('Inflation ACF');
subplot(2,3,4); stem(gdp_pacf); title('GDP PACF');
subplot(2,3,5); stem(r_pacf); title('Rate PACF');
subplot(2,3,6); stem(inf_pacf); title('Inflation PACF');

%% 4. MCMC Setup for C-VAR Model
% --- Housekeeping ---
p = 2;       % Number of lags in the VAR model
Nsim = 10^5; % Number of MCMC simulations
Nburn = 0.2 * Nsim; % Number of burn-in simulations

% --- Load Log-Transformed Data ---
% Log transformation done in Excel, resulting in loss of first data point.
data_log = importdata('D:\Desktop\data1.xlsx');
Y0 = data_log(1:4,:); % Initial conditions (first 4 observations)
Y = data_log(5:end,:);
[T, n] = size(Y);
y = reshape(Y', T * n, 1);

% --- Initialize Regressors ---
k = (n * p) + 1; % Number of coefficients per equation
tmpY = [Y0(end-p+1:end,:); Y];
X_tilde = zeros(T, n * p);
for i = 1:p
    X_tilde(:, (i-1)*n+1 : i*n) = tmpY(p-i+1 : end-i, :);
end
X_tilde = [ones(T,1), X_tilde];
x = kron(speye(n), X_tilde); % Stack regressors for SUR form

% --- Priors ---
nu0 = n + 3;
S0 = eye(n);
beta0 = zeros(n * k, 1);
% Precision for coefficients: 1 for standard, 1/10 for intercepts
tmp = ones(k * n, 1);
tmp(1:k:k*n) = 1/10;
ivbeta = sparse(1:k*n, 1:k*n, tmp);

% --- Storage for MCMC Draws ---
store_sig = zeros(Nsim, n, n);
store_beta = zeros(Nsim, n * k);

%% 5. Start of MCMC Gibbs Sampler
fprintf('Starting MCMC for C-VAR model...\n');

% Initialize the MCMC from MLE estimates
betta = (x'*x) \ (x'*y);
e = reshape(y - x*betta, n, T);
Sig = e*e' / T;
iSig = Sig \ speye(n);

for ii = 1:(Nsim + Nburn)
    % Step 1: Sample beta from its conditional posterior (Normal distribution)
    Xisig = x' * kron(speye(T), iSig);
    XisigX = Xisig * x;
    Dbeta = ivbeta + XisigX;
    XisigY = Xisig * y;
    betahat = Dbeta \ (ivbeta * beta0 + XisigY);
    betta = betahat + chol(Dbeta, 'lower')' * randn(n*k, 1);
    
    % Step 2: Sample Sigma from its conditional posterior (Inverse-Wishart distribution)
    e = reshape(y - x*betta, n, T);
    Sig = iwishrnd(S0 + e*e', nu0 + T);
    iSig = Sig \ speye(n);

    % Store posterior draws after burn-in period
    if ii > Nburn
        jj = ii - Nburn;
        store_beta(jj, :) = betta';
        store_sig(jj, :, :) = Sig;
    end
    
    if mod(ii, 1000) == 0
        fprintf('%d loops completed...\n', ii);
    end
end % End of MCMC

%% 6. Analyze Posterior Results
beta_mean = mean(store_beta);
beta_var = var(store_beta);
Sig_mean = squeeze(mean(store_sig));
Sig_var = squeeze(var(store_sig));

% Display results in the command window
fprintf('\n--- C-VAR Model Estimation Results ---\n');
fprintf('Posterior mean of beta:\n');
disp(reshape(beta_mean, k, n)');
fprintf('Posterior mean of Sigma:\n');
disp(Sig_mean);
fprintf('.......................................................\n');

% Check if zero falls within the credible intervals
beta_CI = quantile(store_beta, [.025 .975]);
Sig_CI = quantile(store_sig, [.025 .975]);

fprintf('Analysis complete.\n');
