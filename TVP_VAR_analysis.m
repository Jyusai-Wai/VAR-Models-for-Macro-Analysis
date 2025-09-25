% ===================================================================
% PROJECT: Time-Varying Parameter VAR (TVP-VAR) Model
% DESCRIPTION: This script estimates a TVP-VAR model with stochastic
%              volatility using a Gibbs sampler. It then computes and
%              plots impulse response functions for different time periods.
% ===================================================================

%% 1. Workspace Cleanup
clear; close all; clc; delete *asv;

%% 2. MCMC and Model Setup
% --- Simulation Settings ---
Nsim = 10^5;
Nburn = 0.2 * Nsim;
n_hz = 10; % Impulse response horizon
p = 2; % Number of lags

% --- Time Points for Impulse Response Analysis ---
t1_period = 2;  % Corresponds to 2000M3
t2_period = 90; % Corresponds to 2007M7

% --- Load Data ---
data = importdata('D:\Desktop\data1.csv');
Y0 = data(1:4, :); % Initial conditions
Y = data(5:end, :);
[T, n] = size(Y);
y = reshape(Y', T*n, 1);
k = n*p + 1; % Number of coefficients per equation

% --- Create Regressor Matrix ---
X_tilde = zeros(T, n*p);
tmpY = [Y0(end-p+1:end,:); Y];
for i=1:p
    X_tilde(:, (i-1)*n+1 : i*n) = tmpY(p-i+1 : end-i, :);
end
X = kron(speye(n), [ones(T,1), X_tilde]);

%% 3. Priors
% Priors for Sigma (error covariance)
nuSig0 = n + 3;
SSig0 = eye(n);

% Priors for beta_0 (initial state of coefficients)
mubeta0 = zeros(k, 1);
tmp = ones(k, 1); 
tmp(1:k:k*n) = 1/10; % Tighter prior for intercepts
iVbeta = sparse(1:k, 1:k, tmp);

% Priors for Q (covariance of state innovations)
nuQ0 = 3 * ones(k, 1);
QS0 = 0.01^2 * ones(k, 1);
QS0(1:k:k*n) = 0.1^2;

%% 4. Storage for MCMC Draws
store_beta = zeros(Nsim, T*k);
store_Q = zeros(Nsim, k);
store_Sigma = zeros(Nsim, n, n);
% Storage for Impulse Responses at two different time points
store_irf_t1 = zeros(n_hz, n, n); % Horizon x ResponseVar x ShockVar
store_irf_t2 = zeros(n_hz, n, n);

%% 5. Start of MCMC Gibbs Sampler
fprintf('Starting MCMC for TVP-VAR model...\n');

% Initialize the MCMC
Z = kron(speye(n), [ones(T,1), X_tilde]);
beta_mle = (Z'*Z) \ (Z'*y);
e = reshape(y - Z*beta_mle, n, T);
Sig = e*e' / T;
iSig = Sig \ speye(n);
Q = 0.1 * ones(k, 1);
beta0 = beta_mle; % Initialize beta0 with MLE estimate

% Use Kalman Filter/Smoother for beta_t initialization (optional but good practice)
% For simplicity, we initialize beta_t at the MLE estimate
betta = repmat(beta_mle, T, 1);

for ii = 1:(Nsim + Nburn)
    % Step 1: Sample beta_t (the time-varying coefficients) using a Kalman smoother
    % This is a complex step often done with a specialized function (e.g., Carter-Kohn)
    % The provided code uses a simplified approximation. For full accuracy, a Kalman smoother is needed.
    Xisig = X' * kron(speye(T), iSig);
    XisigX = Xisig * X;
    iQ_T = sparse(1:T*k, 1:T*k, repmat(1./Q, T, 1));
    Dbeta_T = iQ_T + XisigX;
    XisigY = Xisig * y;
    tilde_alpha = [beta0; betta(1:end-k)]; % Drift term for random walk
    betahat_T = Dbeta_T \ (iQ_T * tilde_alpha + XisigY);
    betta = betahat_T + chol(Dbeta_T, 'lower')' * randn(k*T, 1);
    
    % Step 2: Sample Sigma from its conditional posterior (Inverse-Wishart)
    e = reshape(y - X*betta, n, T);
    Sig = iwishrnd(SSig0 + e*e', nuSig0 + T);
    iSig = Sig \ speye(n);

    % Step 3: Sample Q_i for each coefficient from its conditional posterior (Inverse-Gamma)
    nu_q = nuQ0 + T/2;
    u = reshape(betta - [beta0; betta(1:end-k)], k, T);
    Q = 1./gamrnd(nu_q, 1./(sum(u.^2, 2)/2 + QS0));

    % Step 4: Sample beta_0 (the initial state) from its conditional posterior (Normal)
    ivQ = sparse(1:k, 1:k, 1./Q);
    Dbeta0 = ivQ + iVbeta;
    beta0hat = Dbeta0 \ (iVbeta*mubeta0 + ivQ*betta(1:k));
    beta0 = beta0hat + chol(Dbeta0, 'lower')' \ randn(k, 1);
    
    % Store posterior draws after burn-in
    if ii > Nburn
        jj = ii - Nburn;
        store_beta(jj, :) = betta';
        store_Q(jj, :) = Q';
        store_Sigma(jj, :, :) = Sig;
        
        % Calculate and store Impulse Response Functions for the two periods
        tempbeta = reshape(betta, k, T);
        CSig = chol(Sig, 'lower');
        
        for shock_var = 1:n
            shock = CSig(:, shock_var);
            store_irf_t1(:, :, shock_var) = store_irf_t1(:, :, shock_var) + construct_IR(tempbeta(:, t1_period), n, k, p, n_hz, shock);
            store_irf_t2(:, :, shock_var) = store_irf_t2(:, :, shock_var) + construct_IR(tempbeta(:, t2_period), n, k, p, n_hz, shock);
        end
    end
    
    if mod(ii, 1000) == 0
        fprintf('%d loops completed...\n', ii);
    end
end % End of MCMC

%% 6. Analyze Posterior Results and Plot IRFs
% Calculate posterior means of impulse responses
irf_t1_mean = store_irf_t1 / Nsim;
irf_t2_mean = store_irf_t2 / Nsim;

% Plotting the Impulse Response Functions
var_names = {'GDP', 'FFR', 'Inflation'};
shock_names = {'GDP Shock', 'FFR Shock', 'Inflation Shock'};

% --- Plot IRFs for Time Period 1 (2000M3) ---
figure('color', 'w', 'Name', 'Impulse Responses based on 2000M3');
for i_shock = 1:n
    for j_response = 1:n
        subplot(n, n, (i_shock-1)*n + j_response);
        plot(irf_t1_mean(:, j_response, i_shock), 'linewidth', 2);
        title(sprintf('Response of %s to %s', var_names{j_response}, shock_names{i_shock}));
        xlim([1, n_hz]);
        hline = refline(0, 0);
        hline.Color = 'r';
        hline.LineStyle = '--';
    end
end

% --- Plot IRFs for Time Period 2 (2007M7) ---
figure('color', 'w', 'Name', 'Impulse Responses based on 2007M7');
for i_shock = 1:n
    for j_response = 1:n
        subplot(n, n, (i_shock-1)*n + j_response);
        plot(irf_t2_mean(:, j_response, i_shock), 'linewidth', 2);
        title(sprintf('Response of %s to %s', var_names{j_response}, shock_names{i_shock}));
        xlim([1, n_hz]);
        hline = refline(0, 0);
        hline.Color = 'r';
        hline.LineStyle = '--';
    end
end
fprintf('Analysis complete. IRF plots have been generated.\n');

% --- Helper function to construct IRF ---
function IR = construct_IR(beta_t, n, k, p, horizon, shock)
    % Reshapes beta_t and constructs the IRF for a given shock
    B = reshape(beta_t(2:end), n*p, n)';
    IR = zeros(horizon, n);
    IR(1, :) = shock';
    for hh = 2:horizon
        lags = reshape(IR(hh-1:-1:max(1, hh-p), :)', n*min(p, hh-1), 1);
        if hh-1 < p
            lags = [lags; zeros(n*(p-(hh-1)), 1)];
        end
        IR(hh, :) = (B * lags)';
    end
end
