import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg as linalg
from scipy.optimize import minimize

import numpy as np
import scipy.linalg as linalg
from scipy.special import gammaln
from scipy.stats import invgamma, uniform
# %pip install tqdm
from tqdm import tqdm  # For progress bars
import pandas as pd

np.random.seed(123)


def state_space(params, n_order):
    # unpak parameters
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = params

    # State vector dimensions
    state_dim = 2 + 2*n_order

    # observation matrix Z
    Z = np.zeros(state_dim)
    Z[0] = 1
    Z[state_dim-2] = 1

    # Inovation covariance matrix H
    H = sigma_epsilon2

    # Transition matrix T
    T = np.zeros((state_dim, state_dim))
    T[0,0]=1 # u_t = u_{t-1} + beta_{t-1}
    T[0,1]=1 # u_t = u_{t-1} + beta_{t-1}
    T[1,1]=1 # beta_t = beta_{t-1} + xi_t
    cos_lambda = np.cos(lambda_c)
    sin_lambda = np.sin(lambda_c)
    rotation_matrix = rho * np.array([[cos_lambda, sin_lambda],
                                      [-sin_lambda, cos_lambda]])
    for n in range(1, n_order + 1):
        idx = 2*n
        T[idx:idx+2, idx:idx+2] = rotation_matrix
        if n == 1:
            pass
        else:
            idx_prev = idx - 2
            T[idx:idx+2, idx_prev:idx_prev+2] += np.eye(2)

    # State loading matrix R
    R = np.zeros((state_dim, state_dim))
    R[1,1] = 1 # beta_t = beta_{t-1} + xi_t
    R[2:4, 2:4] = np.eye(2)  # For kappa_t and kappa_t^*
    # Higher-order cycles have no process noise in this model

    # Process covariance matrix Q
    Q = np.zeros((state_dim, state_dim))
    Q[1,1] = sigma_xi2
    Q[2:4, 2:4] = sigma_kappa2 * np.eye(2)  # For kappa_t and kappa_t^*
    # Higher-order cycles have no process noise in this model

    return Z, H, T, R, Q



def simulate_data(params, n_order, n_obs, alpha_init):
    Z, H, T, R, Q = state_space(params, n_order)
    state_dim = 2 + 2*n_order

     # Initialize state and obervation vectors
    alpha = np.zeros((n_obs, state_dim))
    y = np.zeros(n_obs)
    alpha[0, :] = alpha_init
    y[0] = Z @ alpha[0, :]+ np.random.normal(0, H)

    for t in range(1, n_obs):
        #process noise
        v = np.random.multivariate_normal(np.zeros(state_dim), Q)

        alpha[t, :] = T @ alpha[t-1, :] + R @ v
        y[t] = Z @ alpha[t, :] + np.random.normal(0, H)

    return y, alpha



def kalman_filter(y, a1, P1, params, n_order):
    Z, H, T, R, Q = state_space(params, n_order)
    Z = Z.reshape(1, -1)
    n = len(y)
    state_dim = 2 + 2 * n_order

    # Initialize arrays for estimates
    a_pred = np.zeros((n, state_dim))      # Predicted state estimates (a[t|t-1])
    P_pred = np.zeros((n, state_dim, state_dim))  # Predicted state covariances (P[t|t-1])
    a_filt = np.zeros((n, state_dim))      # Filtered state estimates (a[t|t])
    P_filt = np.zeros((n, state_dim, state_dim))  # Filtered state covariances (P[t|t])
    v = np.zeros(n)                        # Prediction errors
    F = np.zeros(n)                        # Prediction error variances
    K = np.zeros((n, state_dim))           # Kalman gain

    # Initial state
    a_filt[0] = a1
    P_filt[0] = P1

    for t in range(n):
        if t == 0:
            # For t=0, predict from initial state
            a_pred[t] = T @ a_filt[0]
            P_pred[t] = T @ P_filt[0] @ T.T + R @ Q @ R.T
        else:
            # Prediction step: 
            a_pred[t] = T @ a_filt[t - 1]
            P_pred[t] = T @ P_filt[t - 1] @ T.T + R @ Q @ R.T

        # Observation prediction: 
        y_pred = Z @ a_pred[t]

        # Prediction error (innovation): 
        v[t] = y[t] - y_pred.item()

        # Prediction error variance: 
        F[t] = (Z @ P_pred[t] @ Z.T).item() + H

        # Kalman gain: 
        K_t = P_pred[t] @ Z.T / F[t]
        K[t] = K_t.flatten()

        # Update state estimate: 
        a_filt[t] = a_pred[t] + K_t.flatten() * v[t]

        # Update covariance estimate
        P_filt[t] = P_pred[t] - K_t @ Z @ P_pred[t]

    return a_pred, P_pred, a_filt, P_filt, v, F, K



def kalman_smoother(y, a1, P1, params, n_order):
    n = len(y)
    state_dim = 2 + 2 * n_order

    # Run Kalman filter
    a_pred, P_pred, a_filt, P_filt, v, F, K = kalman_filter(y, a1, P1, params, n_order)

    # Initialize arrays for smoothed estimates
    a_smooth = np.zeros((n, state_dim))
    P_smooth = np.zeros((n, state_dim, state_dim))

    # Get state-space matrices
    Z, H, T, R, Q = state_space(params, n_order)

    # Backward recursion
    for t in reversed(range(n)):
        if t == n - 1:
            a_smooth[t] = a_filt[t]
            P_smooth[t] = P_filt[t]
        else:
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
            J = P_filt[t] @ T.T @ P_pred_inv
            a_smooth[t] = a_filt[t] + J @ (a_smooth[t + 1] - a_pred[t + 1])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

    return a_smooth, P_smooth




def log_likelihood(params, y, a1, P1, n_order):
    # run kalmann filter
    a_pred, P_pred, a_filt, P_filt, v, F, K = kalman_filter(y, a1, P1, params, n_order)
    
    loglik = -0.5 * (np.sum(np.log(2*np.pi) + np.log(abs(F)) + (v**2/F)))

    return loglik

def negative_log_likelihood(params, y, a1, P1, n_order):
    return -log_likelihood(params, y, a1, P1, n_order)



def transform_params(params_unbounded, a_rho=0.001, b_rho=0.970, a_lambda=0.001, b_lambda=np.pi):
    'Get the bounded paramters from unbounded draws'

    gamma_rho, gamma_lambda_c, gamma_sigma_xi2, gamma_sigma_kappa2, gamma_sigma_epsilon2 = params_unbounded

    # Transform gamma_rho (Uniform transformation for rho)
    exp_gamma_rho = np.exp(gamma_rho)
    rho = (a_rho + b_rho * exp_gamma_rho) / (1 + exp_gamma_rho)

    # Transform gamma_lambda_c (Uniform transformation for lambda_c)
    exp_gamma_lambda = np.exp(gamma_lambda_c)
    lambda_c = (a_lambda + b_lambda * exp_gamma_lambda) / (1 + exp_gamma_lambda)

    # Transform variance parameters (Inverse-Gamma transformation)
    sigma_xi2 = np.exp(gamma_sigma_xi2)
    sigma_kappa2 = np.exp(gamma_sigma_kappa2)
    sigma_epsilon2 = np.exp(gamma_sigma_epsilon2)

    return rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2





def log_derivatives_params(params_unbounded, a_rho=0.001, b_rho=0.970, a_lambda=0.001, b_lambda=np.pi):
    'Get the jacobian of the parameter transformations'

    gamma_rho, gamma_lambda_c, gamma_sigma_xi2, gamma_sigma_kappa2, gamma_sigma_epsilon2 = params_unbounded
    
    # Log derivative for rho
    log_derivative_rho = np.log(b_rho - a_rho) + gamma_rho - 2*np.log(1 + np.exp(gamma_rho))
    # Log derivative for lambda_c
    log_derivative_lambda_c = np.log(b_lambda - a_lambda) + gamma_lambda_c - 2*np.log(1 + np.exp(gamma_lambda_c))

    # Log derivatives for variance parameters
    log_derivative_sigma_xi2 = gamma_sigma_xi2
    log_derivative_sigma_kappa2 = gamma_sigma_kappa2
    log_derivative_sigma_epsilon2 = gamma_sigma_epsilon2

    return (log_derivative_rho, log_derivative_lambda_c, 
            log_derivative_sigma_xi2, log_derivative_sigma_kappa2, log_derivative_sigma_epsilon2)




# Define log prior
def log_prior(theta):
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = theta

    # log prior rho (uniform 0 to 0.999)
    a = 0
    b = 0.999
    log_prior_rho = np.log(1/(b-a))

    # log prior lambda_c (uniform 0 to pi)
    a = 0
    b = np.pi
    log_prior_lambda_c = np.log(1/(b-a))


    # Inverse-Gamma priors for variance parameters
    a = 3
    b = 1
    log_prior_sigma_xi2 = a * np.log(b) - gammaln(a) - (a + 1) * np.log(sigma_xi2) - b / sigma_xi2
    log_prior_sigma_kappa2 = a * np.log(b) - gammaln(a) - (a + 1) * np.log(sigma_kappa2) - b / sigma_kappa2
    log_prior_sigma_epsilon2 = a * np.log(b) - gammaln(a) - (a + 1) * np.log(sigma_epsilon2) - b / sigma_epsilon2

    return log_prior_lambda_c + log_prior_rho + log_prior_sigma_xi2 + log_prior_sigma_kappa2 + log_prior_sigma_epsilon2



# Define log posterior
def log_posterior(gamma, y, a1, P1, n_order):
    theta = transform_params(gamma)
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = theta

    log_lik = log_likelihood(theta, y, a1, P1, n_order)
    log_pri = log_prior(theta)

    if log_pri == -np.inf or np.isnan(log_lik):
        return -np.inf

    log_jacobian = np.sum(log_derivatives_params(gamma))

    return log_lik + log_pri + log_jacobian



def initialize_mcmc(y, a1, P1, n_order, 
                   n_init=40000, 
                   burn_init=5000, 
                   omega_init=0.1):

    dim = 5  # Number of parameters: rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2

    # Initialize storage for the chain
    chain_init = np.zeros((n_init, dim))
    current_gamma = np.zeros(dim)  # Initial value for gamma
    current_log_post = log_posterior(current_gamma, y, a1, P1, n_order)

    accept_init = 0

    for s in tqdm(range(n_init), desc="Initialization Phase"):
        # Propose Gamma*
        gamma_star = np.random.multivariate_normal(current_gamma, omega_init * np.eye(dim))
        
        # Compute log posterior
        log_post_star = log_posterior(gamma_star, y, a1, P1, n_order)
        
        # Acceptance probability
        if log_post_star == -np.inf:
            eta = 0
        else:
            eta = min(1, np.exp(log_post_star - current_log_post))
        
        # Accept or reject
        if np.random.uniform(0,1) < eta:
            chain_init[s] = gamma_star
            current_gamma = gamma_star
            current_log_post = log_post_star
            accept_init +=1
        else:
            chain_init[s] = current_gamma

    acceptance_rate_init = accept_init / n_init
    print(f"Initialization Acceptance Rate: {acceptance_rate_init*100:.2f}%")

    # Discard burn-in
    chain_init_burned = chain_init[burn_init:]

    # Compute covariance matrix from initialization
    Sigma = np.cov(chain_init_burned, rowvar=False)

    return chain_init_burned, Sigma, acceptance_rate_init





def recursion_mcmc(y, a1, P1, n_order, 
                  chain_init_burned, Sigma, 
                  n_rec=20000, 
                  burn_rec=10000, 
                  omega_rec=0.1):
  
    dim = 5  # Number of parameters: rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2

    # Initialize storage for the recursion chain
    chain_rec = np.zeros((n_rec, dim))
    theta_samples = np.zeros((n_rec, dim))

    # Start from the last point of the initialization chain
    current_gamma_rec = chain_init_burned[-1].copy()
    current_log_post_rec = log_posterior(current_gamma_rec, y, a1, P1, n_order)
    
    accept_rec = 0

    # Placeholder for alphas
    state_dim = 2 + 2 * n_order
    alphas = []

    for q in tqdm(range(n_rec), desc="Recursion Phase"):
        # Propose Gamma*
        gamma_star = np.random.multivariate_normal(current_gamma_rec, omega_rec * Sigma)
        # gamma_star = np.random.multivariate_normal(current_gamma_rec, omega_rec * np.eye(dim))
        
        # Compute log posterior
        log_post_star = log_posterior(gamma_star, y, a1, P1, n_order)
        
        # Acceptance probability
        if log_post_star == -np.inf:
            eta = 0
        else:
            eta = min(1, np.exp(log_post_star - current_log_post_rec))
        
        # Accept or reject
        if np.random.uniform(0,1) < eta:
            chain_rec[q] = gamma_star
            current_gamma_rec = gamma_star
            current_log_post_rec = log_post_star
            accept_rec +=1
        else:
            chain_rec[q] = current_gamma_rec
        
        theta_samples[q] = transform_params(current_gamma_rec)

        # After burn-in, perform Gibbs sampling for states
        if q >= (n_rec - burn_rec):
            theta = transform_params(current_gamma_rec)
            rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = theta

            # Sample states using Gibbs sampling
            # Step 1: Draw alpha^+
            alpha_plus_draw = np.random.multivariate_normal(np.zeros(state_dim), P1)
            
            # Step 2: Simulate forward recursion to get y^+
            y_plus,alpha_plus = simulate_data(theta, n_order, len(y), alpha_plus_draw)
            
            # Step 3: Construct y*
            y_star = y - y_plus
            
            # Step 4: Compute E(alpha | y*)
            a_smooth, _ = kalman_smoother(y_star, a1, P1, theta, n_order)
            alpha_hat = a_smooth[:-1]  # Exclude the last state
            
            # Step 5: Draw alpha^*
            alpha_star = alpha_hat + alpha_plus  # Adjusted state
            
            # Store the sampled alpha
            alphas.append(alpha_star)

    acceptance_rate_rec = accept_rec / n_rec
    print(f"Recursion Acceptance Rate: {acceptance_rate_rec*100:.2f}%")

    # Discard burn-in
    chain_rec_burned = chain_rec[burn_rec:]
    theta_samples_burned = theta_samples[burn_rec:]


    # Since Gibbs sampling starts after n_rec - burn_rec, we collect only the last burn_rec samples
    alpha_samples = np.array(alphas)

    return theta_samples, alpha_samples, acceptance_rate_rec

