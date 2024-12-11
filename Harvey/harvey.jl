module julia

export state_space, simulate_data, kalman_filter, kalman_smoother,
       log_likelihood, negative_log_likelihood, transform_params,
       log_derivatives_params, log_prior, log_posterior,
       initialize_mcmc, recursion_mcmc


using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

Random.seed!(123)

function state_space(params, n_order)
    # Unpack parameters
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = params

    # State vector dimensions
    state_dim = 2 + 2 * n_order

    # Observation matrix Z
    Z = zeros(state_dim)
    Z[1] = 1
    Z[state_dim - 1] = 1

    # Innovation covariance matrix H
    H = sigma_epsilon2

    # Transition matrix T
    T = zeros(state_dim, state_dim)
    T[1,1] = 1  # u_t = u_{t-1} + beta_{t-1}
    T[1,2] = 1  # u_t = u_{t-1} + beta_{t-1}
    T[2,2] = 1  # beta_t = beta_{t-1} + xi_t
    cos_lambda = cos(lambda_c)
    sin_lambda = sin(lambda_c)
    rotation_matrix = rho * [cos_lambda sin_lambda; -sin_lambda cos_lambda]

    for n in 1:n_order
        idx = 2*n + 1
        T[idx:idx+1, idx:idx+1] = rotation_matrix
        if n == 1
            # Do nothing
        else
            idx_prev = idx - 2
            T[idx:idx+1, idx_prev:idx_prev+1] += I(2)
        end
    end

    # State loading matrix R
    R = zeros(state_dim, state_dim)
    R[2,2] = 1  # beta_t = beta_{t-1} + xi_t
    R[3:4,3:4] = I(2)  # For kappa_t and kappa_t^*

    # Process covariance matrix Q
    Q = zeros(state_dim, state_dim)
    Q[2,2] = sigma_xi2
    Q[3:4,3:4] = sigma_kappa2 * I(2)

    return Z, H, T, R, Q
end

function simulate_data(params, n_order, n_obs, alpha_init)
    Z, H, T, R, Q = state_space(params, n_order)
    state_dim = 2 + 2*n_order

    # Initialize state and observation vectors
    alpha = zeros(n_obs, state_dim)
    y = zeros(n_obs)
    alpha[1, :] = alpha_init
    y[1] = dot(Z, alpha[1, :]) + rand(Normal(0, sqrt(H)))

    for t in 2:n_obs
        # Process noise
        v = zeros(state_dim)
        v[2] = rand(Normal(0, sqrt(params[3])))  # sigma_xi2
        v[3:4] = rand(MvNormal(zeros(2), params[4] * I(2)))  # sigma_kappa2

        alpha[t, :] = T * alpha[t-1, :] + R * v
        y[t] = dot(Z, alpha[t, :]) + rand(Normal(0, sqrt(H)))
    end

    return y, alpha
end

function kalman_filter(y, a1, P1, params, n_order)
    Z, H, T, R, Q = state_space(params, n_order)
    n = length(y)
    state_dim = 2 + 2 * n_order

    # Initialize arrays for estimates
    a_pred = zeros(n, state_dim)      # Predicted state estimates (a[t|t-1])
    P_pred = Array{Float64, 3}(undef, n, state_dim, state_dim)  # Predicted state covariances (P[t|t-1])
    a_filt = zeros(n, state_dim)      # Filtered state estimates (a[t|t])
    P_filt = Array{Float64, 3}(undef, n, state_dim, state_dim)  # Filtered state covariances (P[t|t])
    v = zeros(n)                        # Prediction errors
    F = zeros(n)                        # Prediction error variances
    K = zeros(n, state_dim)           # Kalman gain

    # Initial state
    a_filt[1, :] = a1
    P_filt[1, :, :] = P1

    for t in 1:n
        if t == 1
            # For t=1, predict from initial state
            a_pred[t, :] = T * a_filt[1, :]
            P_pred[t, :, :] = T * P_filt[1, :, :] * T' + R * Q * R'
        else
            # Prediction step: 
            a_pred[t, :] = T * a_filt[t - 1, :]
            P_pred[t, :, :] = T * P_filt[t - 1, :, :] * T' + R * Q * R'
        end

        # Observation prediction: 
        y_pred = Z'*a_pred[t, :]

        # Prediction error (innovation): 
        v[t] = y[t] - y_pred

        # Prediction error variance: 
        F[t] = Z'* P_pred[t, :, :] * Z + H

        # Kalman gain: 
        K_t = (P_pred[t, :, :] * Z) / F[t]
        K[t, :] = K_t
        

        # Update state estimate: 
        a_filt[t, :] = a_pred[t, :] + K_t * v[t]

        # Update covariance estimate
        P_filt[t, :, :] = (I(state_dim) - K_t * Z') * P_pred[t, :, :]
    end

    return a_pred, P_pred, a_filt, P_filt, v, F, K
end

function kalman_smoother(y, a1, P1, params, n_order)
    n = length(y)
    state_dim = 2 + 2 * n_order

    # Run Kalman filter
    a_pred, P_pred, a_filt, P_filt, v, F, K = kalman_filter(y, a1, P1, params, n_order)

    # Initialize arrays for smoothed estimates
    a_smooth = zeros(n, state_dim)
    P_smooth = Array{Float64, 3}(undef, n, state_dim, state_dim)

    # Get state-space matrices
    Z, H, T, R, Q = state_space(params, n_order)

    # Backward recursion
    for t in n:-1:1
        if t == n
            a_smooth[t, :] = a_filt[t, :]
            P_smooth[t, :, :] = P_filt[t, :, :]
        else
            P_pred_inv = inv(P_pred[t + 1, :, :])
            J = P_filt[t, :, :] * T' * P_pred_inv
            a_smooth[t, :] = a_filt[t, :] + J * (a_smooth[t + 1, :] - a_pred[t+1, :])
            P_smooth[t, :, :] = P_filt[t, :, :] + J * (P_smooth[t + 1, :, :] - P_pred[t + 1, :, :]) * J'
        end
    end

    return a_smooth, P_smooth
end

function log_likelihood(params, y, a1, P1, n_order)
    # Run Kalman filter
    a_pred, P_pred, a_filt, P_filt, v, F, K = kalman_filter(y, a1, P1, params, n_order)

    loglik = -0.5 * sum(log(2 * π) .+ log.(F) .+ (v .^ 2 ./ F))

    return loglik
end

function negative_log_likelihood(params, y, a1, P1, n_order)
    return -log_likelihood(params, y, a1, P1, n_order)
end

function transform_params(params_unbounded, priors)
    gamma_rho, gamma_lambda_c, gamma_sigma_xi2, gamma_sigma_kappa2, gamma_sigma_epsilon2 = params_unbounded
    a_rho, b_rho, a_lambda, b_lambda, a_xi, b_xi, a_kappa, b_kappa, a_epsilon, b_epsilon = priors

    # Transform gamma_rho (Uniform transformation for rho)
    exp_gamma_rho = exp(gamma_rho)
    rho = (a_rho + b_rho * exp_gamma_rho) / (1 + exp_gamma_rho)

    # Transform gamma_lambda_c (beta transformation for lambda_c support on (0,1))
    exp_gamma_lambda = exp(gamma_lambda_c)
    lambda_c = (a_lambda + b_lambda * exp_gamma_lambda) / (1 + exp_gamma_lambda)
  
    # # Transform variance parameters (Inverse-Gamma transformation)
    # sigma_xi2 = exp(gamma_sigma_xi2)
    # sigma_kappa2 = exp(gamma_sigma_kappa2)
    # sigma_epsilon2 = exp(gamma_sigma_epsilon2)

    # Transform variance parameters (Uniform transformation)
    exp_gamma_sigma_xi2 = exp(gamma_sigma_xi2)
    sigma_xi2 = (a_xi + b_xi * exp_gamma_sigma_xi2) / (1 + exp_gamma_sigma_xi2)

    exp_gamma_sigma_kappa2 = exp(gamma_sigma_kappa2)
    sigma_kappa2 = (a_kappa + b_kappa * exp_gamma_sigma_kappa2) / (1 + exp_gamma_sigma_kappa2)

    exp_gamma_sigma_epsilon2 = exp(gamma_sigma_epsilon2)
    sigma_epsilon2 = (a_epsilon + b_epsilon * exp_gamma_sigma_epsilon2) / (1 + exp_gamma_sigma_epsilon2)

    return rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2
end



function log_derivatives_params(params_unbounded, priors)
    gamma_rho, gamma_lambda_c, gamma_sigma_xi2, gamma_sigma_kappa2, gamma_sigma_epsilon2 = params_unbounded
    a_rho, b_rho, a_lambda, b_lambda, a_xi, b_xi, a_kappa, b_kappa, a_epsilon, b_epsilon = priors

    # Log derivative for rho
    log_derivative_rho = log(b_rho - a_rho) + gamma_rho - 2 * log(1 + exp(gamma_rho))

    # Log derivative for lambda_c
    log_derivative_lambda_c= log(b_lambda - a_lambda) + gamma_lambda_c - 2 * log(1 + exp(gamma_lambda_c))

    # # Inverse Gamma Log derivatives for variance parameters
    # log_derivative_sigma_xi2 = gamma_sigma_xi2
    # log_derivative_sigma_kappa2 = gamma_sigma_kappa2
    # log_derivative_sigma_epsilon2 = gamma_sigma_epsilon2

    # Uniform Log derivatives for variance parameters
    log_derivative_sigma_xi2 = log(b_xi - a_xi) + gamma_sigma_xi2 - 2 * log(1 + exp(gamma_sigma_xi2))
    log_derivative_sigma_kappa2 = log(b_kappa - a_kappa) + gamma_sigma_kappa2 - 2 * log(1 + exp(gamma_sigma_kappa2))
    log_derivative_sigma_epsilon2 = log(b_epsilon - a_epsilon) + gamma_sigma_epsilon2 - 2 * log(1 + exp(gamma_sigma_epsilon2))

    return (log_derivative_rho, log_derivative_lambda_c, 
            log_derivative_sigma_xi2, log_derivative_sigma_kappa2, log_derivative_sigma_epsilon2)
end



function log_prior(theta, priors)
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = theta
    a_rho, b_rho, a_lambda, b_lambda, a_xi, b_xi, a_kappa, b_kappa, a_epsilon, b_epsilon = priors

    # Log prior rho 
    log_prior_rho = log(1 / (b_rho - a_rho))
   
    # Log prior lambda_c 
    log_prior_lambda_c = log(1 / (b_lambda - a_lambda))  #uniform
   
    # # Inverse-Gamma priors for variance parameters
    # log_prior_sigma_xi2 = a_xi * log(b_xi) - lgamma(a_xi) - (a_xi + 1) * log(sigma_xi2) - b_xi / sigma_xi2
    # log_prior_sigma_kappa2 = a_kappa * log(b_kappa) - lgamma(a_kappa) - (a_kappa + 1) * log(sigma_kappa2) - b_kappa / sigma_kappa2
    # log_prior_sigma_epsilon2 = a_epsilon * log(b_epsilon) - lgamma(a_epsilon) - (a_epsilon + 1) * log(sigma_epsilon2) - b_epsilon / sigma_epsilon2

    # Uniform priors for variance parameters
    log_prior_sigma_xi2 = log(1 / (b_xi - a_xi))
    log_prior_sigma_kappa2 = log(1 / (b_kappa - a_kappa))
    log_prior_sigma_epsilon2 = log(1 / (b_epsilon - a_epsilon))


    return log_prior_lambda_c + log_prior_rho + log_prior_sigma_xi2 + log_prior_sigma_kappa2 + log_prior_sigma_epsilon2
end


function log_posterior(gamma, priors, y, a1, P1, n_order)
    theta = transform_params(gamma, priors)
    rho, lambda_c, sigma_xi2, sigma_kappa2, sigma_epsilon2 = theta

    log_lik = log_likelihood(theta, y, a1, P1, n_order)
    log_pri = log_prior(theta, priors)

    if log_pri == -Inf || isnan(log_lik)
        return -Inf
    end

    log_jacobian = sum(log_derivatives_params(gamma, priors))

    return log_lik + log_jacobian  #+ log_pri
end




function initialize_mcmc(y, priors, a1, P1, n_order; n_init=40000, burn_init=5000, omega_init=0.1)
    dim = length(priors)/2  # Number of parameters
    dim = Int(dim)

    # Initialize storage for the chain
    chain_init = zeros(n_init, dim)
    current_gamma = zeros(dim)  # Initial value for gamma
    current_log_post = log_posterior(current_gamma, priors, y, a1, P1, n_order)

    accept_init = 0

    pb = Progress(n_init; desc="Initialization Phase")

    for s in 1:n_init
        # Propose Gamma*
        gamma_star = rand(MvNormal(current_gamma, omega_init * I(dim)))

        # Compute log posterior
        log_post_star = log_posterior(gamma_star, priors, y, a1, P1, n_order)

        # Acceptance probability
        if log_post_star == -Inf
            eta = 0
        else
            eta = min(1, exp(log_post_star - current_log_post))
        end

        # Accept or reject
        if rand() < eta
            chain_init[s, :] = gamma_star
            current_gamma = gamma_star
            current_log_post = log_post_star
            accept_init += 1
        else
            chain_init[s, :] = current_gamma
        end

        next!(pb)
    end

    acceptance_rate_init = accept_init / n_init
    println("Initialization Acceptance Rate: $(acceptance_rate_init * 100) %")

    # Discard burn-in
    chain_init_burned = chain_init[burn_init+1:end, :]

    # Compute covariance matrix from initialization
    Sigma = cov(chain_init_burned)


    return chain_init_burned, Sigma, acceptance_rate_init, chain_init
end




function recursion_mcmc(y, priors, a1, P1, n_order, chain_init_burned, Sigma; n_rec=20000, burn_rec=10000, omega_rec=0.1)

    dim = length(priors)/2  # Number of parameters
    dim = Int(dim)

    # Initialize storage for the recursion chain
    chain_rec = zeros(n_rec, dim)
    theta_samples = zeros(n_rec, dim)

    # Start from the last point of the initialization chain
    current_gamma_rec = copy(chain_init_burned[end, :])
    current_log_post_rec = log_posterior(current_gamma_rec, priors, y, a1, P1, n_order)
    
    accept_rec = 0

    # Placeholder for alphas
    state_dim = 2 + 2 * n_order
    alphas = []

    pb = Progress(n_rec; desc="Recursion Phase")

    for q in 1:n_rec
        # Propose Gamma*
        gamma_star = rand(MvNormal(current_gamma_rec, omega_rec * Sigma))
        # gamma_star = rand(MvNormal(current_gamma_rec, omega_rec * I(dim)))
        
        # Compute log posterior
        log_post_star = log_posterior(gamma_star, priors, y, a1, P1, n_order)
        
        # Acceptance probability
        if log_post_star == -Inf
            eta = 0
        else
            eta = min(1, exp(log_post_star - current_log_post_rec))
        end
        
        # Accept or reject
        if rand() < eta
            chain_rec[q, :] = gamma_star
            current_gamma_rec = gamma_star
            current_log_post_rec = log_post_star
            accept_rec += 1
        else
            chain_rec[q, :] = current_gamma_rec
        end
        
        theta_samples[q, :] = collect(transform_params(current_gamma_rec, priors))


        # After burn-in, perform Gibbs sampling for states
        if q >= (burn_rec)
            theta = transform_params(current_gamma_rec, priors)

            # Sample states using Gibbs sampling
            # Step 1: Draw alpha^+
            alpha_plus_draw = rand(MvNormal(zeros(state_dim), P1))
            
            # Step 2: Simulate forward recursion to get y^+
            y_plus, alpha_plus = simulate_data(theta, n_order, length(y), alpha_plus_draw)
            
            # Step 3: Construct y*
            y_star = y - y_plus
            
            # Step 4: Compute E(alpha | y*)
            a_smooth, _ = kalman_smoother(y_star, a1, P1, theta, n_order)
            # a_pred, P_pred, a_filt, P_filt, v, F, K  = kalman_filter(y, a1, P1, theta, n_order)
            alpha_hat = a_smooth[:, :]  # exclude first and last
            
            # Step 5: Draw alpha^*
            alpha_star = alpha_hat .+ alpha_plus[:, :]  # Adjusted state
            
            # Store the sampled alpha
            push!(alphas, alpha_star)
        end

        next!(pb)
    end

    acceptance_rate_rec = accept_rec / n_rec
    println("Recursion Acceptance Rate: $(acceptance_rate_rec * 100) %")

    # Discard burn-in
    chain_rec_burned = chain_rec[burn_rec+1:end, :]
    theta_samples_burned = theta_samples[burn_rec+1:end, :]

    # Collect the sampled alphas
    alpha_samples = alphas

    return theta_samples_burned, alpha_samples, acceptance_rate_rec
end

end