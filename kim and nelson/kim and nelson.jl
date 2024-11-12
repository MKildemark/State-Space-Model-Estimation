module julia

export state_space, simulate_data, switching_kalman_filter,
       neg_log_likelihood, negative_log_likelihood, transform_params,
       log_derivatives_params, log_prior, log_posterior,
       initialize_mcmc, recursion_mcmc


using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

Random.seed!(123)



function state_space(params)
    # Unpack parameters
    psi1, psi2, sigma_xi2, sigma_kappa2_0, sigma_kappa2_1,
    sigma_epsilon2_0, sigma_epsilon2_1, omega, p, q = params
    

    # State vector dimensions u_t B_t c_t c_t-1
    state_dim = 4

    H = 0 #obs noise
    

    # Observation matrix Z
    Z = zeros(state_dim)
    Z[1] = 1
    Z[end - 1] = 1

    # Innovation covariance matrix H
    H = 0

    # Transition matrix T
    T = zeros(state_dim, state_dim)
    T[1, 1] = 1  # u_t = u_{t-1} + beta_{t-1}
    T[1, 2] = 1
    T[2, 2] = 1  # beta_t = beta_{t-1} + xi_t
    T[3, 3] = psi1 # c_t = psi1 * c_{t-1} + psi2 * c_{t-2} + kappa_t
    T[3,4] = psi2 
    T[4, 3] = 1  #c_t-1 = c_{t}


    # State loading matrix R
    R = zeros(state_dim, state_dim)
    R[1, 1] = 1  # u_t = u_{t-1} + beta_{t-1} + epsilon_t
    R[2, 2] = 1  # beta_t = beta_{t-1} + xi_t
    R[3,3] = 1  

    # Process covariance matrices Q_0 and Q_1
    Q_0 = zeros(state_dim, state_dim)
    Q_0[1, 1] = sigma_epsilon2_0
    Q_0[2, 2] = sigma_xi2
    Q_0[3, 3] = sigma_kappa2_0 

    Q_1 = zeros(state_dim, state_dim)
    Q_1[1, 1] = sigma_epsilon2_1
    Q_1[2, 2] = sigma_xi2
    Q_1[3, 3] = sigma_kappa2_1 

    nu0 = zeros(state_dim)
    nu1 = zeros(state_dim)

    nu1[3] = omega
    

    return Z,T, R, Q_0, Q_1, nu0, nu1, H
end



function simulate_data(params, n_obs, alpha_init)
    Z,T, R, Q_0, Q_1, nu0, nu1, H = state_space(params)
    
    psi1, psi2, sigma_xi2, sigma_kappa2_0, sigma_kappa2_1,
    sigma_epsilon2_0, sigma_epsilon2_1, omega, p, q = params

    state_dim = 4

    # Initialize state and observation vectors
    alpha = zeros(n_obs, state_dim)
    y = zeros(n_obs)
    # Initialize S_t
    S = zeros(Int, n_obs)

    # Initial state for S
    prob_S0 = [(1 - p) / (2 - p - q), (1 - q) / (2 - p - q)]
    S[1] = rand(Categorical(prob_S0))-1 #S is 0 or 1
    
    v = zeros(state_dim)
    if S[1] == 1  #Recesion
        # v = rand(MvNormal(zeros(state_dim), Q_1)) + nu1
        v[1] = rand(Normal(0, sqrt(params[7])))
        v[2] = rand(Normal(0, sqrt(params[3])))  # sigma_xi2
        v[3] = rand(Normal(0, sqrt(params[5]))) #sigma kappa
        v = v + nu1
    else
        # v = rand(MvNormal(zeros(state_dim), Q_0)) + nu0
        v[1] = rand(Normal(0, sqrt(params[6]))) # sigma_epsilon
        v[2] = rand(Normal(0, sqrt(params[3])))  # sigma_xi2
        v[3] = rand(Normal(0, sqrt(params[4]))) #sigma kappa
        v = v + nu0
    end
    alpha[1, :] = alpha_init + R * v
    y[1] = Z' * alpha[1, :] 

    # Simulate forwards
    for t in 2:n_obs
        if S[t - 1] == 1
            S[t] = rand(Categorical([1-p, p]))-1
        else
            S[t] = rand(Categorical([q, 1-q]))-1
        end

        v = zeros(state_dim)
        if S[t] == 1 #Recesion
            # v = rand(MvNormal(zeros(state_dim), Q_1)) + nu1
            v[1] = rand(Normal(0, sqrt(params[7]))) #sigma epsilon
            v[2] = rand(Normal(0, sqrt(params[3])))  # sigma_xi2
            v[3] = rand(Normal(0, sqrt(params[5]))) #sigma kappa
            v = v + nu1

        else
            # v = rand(MvNormal(zeros(state_dim), Q_0)) + nu0
            v[1] = rand(Normal(0, sqrt(params[6]))) # sigma_epsilon
            v[2] = rand(Normal(0, sqrt(params[3])))  # sigma_xi2
            v[3] = rand(Normal(0, sqrt(params[4]))) #sigma kappa
            v = v + nu0
     

        end

        alpha[t, :] = T * alpha[t - 1, :] + R * v
        y[t] = Z' * alpha[t, :] 
    end

    return y, alpha
end



function switching_kalman_filter(y, params, a1, P1)
    rho, lambda_c, sigma_xi2, sigma_kappa2_0, sigma_kappa2_1,
    sigma_epsilon2_0, sigma_epsilon2_1, omega, p, q = params


    Z,T, R, Q0, Q1, nu0, nu1, H = state_space(params)
    Q = [Q0, Q1]
    nu = [nu0, nu1]
    n = length(y)
    state_dim = 4

    # Initialize log-likelihood and initial state
    L = 0.0
    a_prev = zeros(2, state_dim)
    a_prev[1, :] = a1
    a_prev[2, :] = a1

    P_prev = zeros(2, state_dim, state_dim)
    P_prev[1, :, :] = P1
    P_prev[2, :, :] = P1

    # Initial probabilities
    p_S_ohm_prev = [(1 - p) / (2 - p - q), (1 - q) / (2 - p - q)]

    # Corrected definition of p_S_S
    p_S_S = zeros(2, 2)
    p_S_S[1, 1] = q # prob s = 0 given s = 0
    p_S_S[1, 2] = 1 - q # prob s = 1 given s = 0
    p_S_S[2, 1] = 1-p # prob s =0 given s = 1
    p_S_S[2, 2] = p # prob s = 1 given s = 1
    # List to store estimated states
    a_est_list = []
    #List to store probability of recesion 
    p_res_list = []

    S = [1, 2]  # Possible state index

    for t in 1:n
        # Initialize variables for current time step
        a_pred = zeros(2,2, state_dim)
        P_pred = zeros(2, 2, state_dim, state_dim)
        a_filt = zeros(2, 2, state_dim)
        P_filt = zeros(2, 2, state_dim, state_dim)
        v = zeros(2,2)
        F = zeros(2, 2)
        K = zeros(2, 2, state_dim)
        p_y_S_S_ohm = zeros(2, 2)
    

        for i in S
            for j in S
                # Prediction step
                a_pred_i_j = T * a_prev[i, :] + nu[j]
                P_prev_i = reshape(P_prev[i, :, :], state_dim, state_dim)
                P_pred_i_j = T * P_prev_i * T' + R * Q[j] * R'

                # Observation prediction
                y_pred = Z' * a_pred_i_j

                # Prediction error (innovation)
                v_i_j = y[t] - y_pred

                # Prediction error variance
                F_i_j = Z' * P_pred_i_j * Z #+ H

                # Kalman gain
                K_i_j = (P_pred_i_j * Z) / F_i_j  # Shape: (state_dim,)

                # Update state estimate
                a_filt_i_j = a_pred_i_j + K_i_j * v_i_j

                # Update covariance estimate
                P_filt_i_j = (I(state_dim) - K_i_j * Z') * P_pred_i_j
         
                # Store results
                a_pred[i,j, :] = a_pred_i_j
                P_pred[i, j, :, :] = P_pred_i_j
                a_filt[i, j, :] = a_filt_i_j
                P_filt[i, j, :, :] = P_filt_i_j
                v[i,j] = v_i_j
                F[i, j] = F_i_j
                K[i, j, :] = K_i_j

                # Compute p(y_t | S_{t-1}=i, S_t=j, Ψ_{t-1})
                p_y_S_S_ohm[i, j] = (1 / sqrt(2 * π * F[i,j])) * exp(-0.5 * v[i,j]^2 / F[i,j])
                # if error set p_y_S_S_ohm -1e8
                if isnan(p_y_S_S_ohm[i, j])
                    p_y_S_S_ohm[i, j] = -1e8
                end
            end
        end


        # Compute p(y_t | Ψ_{t-1})
        p_S_S_ohm_prev = zeros(2, 2)
        for i in S
            for j in S
                p_S_S_ohm_prev[i, j] = p_S_ohm_prev[i] * p_S_S[i, j]
            end
        end

        p_y_ohm_t = 0.0
        for i in S
            for j in S
                p_y_ohm_t += p_S_S_ohm_prev[i, j] * p_y_S_S_ohm[i, j] 
            end
        end
        L += log(p_y_ohm_t)

        # Update probabilities
        p_S_S_ohm_t = zeros(2, 2)
        for i in S
            for j in S
                p_S_S_ohm_t[i, j] = (p_S_S_ohm_prev[i, j] * p_y_S_S_ohm[i, j]) / p_y_ohm_t
            end
        end

        

        p_S_ohm_t = zeros(2)
        for j in S
            for i in S
                p_S_ohm_t[j] += p_S_S_ohm_t[i, j]
            end
        end
        #ensure no zeros in p_S_ohm_t
        for j in S
            if p_S_ohm_t[j] == 0
                p_S_ohm_t[j] = 1e-8
            end
        end
  
        push!(p_res_list, p_S_ohm_t)

        # Combine filtered estimates
        a_t = zeros(2, state_dim)
        P_t = zeros(2, state_dim, state_dim)

        for j in S
            numerator_a = zeros(state_dim)
            denominator = p_S_ohm_t[j]
            for i in S
                numerator_a += p_S_S_ohm_t[i, j] * a_filt[i, j, :]
            end
            a_t_j = numerator_a / denominator
            a_t[j, :] = a_t_j

            numerator_P = zeros(state_dim, state_dim)
            for i in S
                diff = a_t_j - a_filt[i, j, :]
                P_filt_i_j = reshape(P_filt[i, j, :, :], state_dim, state_dim)
                numerator_P += p_S_S_ohm_t[i, j] * (P_filt_i_j + diff * diff')
            end
            P_t[j, :, :] = numerator_P / denominator
        end

        # Estimate state
        a_est_t = zeros(state_dim)
        for j in S
            a_est_t += p_S_ohm_t[j] * a_t[j, :]
        end
        push!(a_est_list, a_est_t)

        # Prepare for next iteration
        a_prev = a_t
        P_prev = P_t
        p_S_ohm_prev = p_S_ohm_t
    end

    # Convert a_est_list to array
    a_est = hcat(a_est_list...)'  # Each row corresponds to a time step
    p_res = hcat(p_res_list...)'  
   
    return L, a_est, p_res
end


function neg_log_likelihood(params, y, a1, P1)
    logL, a, p_res = switching_kalman_filter(y, params, a1, P1)
    return -logL
end




function transform_params(params_unbounded, priors)
    # Unpack parameters
    gamma_psi1, gamma_psi2, gamma_sigma_xi2, gamma_sigma_kappa2_0, gamma_sigma_kappa2_1, gamma_sigma_epsilon2_0, gamma_sigma_epsilon2_1, gamma_omega, gamma_p, gamma_q = params_unbounded
    a_psi1, b_psi1, a_psi2, b_psi2, a_xi, b_xi, a_kappa_0, b_kappa_0, a_kappa_1, b_kappa_1, a_epsilon_0, b_epsilon_0, a_epsilon_1, b_epsilon_1, a_omega, b_omega, a_p, b_p, a_q, b_q = priors


    # Transform gamma_psi1 and gamma_psi2 (Uniform 0 to 1)
    exp_gamma_psi1 = exp(gamma_psi1)
    psi1 = (a_psi1 + b_psi1 * exp_gamma_psi1) / (1 + exp_gamma_psi1)

    exp_gamma_psi2 = exp(gamma_psi2)
    psi2 = (a_psi2 + b_psi2 * exp_gamma_psi2) / (1 + exp_gamma_psi2)

    #Transform p and q (Uniform transformation for p and q)
    exp_gamma_p = exp(gamma_p)
    p = (a_p + b_p * exp_gamma_p) / (1 + exp_gamma_p)
    exp_gamma_q = exp(gamma_q)
    q = (a_p + b_p * exp_gamma_q) / (1 + exp_gamma_q)

    # Transform variance parameters (Inverse-Gamma transformation)
    sigma_xi2 = exp(gamma_sigma_xi2)
    sigma_kappa2_0 = exp(gamma_sigma_kappa2_0)
    sigma_kappa2_1 = exp(gamma_sigma_kappa2_1)
    sigma_epsilon2_0 = exp(gamma_sigma_epsilon2_0)
    sigma_epsilon2_1 = exp(gamma_sigma_epsilon2_1)
    # sigma_y = exp(gamma_sigma_y)

    # transform omega (normal)
    # omega = gamma_omega

    #transform omega (uniform)
    exp_gamma_omega = exp(gamma_omega)
    omega = (a_omega + b_omega * exp_gamma_omega) / (1 + exp_gamma_omega)


    return (psi1, psi2, sigma_xi2, sigma_kappa2_0, sigma_kappa2_1, sigma_epsilon2_0, sigma_epsilon2_1, omega, p, q)

end


function log_derivatives_params(params_unbounded, priors)
    # Unpack parameters
    gamma_psi1, gamma_psi2, gamma_sigma_xi2, gamma_sigma_kappa2_0, gamma_sigma_kappa2_1, gamma_sigma_epsilon2_0, gamma_sigma_epsilon2_1, gamma_omega, gamma_p, gamma_q = params_unbounded
    a_psi1, b_psi1, a_psi2, b_psi2, a_xi, b_xi, a_kappa_0, b_kappa_0, a_kappa_1, b_kappa_1, a_epsilon_0, b_epsilon_0, a_epsilon_1, b_epsilon_1, a_omega, b_omega, a_p, b_p, a_q, b_q = priors
    
    # Log derivative for psi
    log_derivative_psi1 = log(b_psi1 - a_psi1) + gamma_psi1 - 2 * log(1 + exp(gamma_psi1))
    log_derivative_psi2 = log(b_psi2 - a_psi2) + gamma_psi2 - 2 * log(1 + exp(gamma_psi2))

    # Log derivatives for p and q
    log_derivative_p = log(b_p - a_p) + gamma_p - 2 * log(1 + exp(gamma_p))
    log_derivative_q = log(b_p - a_p) + gamma_q - 2 * log(1 + exp(gamma_q))

    # Log derivatives for variance parameters
    log_derivative_sigma_xi2 = gamma_sigma_xi2
    log_derivative_sigma_kappa2_0 = gamma_sigma_kappa2_0
    log_derivative_sigma_kappa2_1 = gamma_sigma_kappa2_1
    log_derivative_sigma_epsilon2_0 = gamma_sigma_epsilon2_0
    log_derivative_sigma_epsilon2_1 = gamma_sigma_epsilon2_1
    # log_derivative_sigma_y = gamma_sigma_y

    # Log derivative for omega
    # log_derivative_omega = 0
    log_derivative_omega = log(b_omega - a_omega) + gamma_omega - 2 * log(1 + exp(gamma_omega))
 

    return (log_derivative_psi1, log_derivative_psi2, log_derivative_sigma_xi2, log_derivative_sigma_kappa2_0, log_derivative_sigma_kappa2_1, log_derivative_sigma_epsilon2_0, log_derivative_sigma_epsilon2_1, log_derivative_omega, log_derivative_p, log_derivative_q)
    
end




function log_prior(theta, priors)
    # Unpack parameters
    psi1, psi2, sigma_xi2, sigma_kappa2_0, sigma_kappa2_1, sigma_epsilon2_0, sigma_epsilon2_1, omega, p, q = theta
    a_psi1, b_psi1, a_psi2, b_psi2, a_xi, b_xi, a_kappa_0, b_kappa_0, a_kappa_1, b_kappa_1, a_epsilon_0, b_epsilon_0, a_epsilon_1, b_epsilon_1, a_omega, b_omega, a_p, b_p, a_q, b_q = priors

    # Log prior psi1
    log_prior_psi1 = log(1/(b_psi1 - a_psi1))

    # Log prior psi2
    log_prior_psi2 = log(1/(b_psi2 - a_psi2))

    # Log prior p and q (uniform 0 to 1)
    log_prior_p = log(1 / (b_p - a_p))
    log_prior_q = log(1 / (b_q - a_q))

    # Inverse-Gamma priors for variance parameters
    log_prior_sigma_xi2 = a_xi* log(b_xi) - lgamma(a_xi) - (a_xi + 1) * log(sigma_xi2) - b_xi / sigma_xi2
    log_prior_sigma_kappa2_0 = a_kappa_0 * log(b_kappa_0) - lgamma(a_kappa_0) - (a_kappa_0 + 1) * log(sigma_kappa2_0) - b_kappa_0 / sigma_kappa2_0
    log_prior_sigma_kappa2_1 = a_kappa_1 * log(b_kappa_1) - lgamma(a_kappa_1) - (a_kappa_1 + 1) * log(sigma_kappa2_1) - b_kappa_1 / sigma_kappa2_1
    log_prior_sigma_epsilon2_0 = a_epsilon_0 * log(b_epsilon_0) - lgamma(a_epsilon_0) - (a_epsilon_0 + 1) * log(sigma_epsilon2_0) - b_epsilon_0 / sigma_epsilon2_0
    log_prior_sigma_epsilon2_1 = a_epsilon_1 * log(b_epsilon_1) - lgamma(a_epsilon_1) - (a_epsilon_1 + 1) * log(sigma_epsilon2_1) - b_epsilon_1 / sigma_epsilon2_1
    # log_prior_sigma_y = a_sigma * log(b_sigma) - lgamma(a_sigma) - (a_sigma + 1) * log(sigma_y) - b_sigma / sigma_y
   
    # log prior omega (normal mean a_omea, var b_omega)
    # log_prior_omega = -0.5 * log(2 * π * b_omega) - 0.5 * (omega - a_omega)^2 / b_omega
    #Log prior omega (uniform)
    log_prior_omega = log(1 / (b_omega - a_omega))

    return log_prior_psi1 + log_prior_psi2 + log_prior_sigma_xi2 + log_prior_sigma_kappa2_0 + log_prior_sigma_kappa2_1 + log_prior_sigma_epsilon2_0 + log_prior_sigma_epsilon2_1 + log_prior_omega + log_prior_p + log_prior_q 
    
end




function log_posterior(gamma, priors, y, a1, P1)
    theta = transform_params(gamma, priors)
    
    log_lik, a_est, p_res = switching_kalman_filter(y, theta, a1, P1)
 
    log_pri = log_prior(theta, priors)

    if log_pri == -Inf || isnan(log_lik)
        return -Inf
    end

    log_jacobian = sum(log_derivatives_params(gamma, priors))

    return log_lik + log_pri + log_jacobian
end



function initialize_mcmc(y, priors, a1, P1; n_init=40000, burn_init=5000, omega_init=0.1)
    dim = length(priors)/2  # Number of parameters
    dim = Int(dim)
    

    # Initialize storage for the chain
    chain_init = zeros(n_init, dim)
    current_gamma = zeros(dim)  # Initial value for gamma
    current_log_post = log_posterior(current_gamma, priors, y, a1, P1)
    theta_samples_init = zeros(n_init, dim)

    accept_init = 0

    pb = Progress(n_init; desc="Initialization Phase")

    for s in 1:n_init
        # Propose Gamma*
        gamma_star = rand(MvNormal(current_gamma, omega_init * I(dim)))

        # Compute log posterior
        log_post_star = log_posterior(gamma_star, priors, y, a1, P1)

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

        theta_samples_init[s,:] = collect(transform_params(current_gamma, priors))

        next!(pb)
    end

    acceptance_rate_init = accept_init / n_init
    println("Initialization Acceptance Rate: $(acceptance_rate_init * 100) %")

    # Discard burn-in
    chain_init_burned = chain_init[burn_init+1:end, :]
    theta_samples_init = theta_samples_init[burn_init+1:end,:]

    # Compute covariance matrix from initialization
    Sigma = cov(chain_init_burned)



    return chain_init_burned, Sigma, acceptance_rate_init, chain_init, theta_samples_init
end




function recursion_mcmc(y, priors, a1, P1, chain_init_burned, Sigma; n_rec=20000, burn_rec=10000, omega_rec=0.1)
    dim = length(priors)/2  # Number of parameters
    dim = Int(dim)


    # Initialize storage for the recursion chain
    chain_rec = zeros(n_rec, dim)
    theta_samples = zeros(n_rec, dim)

    # Start from the last point of the initialization chain
    current_gamma_rec = copy(chain_init_burned[end, :])
    current_log_post_rec = log_posterior(current_gamma_rec,priors, y, a1, P1)
    
    accept_rec = 0

    # Placeholder for alphas
    state_dim = 4
    alphas = []
    p_res_list = []
    

    pb = Progress(n_rec; desc="Recursion Phase")

    for q in 1:n_rec
        # Propose Gamma*
        gamma_star = rand(MvNormal(current_gamma_rec, omega_rec * Sigma))
        # gamma_star = rand(MvNormal(current_gamma_rec, omega_rec * I(dim)))
        
        # Compute log posterior
        log_post_star = log_posterior(gamma_star, priors, y, a1, P1)
        
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

            # # Sample states using Gibbs sampling
            # # Step 1: Draw alpha^+
            # alpha_plus_draw = rand(MvNormal(zeros(state_dim), P1))
            
            # # Step 2: Simulate forward recursion to get y^+
            # y_plus, alpha_plus = simulate_data(theta, n_order, length(y), alpha_plus_draw)
            
            # # Step 3: Construct y*
            # y_star = y - y_plus
            
            # # Step 4: Compute E(alpha | y*)
            # _, a_smooth = switching_kalman_filter(y_star, theta, n_order, a1, P1)
            # alpha_hat = a_smooth[1:end-1, :]  # Exclude the last state
            
            # # Step 5: Draw alpha^*
            # alpha_star = alpha_hat .+ alpha_plus[1:end-1, :]  # Adjusted state
            #sample states from filters
            _, alpha_star, p_res = switching_kalman_filter(y, theta, a1, P1)
            
            # Store the sampled alpha
            push!(alphas, alpha_star)
            push!(p_res_list, p_res)
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

    return theta_samples_burned, alpha_samples, acceptance_rate_rec, p_res_list
end

end

