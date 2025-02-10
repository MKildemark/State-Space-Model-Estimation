module MCMC

export initialize_mcmc , mcmc_recursion

using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

include("State_Space_Model.jl")
using .state_space_model

include("Kalman.jl")
using .kalman

Random.seed!(123)


#########################
# Helpers to Transform parameters from unbounded to bounded support
#########################

function θ_bounded_above_and_below(Γ, a, b)
    θ = (a + b*exp(Γ)) / (1 + exp(Γ))
    log_jac = log(b - a) + Γ - 2*log(1 + exp(Γ))
    return θ, log_jac
end

function θ_bounded_below(Γ, a)
    θ = exp(Γ) + a
    log_jac = Γ
    return θ, log_jac
end

function θ_unbounded(Γ)
    θ = Γ
    log_jac = 0
    return θ, log_jac
end

#########################
# Helpers to Transform parameters from bounded to unbounded support (used for initial guess of θ)
#########################

function Γ_bounded_above_and_below(θ, a, b)
    Γ = log((θ - a) / (b - θ))
    return Γ
end

function Γ_bounded_below(θ, a)
    Γ = log(θ - a)
    return Γ
end

function Γ_unbounded(θ)
    Γ = θ
    return Γ
end


#########################
# Helpers for Prior Densities
#########################

function normal(θ, a, b)
    # a is mean and b is variance
    # unboundend
    log_p = -log(sqrt(2*π*b)) - ((θ - a)^2) / (2*b) 
    return log_p
end

function inverse_gamma(θ, a, b)
    # a is shape and b is scale
    # non-negative
    log_p = a*log(b) - lgamma(a) - (a+1)*log(θ) - b/θ
    return log_p
end

function beta(θ, a, b)
    # a is shape1 and b is shape2
    # bounded between 0 and 1
    log_p = (a-1)*log(θ) + (b-1)*log(1-θ) - lgamma(a) - lgamma(b) + lgamma(a+b)
    return log_p
end

function uniform(θ, a, b)
    # bounded between a and b
    log_p = -log(b - a)
    return log_p
end




#########################
# Tranform unbounded parameters Γ to bounded θ
#########################

function transform_to_bounded(Γ, support)
    θ = zeros(length(Γ))
    log_jac = zeros(length(Γ))
    for i in eachindex(Γ)
        if support[i,1] == -Inf && support[i,2] == Inf
            θ[i], log_jac[i] = θ_unbounded(Γ[i])
        elseif support[i,1] == 0 && support[i,2] == Inf
            θ[i], log_jac[i] = θ_bounded_below(Γ[i], support[i,1])
        else
            θ[i], log_jac[i] = θ_bounded_above_and_below(Γ[i], support[i,1], support[i,2])
        end
    end
    return θ, log_jac
end

#########################
# Tranform bounded parameters θ to unbounded Γ (Used for θ0)
#########################

function transform_to_unbounded(θ, support)
    Γ = zeros(length(θ))
    for i in eachindex(θ)
        if support[i,1] == -Inf && support[i,2] == Inf
            Γ[i] = Γ_unbounded(θ[i])
        elseif support[i,1] == 0 && support[i,2] == Inf
            Γ[i] = Γ_bounded_below(θ[i], support[i,1])
        else
            Γ[i] = Γ_bounded_above_and_below(θ[i], support[i,1], support[i,2])
        end
    end
    return Γ
    
end


#########################
# Priors
#########################

function priors(θ, prior_distributions, prior_parameters)
    log_p = zeros(length(θ))
    for i in eachindex(θ)
        if prior_distributions[i] == "normal"
            log_p[i] = normal(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "inverse_gamma"
            log_p[i] = inverse_gamma(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "beta"
            log_p[i] = beta(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "uniform"
            log_p[i] = uniform(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        end
    end
    return log_p
end



#########################
# Log Posterir
#########################

function log_posterior(θ, log_jac, prior_info, y, a1, P1, cycle_order)
    log_lik = -neg_log_likelihood(θ, y, a1, P1, cycle_order)
    log_prior = priors(θ, prior_info.distributions, prior_info.parameters)
    log_pri = sum(log_prior)
    if log_pri == -Inf || isnan(log_lik)
        return -Inf
    end
    log_jacobian = sum(log_jac)
    return log_lik + log_jacobian + log_pri
end


#########################
# MCMC Initialization
#########################

function initialize_mcmc(y, prior_info, a1, P1, θ_init, cycle_order; iter =40000, burn=5000, ω=0.1)
    # The number of parameters
    dim = size(prior_info.support, 1)
    
    # Storage for parameter draws
    θ = zeros(iter, dim)
    Γ = zeros(iter, dim)

    # Initialize the chain
    Γ_current = transform_to_unbounded(θ_init, prior_info.support)
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, a1, P1, cycle_order)
    accept = 0

    pb = Progress(iter; desc="Initialization Phase")
    for s in 1:iter
        # Propose a new  Γ using a random-walk Gaussian proposal.
        Γ_star = rand(MvNormal(Γ_current, ω * I(dim)))
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, a1, P1, cycle_order)
        # Compute acceptance probability (using the usual Metropolis–Hastings ratio).
        η = min(1, exp(log_post_star - current_log_post))
        if rand() < η
            θ[s, :] = θ_star
            Γ[s, :] = Γ_star
            θ_current = θ_star
            Γ_current = Γ_star
            current_log_post = log_post_star
            accept += 1
        else
            θ[s, :] = θ_current
            Γ[s, :] = Γ_current
            
        end


        next!(pb)
    end

    acceptance_rate = accept / iter
    println("Initialization Acceptance Rate: $(acceptance_rate * 100) %")
  
    # Compute empirical covariance.
    Σ = cov(Γ[burn+1:end, :])
    return θ, Σ, acceptance_rate
end


#########################
# MCMC Recursion
#########################

function mcmc_recursion(y,prior_info,a1,P1,cycle_order;
    iter = 40000,
    burn = 5000,
    θ_init = zeros(size(prior_info.support, 1)),
    Σ = I(size(prior_info.support, 1)),
    ω = 0.1
)

    # Number of parameters
    n_params = size(prior_info.support, 1)
    #Number of observation
    n_obs = size(y, 1)
    #Number of states
    state_dim = size(P1, 1)

    # Storage for parameter and state draws
    θ_chain = zeros(iter, n_params)
    α_draws = zeros(iter, n_obs, state_dim)

    # Current (unbounded) parameter
    Γ_current = transform_to_unbounded(θ_init, prior_info.support)

    # Transform to bounded space & compute log posterior at initial
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)

    # Current posterior
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, a1, P1, cycle_order)

    accept = 0


    pb = Progress(iter; desc="Recursion Phase")
    for s in 1:iter
        # 1) Propose Γ_star using random-walk with covariance ω * Σ
        Γ_star = rand(MvNormal(Γ_current, ω * I(n_params)))

        # 2) Transform to bounded space
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)

        # 3) Compute log posterior for proposal
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, a1, P1, cycle_order)

        # 4) Metropolis acceptance step
        # Ratio (in log scale) = log_post_star - current_log_post
        log_accept_ratio = log_post_star - current_log_post
        if log_accept_ratio >= 0 || (log_accept_ratio > log(rand()))
            # Accept
            Γ_current = Γ_star
            θ_current = θ_star
            log_jac_current = log_jac_star
            current_log_post = log_post_star
            accept += 1
        end

        # Store the accepted (θ_current) in the chain
        θ_chain[s, :] = θ_current

        # 5) After burn-in, draw states alpha using the simulation smoother
        if s > burn
            _, α_samp, _ = diffuse_kalman_filter(
                y, θ_current, a1, P1, cycle_order,
                true, true
            )
            # α_samp is an (n_obs × state_dim) sample
            α_draws[s,:,:] = α_samp
        end

        next!(pb)
    end

    α_draws =  α_draws[burn+1:end, :, :]

    # Acceptance rate
    accept_rate = accept / iter
    println("Recursion Acceptance Rate: $(accept_rate * 100) %")

    return θ_chain, α_draws, accept_rate
end



end  # module MCMCRoutines


