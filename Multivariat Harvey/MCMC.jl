module MCMC

export MCMC_estimation

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
    log_p = -log(sqrt(2*π*b)) - ((θ - a)^2) / (2*b) 
    return log_p
end

function inverse_gamma(θ, a, b)
    # a is shape and b is scale
    log_p = a*log(b) - lgamma(a) - (a+1)*log(θ) - b/θ
    return log_p
end

function beta(θ, a, b)
    # a is shape1 and b is shape2; support: (0,1)
    log_p = (a-1)*log(θ) + (b-1)*log(1-θ) - lgamma(a) - lgamma(b) + lgamma(a+b)
    return log_p
end

function uniform(θ, a, b)
    # bounded between a and b
    log_p = -log(b - a)
    return log_p
end

#########################
# Transform unbounded parameters Γ to bounded θ
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
# Transform bounded parameters θ to unbounded Γ (Used for θ₀)
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
# Log Posterior
#########################

function log_posterior(θ, log_jac, prior_info, y, a1, P1, cycle_order)
    # The likelihood is computed in a state-space context (using the Kalman filter)
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
# MCMC Estimation
#########################

function MCMC_estimation(y, prior_info, a1, P1, cycle_order;
              iter_init = 40000,
              burn_init = 5000,
              iter_rec = 10000,
              burn_rec = 5000,
              θ_init = zeros(size(prior_info.support, 1)),
              ω = 0.1,
              adapt_interval = 100,
              target_low = 0.25,
              target_high = 0.35)

    #############################
    # Initialization Phase 
    #############################
    dim = size(prior_info.support, 1)
    θ_init_chain = zeros(iter_init, dim)
    Γ_chain = zeros(iter_init, dim)

    # Initialize in unbounded space
    Γ_current = transform_to_unbounded(θ_init, prior_info.support)
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, a1, P1, cycle_order)
    accept_init_total = 0
    block_accept_count = 0

    pb = Progress(iter_init; desc="Initialization Phase")
    for s in 1:iter_init
        # Propose new Γ using a random walk with covariance ω * I
        Γ_star = rand(MvNormal(Γ_current, ω * I(dim)))
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, a1, P1, cycle_order)

        # Metropolis acceptance step
        η = min(1, exp(log_post_star - current_log_post))
        if rand() < η
            θ_init_chain[s, :] = θ_star
            Γ_chain[s, :] = Γ_star
            Γ_current = Γ_star
            θ_current = θ_star
            log_jac_current = log_jac_star
            current_log_post = log_post_star
            accept_init_total += 1
            block_accept_count += 1
        else
            θ_init_chain[s, :] = θ_current
            Γ_chain[s, :] = Γ_current
        end

        # Adapt ω every adapt_interval iterations
        if mod(s, adapt_interval) == 0
            block_accept_rate = block_accept_count / adapt_interval
            if block_accept_rate < target_low
                ω *= 0.9  # proposals too large → lower ω to increase acceptance
            elseif block_accept_rate > target_high
                ω *= 1.1  # proposals too small → increase ω to explore more
            end
            block_accept_count = 0  # reset counter for next block
        end
        next!(pb)
    end

    init_accept_rate = accept_init_total / iter_init
    println("Initialization Acceptance Rate: $(init_accept_rate * 100)%")
    
    # Use the draws after burn-in to compute an empirical covariance for the unbounded parameters.
    Σ_emp = cov(Γ_chain[burn_init+1:end, :])

    #############################
    # Recursion Phase
    #############################
    # Use the final draw from initialization as the starting value.
    Γ_current = Γ_chain[end, :]
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, a1, P1, cycle_order)

    n_params = dim
    n_obs = size(y, 1)
    state_dim = size(P1, 1)

    θ_chain = zeros(iter_rec, n_params)
    α_draws = zeros(iter_rec, n_obs, state_dim)
    accept_rec_total = 0
    block_accept_count = 0  # for adaptive updates during recursion burn-in

    pb = Progress(iter_rec; desc="Recursion Phase")
    for s in 1:iter_rec
        # Propose new Γ using the tuned covariance ω * Σ_emp
        Γ_star = rand(MvNormal(Γ_current, ω * Σ_emp))
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, a1, P1, cycle_order)
        log_accept_ratio = log_post_star - current_log_post

        if log_accept_ratio >= 0 || (log_accept_ratio > log(rand()))
            Γ_current = Γ_star
            θ_current = θ_star
            log_jac_current = log_jac_star
            current_log_post = log_post_star
            accept_rec_total += 1
            block_accept_count += 1
        end

        θ_chain[s, :] = θ_current

        # After burn-in, draw state trajectories using the simulation smoother.
        if s > burn_rec
            _, α_samp, _ = diffuse_kalman_filter(
                y, θ_current, a1, P1, cycle_order,
                true, true
            )
            α_draws[s, :, :] = α_samp
        end

        # Adapt ω every adapt_interval iterations.
        if mod(s, adapt_interval) == 0
            block_accept_rate = block_accept_count / adapt_interval
            if block_accept_rate < target_low
                ω *= 0.9
            elseif block_accept_rate > target_high
                ω *= 1.1
            end
            block_accept_count = 0
        end
        next!(pb)
    end

    rec_accept_rate = accept_rec_total / iter_rec
    println("Recursion Acceptance Rate: $(rec_accept_rate * 100)%")

    # Retain only the draws after burn-in for the recursion phase.
    θ_chain_post = θ_chain[burn_rec+1:end, :]
    α_draws_post = α_draws[burn_rec+1:end, :, :]

    return θ_chain_post, θ_init_chain, α_draws_post
end


end  # module MCMC
