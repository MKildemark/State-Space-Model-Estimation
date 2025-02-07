module MCMC

export mcmc_initialization

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

function bounded_above_and_below(Γ, a, b)
    θ = (a + b*exp(Γ)) / (1 + exp(Γ))
    log_jac = log(b - a) + Γ - 2*log(1 + exp(Γ))
    return θ, log_jac
end

function bounded_below(Γ, a)
    θ = exp(Γ) + a
    log_jac = Γ
    return θ, log_jac
end

function unbounded(Γ)
    θ = Γ
    log_jac = 0
    return θ, log_jac
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

function transform(Γ, support)
    θ = zeros(length(Γ))
    log_jac = zeros(length(Γ))
    for i in eachindex(Γ)
        if support[i,1] == -Inf && support[i,2] == Inf
            θ[i], log_jac = unbounded(Γ[i])
        elseif support[i,1] == 0 && support[i,2] == Inf
            θ[i], log_jac = bounded_below(Γ[i], support[i,1])
        else
            θ[i], log_jac = bounded_above_and_below(Γ[i], support[i,1], support[i,2])
        end
    end
    return θ, log_jac
end


#########################
# Get Priors
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
# Sample Parameters and States
#########################
   
function log_posterior(gamma, priors, y, a1, P1, cycle_order)
    theta = transform_params(gamma, priors)
    log_lik = log_likelihood(theta, y, a1, P1, cycle_order)
    log_pri = log_prior(theta, priors)
    if log_pri == -Inf || isnan(log_lik)
        return -Inf
    end
    log_jacobian = sum(log_derivatives_params(gamma, priors))
    return log_lik + log_jacobian + log_pri
end



function mcmc_initialization(y, prior_distributions, prior_parameters, support, cycle_order = 2, n_iter = 10000, n_burn = 5000, ω = 0.01)
    
    # initialize
    n_params = length(prior_distributions)
    Γ0 = zeros(n_params)
    θ0, jac0 = transform(Γ0, support)
    Z, H, T, R, Q = state_space(θ0, cycle_order)
    state_dim = size(T, 1)
    α0 = zeros(state_dim)
    P0 = 1e6 * Matrix(I, state_dim, state_dim)
    
    Γ_draws = zeros(n_iter, n_params)
    θ_draws = zeros(n_iter, n_params)

    Γ_current = Γ0
    θ_current = θ0
    jac_current = jac0

    # Run kalman to get initial likelihood
    LogL, α_f, P_f, α_p, P_p = kalman_filter(y, θ_current,  α0 , P0, cycle_order)

    log_prior = priors(θ_current, prior_distributions, prior_parameters)

    log_posterior_current = LogL + sum(log_prior) + sum(jac_current)

    accepted = 0  # counter for accepted proposals

    Σ = ω * Matrix(I, n_params, n_params)

    # MCMC
    @showprogress for i in 1:n_iter
        Γ_proposal = rand(MvNormal(Γ_current, Σ))
        θ_proposal, jac_proposal = transform(Γ_proposal, support)
        log_prior = priors(θ_proposal, prior_distributions, prior_parameters)
        LogL, α_f, P_f, α_p, P_p = kalman_filter(y, θ_proposal,  α0 , P0, cycle_order)
        log_posterior_proposal = LogL + sum(log_prior) + sum(jac_proposal)

        if log(rand()) < log_posterior_proposal - log_posterior_current
            Γ_current = Γ_proposal
            θ_current = θ_proposal
            log_posterior_current = log_posterior_proposal
            accepted += 1
        end

        Γ_draws[i, :] = Γ_current
        θ_draws[i, :] = θ_current
    end

    Σ_Γ = cov(Γ_draws[n_burn+1:end, :])
    accept_rate = accepted/n_iter

    return θ_draws, Γ_draws, Σ_Γ, accept_rate

end


end

    

    
    




