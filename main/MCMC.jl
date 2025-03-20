module MCMC

export MCMC_estimation, neg_log_likelihood, MCMC_estimation_parallel

include("import_packages.jl")


using Main.state_space_model
using Main.kalman
using Main.ParticleFilter

# Remove or comment out the global seed if you use thread-local RNGs.
# Random.seed!(123)

#########################
#  Helper Check Positive Definite
#########################
function positive_definite_check(θ, obs_dim)
    if obs_dim == 1
        return true
    else
        Z, H, d, T, R, Q, c, P_diffuse = get_state_space(θ)
        return isposdef(H) && isposdef(Q)
    end
end

#########################
# Helpers to Transform Parameters from Unbounded to Bounded
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
# Helpers to Transform Parameters from Bounded to Unbounded (for initial guess)
#########################
function Γ_bounded_above_and_below(θ, a, b)
    return log((θ - a) / (b - θ))
end

function Γ_bounded_below(θ, a)
    return log(θ - a)
end

function Γ_unbounded(θ)
    return θ
end

#########################
# Helpers for Prior Densities
#########################
function normal(θ, a, b)
    -log(sqrt(2*π*b)) - ((θ - a)^2) / (2*b)
end

function inverse_gamma(θ, a, b)
    if θ < 0
        return -Inf
    end
    a*log(b) - lgamma(a) - (a+1)*log(θ) - b/θ
end

function beta(θ, a, b)
    if θ < 0 || θ > 1
        return -Inf
    end
    (a-1)*log(θ) + (b-1)*log(1-θ) - lgamma(a) - lgamma(b) + lgamma(a+b)
end

function uniform(θ, a, b)
    if θ < a || θ > b
        return -Inf
    end
    -log(b - a)
end

#########################
# Transformations Between Bounded and Unbounded
#########################
function transform_to_bounded(Γ, support)
    θ = zeros(length(Γ))
    log_jac = zeros(length(Γ))
    for i in eachindex(Γ)
        if support[i,1] == -Inf && support[i,2] == Inf
            θ[i], log_jac[i] = θ_unbounded(Γ[i])
        elseif isfinite(support[i,1]) && support[i,2] == Inf
            θ[i], log_jac[i] = θ_bounded_below(Γ[i], support[i,1])
        else
            θ[i], log_jac[i] = θ_bounded_above_and_below(Γ[i], support[i,1], support[i,2])
        end
    end
    return θ, log_jac
end

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

########################
# Negative Likelihood 
########################
function neg_log_likelihood(θ, y, X, a1, P1; filter_type="kalman", N_particles=1000, rng=Random.GLOBAL_RNG)
    if filter_type == "kalman"
        LogL, _, _ = diffuse_kalman_filter(y, X, θ, a1, P1; do_smooth = false, do_sim_smooth = false, rng=rng)
        return -LogL
    elseif filter_type == "particle"
        logL, _ = ParticleFilter.particle_filter(y, θ, a1, P1; N_particles=N_particles)
        return -logL
    else
        error("Unknown filter type: $filter_type")
    end
end

#########################
# Log Posterior
#########################
function log_posterior(θ,  log_jac, prior_info, y, X, a1, P1; filter_type="kalman", N_particles=1000, rng=Random.GLOBAL_RNG)
    log_lik = -neg_log_likelihood(θ, y, X, a1, P1; filter_type=filter_type, N_particles=N_particles, rng=rng)
    log_pri = priors(θ, prior_info.distributions, prior_info.parameters)
    log_jacobian = sum(log_jac)
    return log_lik + log_jacobian + log_pri
end

#########################
# Priors and Sampling
#########################
function priors(θ, prior_distributions, prior_parameters)
    logp = 0
    for i in eachindex(θ)
        if prior_distributions[i] == "normal"
            p = normal(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "inverse_gamma"
            p = inverse_gamma(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "beta"
            p = beta(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        elseif prior_distributions[i] == "uniform"
            p = uniform(θ[i], prior_parameters[i, 1], prior_parameters[i, 2])
        end
        logp += p
    end
    return logp
end

function sample_from_prior(prior_info; rng=Random.GLOBAL_RNG)
    n_params = length(prior_info.distributions)
    θ = zeros(n_params)
    for i in 1:n_params
        dist_type = prior_info.distributions[i]
        a = prior_info.parameters[i, 1]
        b = prior_info.parameters[i, 2]
        if dist_type == "normal"
            θ[i] = rand(rng, Normal(a, b))
        elseif dist_type == "inverse_gamma"
            # θ[i] = rand(rng, InverseGamma(a, b))
            θ[i] = rand(rng, Uniform(0.0, 5.0)) # sample from inverse gamma will give inf due to infinite right tail and create problems
        elseif dist_type == "beta"
            θ[i] = rand(rng, Beta(a, b))
        elseif dist_type == "uniform"
            θ[i] = rand(rng, Uniform(a, b))
        else
            error("Unknown prior distribution: $dist_type")
        end
    end
    return θ
end
#########################
# MCMC Estimation (Sequential)
#########################
function MCMC_estimation(y, X, prior_info_collection, a1, P1;
    filter_type="kalman",
    iter_init = 20000,
    burn_init = 10000,
    iter_rec = 20000,
    burn_rec = 15000,
    N_particles = 1000,
    ω = 0.01,
    adapt_interval = 50,
    target_low = 0.25,
    target_high = 0.35,
    n_chains = 1,
    θ_0 = nothing,
    forecast_in_sample = 0,
    forecast_out_sample = 0,
    X_forecast = nothing,
    rng=Random.GLOBAL_RNG)

dim = size(prior_info_collection[1].support, 1)
obs_dim = size(y, 1)

θ_init_chain_all = zeros(iter_init, dim, n_chains)
θ_chain_all       = zeros(iter_rec, dim, n_chains)
state_dim = size(P1, 1)
# Allocate a vector (per chain) to store the state draws as matrices.
α_draws_all = Vector{Vector{Matrix{Float64}}}(undef, n_chains)

for chain in 1:n_chains
println("Starting chain $chain ...")
chain_start_time = time()  # <-- Timing start for this chain
prior_info = prior_info_collection[chain]

# Initialization Phase
if θ_0 !== nothing
  θ_start = θ_0
else
  θ_start = sample_from_prior(prior_info; rng=rng)
end
println("Chain $chain, Initial Parameters: $θ_start")
Γ_current = transform_to_unbounded(θ_start, prior_info.support)
θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, X, a1, P1;
                               filter_type=filter_type, N_particles=N_particles, rng=rng)

θ_init_chain = zeros(iter_init, dim)
Γ_chain      = zeros(iter_init, dim)
accept_init_total = 0
block_accept_count = 0
accept_target = (target_high+target_low)/2

pb_init = Progress(iter_init; desc="Initialization Phase (chain $chain)")
for s in 1:iter_init
  Γ_star = rand(rng, MvNormal(Γ_current, ω * I(dim)))
  θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
  log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, X, a1, P1;
                                filter_type=filter_type, N_particles=N_particles, rng=rng)
  log_accept_ratio = log_post_star - current_log_post
  if log_accept_ratio > log(rand(rng))
      Γ_current = Γ_star
      θ_current = θ_star
      log_jac_current = log_jac_star
      current_log_post = log_post_star
      accept_init_total += 1
      block_accept_count += 1
  end

  θ_init_chain[s, :] = θ_current
  Γ_chain[s, :]      = Γ_current 

  if mod(s, adapt_interval) == 0 
      block_accept_rate = block_accept_count / adapt_interval
      ω *= exp((block_accept_rate - accept_target))
      block_accept_count = 0
  end
  next!(pb_init)
end

init_accept_rate = accept_init_total / iter_init
println("Chain $chain, Initialization Acceptance Rate: $(init_accept_rate * 100)%")
θ_init_chain_all[:, :, chain] = θ_init_chain

Σ_emp = cov(Γ_chain[burn_init+1:end, :])

# Recursion Phase
Γ_current = Γ_chain[end, :]
θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, X, a1, P1;
                               filter_type=filter_type, N_particles=N_particles, rng=rng)

θ_chain = zeros(iter_rec, dim)
# Allocate a vector of matrices; each element will store one draw from the Kalman filter.
α_draws = Vector{Matrix{Float64}}(undef, iter_rec)
accept_rec_total = 0
block_accept_count = 0
ω_rec = ω * Σ_emp

pb_rec = Progress(iter_rec; desc="Recursion Phase (chain $chain)")
for s in 1:iter_rec
  Γ_star = rand(rng, MvNormal(Γ_current, ω_rec))
  θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
  log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, X, a1, P1;
                                 filter_type=filter_type, N_particles=N_particles, rng=rng)
  log_accept_ratio = log_post_star - current_log_post
  if log_accept_ratio > log(rand(rng))
      Γ_current = Γ_star
      θ_current = θ_star
      log_jac_current = log_jac_star
      current_log_post = log_post_star
      accept_rec_total += 1
      block_accept_count += 1
  end

  θ_chain[s, :] = θ_current

  if s > burn_rec
      if filter_type == "kalman"
          _, α_samp, _ = diffuse_kalman_filter(
              y, X, θ_current, a1, P1; do_smooth=true, do_sim_smooth=true, 
              forecast_out_sample = forecast_out_sample, forecast_in_sample = forecast_in_sample, 
              X_forecast = X_forecast, rng=rng)
          α_draws[s] = α_samp
      # elseif filter_type == "particle"
      #     _, α_samp = ParticleFilter.particle_filter(
      #         y, θ_current, a1, P1; N_particles=N_particles)
      end 
  end

  if mod(s, adapt_interval) == 0 
      block_accept_rate = block_accept_count / adapt_interval
      ω_rec *= exp((block_accept_rate - accept_target))
      block_accept_count = 0
  end
  next!(pb_rec)
end

rec_accept_rate = accept_rec_total / iter_rec
println("Chain $chain, Recursion Acceptance Rate: $(rec_accept_rate * 100)%")

θ_chain_all[:, :, chain] = θ_chain
α_draws_all[chain] = α_draws

println("Chain $chain completed in $(time() - chain_start_time) seconds.")
end

# Apply burn-in for θ_chain_all for each chain
θ_chain_all = θ_chain_all[burn_rec+1:end, :, :]
# Apply burn-in for α_draws_all for each chain 
for chain in 1:n_chains
    α_draws_all[chain] = α_draws_all[chain][burn_rec+1:end]
end

return θ_chain_all, θ_init_chain_all, α_draws_all
end



#########################
# Helper Function: Run One Chain (for Parallel)
#########################
function run_chain(chain::Int, y, X, prior_info, a1, P1;
                   filter_type="kalman",
                   iter_init=20000,
                   burn_init=10000,
                   iter_rec=20000,
                   burn_rec=15000,
                   ω=0.01,
                   adapt_interval=50,
                   target_low=0.25,
                   target_high=0.35,
                    θ_0=nothing,
                   rng=Random.GLOBAL_RNG)
    chain_start_time = time()  # <-- Timing start for this chain
    dim = size(prior_info.support, 1)
    obs_dim = size(y, 1)
    state_dim = size(P1, 1)
    
    θ_init_chain = zeros(iter_init, dim)
    Γ_chain = zeros(iter_init, dim)
    
     # Initialization Phase
    if θ_0 !== nothing
        θ_start = θ_0
    else
        θ_start = sample_from_prior(prior_info; rng=rng)
    end
    Γ_current = transform_to_unbounded(θ_start, prior_info.support)
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, X, a1, P1;
                                     filter_type=filter_type, rng=rng)
    
    accept_init_total = 0
    block_accept_count = 0
    accept_target = (target_high + target_low) / 2

    for s in 1:iter_init
        Γ_star = rand(rng, MvNormal(Γ_current, ω * I(dim)))
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
        
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, X, a1, P1;
                                        filter_type=filter_type, rng=rng)
        log_accept_ratio = log_post_star - current_log_post
        if log_accept_ratio > log(rand(rng))
            Γ_current = Γ_star
            θ_current = θ_star
            log_jac_current = log_jac_star
            current_log_post = log_post_star
            accept_init_total += 1
            block_accept_count += 1
        end
    
        θ_init_chain[s, :] = θ_current
        Γ_chain[s, :] = Γ_current

        if mod(s, adapt_interval) == 0 
            block_accept_rate = block_accept_count / adapt_interval
            ω *= exp((block_accept_rate - accept_target))
            block_accept_count = 0
            println("Chain $chain, Iteration $s, ω: $ω", " Acceptance Rate: $(block_accept_rate * 100)%")
        end
    end

    Σ_emp = cov(Γ_chain[burn_init+1:end, :])


    Γ_current = Γ_chain[end, :]
    θ_current, log_jac_current = transform_to_bounded(Γ_current, prior_info.support)
    current_log_post = log_posterior(θ_current, log_jac_current, prior_info, y, X, a1, P1;
                                     filter_type=filter_type, rng=rng)

    θ_chain = zeros(iter_rec, dim)
    α_draws = zeros(iter_rec, state_dim, size(y,2))
    accept_rec_total = 0
    block_accept_count = 0
    ω_rec = ω * Σ_emp

    for s in 1:iter_rec
        Γ_star = rand(rng, MvNormal(Γ_current, ω_rec))
        θ_star, log_jac_star = transform_to_bounded(Γ_star, prior_info.support)
    
        log_post_star = log_posterior(θ_star, log_jac_star, prior_info, y, X, a1, P1;
                                        filter_type=filter_type, rng=rng)
        log_accept_ratio = log_post_star - current_log_post
        if log_accept_ratio > log(rand(rng))
            Γ_current = Γ_star
            θ_current = θ_star
            log_jac_current = log_jac_star
            current_log_post = log_post_star
            accept_rec_total += 1
            block_accept_count += 1
        end


        θ_chain[s, :] = θ_current

        if s > burn_rec
            if filter_type == "kalman"
                _, α_samp, _ = diffuse_kalman_filter(
                    y, X, θ_current, a1, P1; do_smooth = true, do_sim_smooth = true, rng=rng)
            # elseif filter_type == "particle"
            #     _, α_samp = ParticleFilter.particle_filter(
            #         y, θ_current, a1, P1; N_particles=1000)
            end
            α_draws[s, :, :] = α_samp
        end

        if mod(s, adapt_interval) == 0 
            block_accept_rate = block_accept_count / adapt_interval
            ω_rec *= exp((block_accept_rate - accept_target))
            block_accept_count = 0
            println("Chain $chain, Iteration $s, ω_rec: ", " Acceptance Rate: $(block_accept_rate * 100)%")
        end
    end

    θ_chain = θ_chain[burn_rec+1:end, :]
    α_draws = α_draws[burn_rec+1:end, :, :]

    # Print elapsed time for this chain (parallel)
    println("Chain $chain on thread $(threadid()) completed in $(time() - chain_start_time) seconds.")

    return (θ_chain = θ_chain, θ_init_chain = θ_init_chain, α_draws = α_draws)
end

#########################
# Parallel MCMC Estimation
#########################
function MCMC_estimation_parallel(y, X, prior_info_collection::Vector, a1, P1;
    filter_type="kalman",
    iter_init = 20000,
    burn_init = 10000,
    iter_rec = 20000,
    burn_rec = 15000,
    ω = 0.01,
    adapt_interval = 50,
    target_low = 0.25,
    target_high = 0.35,
    n_chains = length(prior_info_collection),
    θ_0 = nothing)
    
    dim = size(prior_info_collection[1].support, 1)
    state_dim = size(P1, 1)
    T = size(y,2)
    iter_rec_final = iter_rec - burn_rec

    θ_init_chain_all = zeros(iter_init, dim, n_chains)
    θ_chain_all       = zeros(iter_rec_final, dim, n_chains)
    α_draws_all       = zeros(iter_rec_final, state_dim, T, n_chains)

    Threads.@threads for chain in 1:n_chains
        local_rng = MersenneTwister(123 + chain)
        # Pass the chain-specific prior_info
        res = run_chain(chain, y, X, prior_info_collection[chain], a1, P1;
            filter_type=filter_type,
            iter_init=iter_init,
            burn_init=burn_init,
            iter_rec=iter_rec,
            burn_rec=burn_rec,
            ω=ω,
            adapt_interval=adapt_interval,
            target_low=target_low,
            target_high=target_high,
            θ_0=θ_0,
            rng=local_rng)
        
        θ_init_chain_all[:, :, chain] = res.θ_init_chain
        θ_chain_all[:, :, chain]       = res.θ_chain
        α_draws_all[:, :, :, chain]     = res.α_draws
    end

    return θ_chain_all, θ_init_chain_all, α_draws_all
end


end  # module MCMC
