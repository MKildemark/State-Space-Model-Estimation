module ParticleFilter

export particle_filter

using Random
using LinearAlgebra
using Statistics
using Distributions
using StatsBase
using Revise  # auto reload

includet("State_Space_Model.jl")
using .state_space_model


function particle_filter(model, y, θ, a1, P1, σy; N_particles=1000)
    # Retrieve the state-space matrices from your model.
    Z, H, T_mat, R_mat, Q, _ = state_space(model, θ, σy)
    state_dim = size(T_mat, 1)
    n_obs = size(y, 2)
    shock_dim = size(Q, 1)

    # Initialize particles at initial states.
    particles = zeros(state_dim, N_particles)
    filtered_states = zeros(state_dim, n_obs)

    for i in 1:N_particles
        particles[:, i] = a1 + rand(MvNormal(zeros(state_dim), P1))
    end

    # Initialize uniform weights.
    weights = fill(1.0 / N_particles, N_particles)

    logL = 0.0  # Log-likelihood accumulator


    #Loop through observations
    for t in 1:n_obs
        y_t = y[:, t]
    
        log_weights = zeros(N_particles)
        # Propagate particles and get weights (in logs to prevent underflow)
        for i in 1:N_particles
            particles[:, i] = T_mat * particles[:, i] + R_mat * rand(MvNormal(zeros(shock_dim), Q))
            log_weights[i] = logpdf(MvNormal(Z * particles[:, i], H), y_t)
            # weights[i] = pdf(MvNormal(Z * particles[:, i], H), y_t)
        end
        
        # Normalize weights (taking logs into account)
        max_log_weight = maximum(log_weights)
        weights = exp.(log_weights .- max_log_weight)
        weight_sum = sum(weights)
        weights ./= weight_sum
    
        # Get filtered state estimate
        filtered_states[:, t] = particles * weights
    
        # Increment the log-likelihood.
        logL += max_log_weight + log(weight_sum / N_particles)
        # logL += log(weight_sum / N_particles)
    
        # Resample particles and reset weights
        # indexResampled = systematic_resample(weights)
        indexResampled = sample(1:N_particles, Weights(weights), N_particles)
        particles = particles[:, indexResampled]
        weights .= 1.0 / N_particles
    end
   

    return logL, filtered_states
end




# function particle_filter(model, y, θ, a1, P1, σy; N_particles=1000)
#     # Extract parameters from θ
#     ρ   = θ[1]
#     λ   = θ[2]
#     σ_ε = sqrt(θ[3])
#     σ_ξ = sqrt(θ[4])
#     σ_κ = sqrt(θ[5])

#     state_dim = 4
#     n_obs = size(y, 2)   # assume y is 1×T

#     # Initialize particles using a1 and P1
#     particles = zeros(state_dim, N_particles)
#     for i in 1:N_particles
#         particles[:, i] = a1 + rand(MvNormal(zeros(state_dim), P1))
#     end

#     # Initialize weights uniformly
#     weights = fill(1.0 / N_particles, N_particles)
#     # Allocate array for filtered state estimates
#     filtered_states = zeros(state_dim, n_obs)
#     logL = 0.0

#     # Precompute the rotation matrix for the cycle update
#     rot = [cos(λ) sin(λ); -sin(λ) cos(λ)]

#     for t in 1:n_obs
#         # Get the observation at time t
#         y_t = y[1, t]
#         log_weights = zeros(N_particles)

#         # Propagate each particle and compute its weight
#         for i in 1:N_particles
#             # Unpack current state of particle i
#             μ  = particles[1, i]
#             β  = particles[2, i]
#             ψ  = particles[3, i]
#             ψs = particles[4, i]

#             # Trend update
#             new_μ = μ + β
#             new_β = β + rand(Normal(0, σ_ξ))

#             # Cycle update
#             cycle_prev = [ψ, ψs]
#             noise_cycle = [rand(Normal(0, σ_κ)), rand(Normal(0, σ_κ))]
#             cycle_new = ρ * (rot * cycle_prev) + noise_cycle
#             new_ψ  = cycle_new[1]
#             new_ψs = cycle_new[2]

#             # Update particle state
#             particles[:, i] = [new_μ, new_β, new_ψ, new_ψs]

#             # Predicted observation (μ + ψ)
#             pred_y = new_μ + new_ψ
#             log_weights[i] = logpdf(Normal(pred_y, σ_ε), y_t)
#         end

#         # Normalize weights using the log-sum-exp trick
#         max_log_weight = maximum(log_weights)
#         weights = exp.(log_weights .- max_log_weight)
#         weight_sum = sum(weights)
#         weights ./= weight_sum

#         # Update log-likelihood
#         logL += max_log_weight + log(weight_sum / N_particles)

#         # Compute filtered state estimate (weighted average)
#         filtered_states[:, t] = particles * weights

#         # Resample particles (using multinomial resampling)
#         indices = sample(1:N_particles, Weights(weights), N_particles)
#         particles = particles[:, indices]
#         weights .= 1.0 / N_particles
#     end

#     return logL, filtered_states
# end

end  # module ParticleFilter
