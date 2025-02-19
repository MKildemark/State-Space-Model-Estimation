module ParticleFilter

export particle_filter

using Random
using LinearAlgebra
using Statistics
using Distributions

include("State_Space_Model.jl")
using .state_space_model


function systematic_resample(weights)
    N = length(weights)
    positions = ((0:N-1) .+ rand()) / N
    cumulative_sum = cumsum(weights)
    indices = Vector{Int}(undef, N)
    j = 1
    for i in 1:N
        while positions[i] > cumulative_sum[j]
            j += 1
        end
        indices[i] = j
    end
    return indices
end



function particle_filter(y, θ, a1, P1, cycle_order, σy; N_particles=1000)
    # Retrieve the state-space matrices from your model.
    Z, H, T_mat, R_mat, Q, _ = state_space(θ, cycle_order, σy)
    state_dim = size(T_mat, 1)
    n_obs = size(y, 2)
    shock_dim = size(Q, 1)

    # Initialize particles at initial states.
    particles = zeros(state_dim, N_particles)
    filtered_states = zeros(state_dim, n_obs)

    for i in 1:N_particles
        particles[:, i] = a1 #+ rand(MvNormal(zeros(state_dim), P1))
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
    
        # Resample particles and reset weights
        indexResampled = systematic_resample(weights)
        particles = particles[:, indexResampled]
        weights .= 1.0 / N_particles
    end
   

    return logL, filtered_states
end

end  # module ParticleFilter
