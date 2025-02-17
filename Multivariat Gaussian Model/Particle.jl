module ParticleFilter

export particle_filter, particle_filter_all_states

using Random
using Distributions
using LinearAlgebra
using StatsBase


include("State_Space_Model.jl")
using .state_space_model

"""
    particle_filter(y, θ, a1, P1, cycle_order, σy; N_particles=1000)

Runs a particle filter for the univariate state-space model defined in
`state_space_model.state_space`. Here:

- `y` is the observed data (an array or 1×T matrix).
- `θ` is the parameter vector (e.g. [ρ, λ_c, σ_ε, σ_ξ, σ_κ]).
- `a1` and `P1` are the initial state mean and covariance.
- `cycle_order` is the number of cycle blocks in the state.
- `σy` is the scaling used in the measurement equation.
- `N_particles` is the number of particles.

It returns a tuple `(log_likelihood, filtered_state)` where `filtered_state`
is the matrix of filtered state estimates (each column corresponds to a time step).
"""
function particle_filter(y, θ, a1, P1, cycle_order, σy; N_particles=1000)
    # Get the system matrices from the state-space model.
    # state_space returns: (Z, H, T, R, Q)
    Z, H, T_mat, R, Q = state_space(θ, cycle_order, σy)
    
    # Dimensions
    T_total = size(y, 2)
    state_dim = length(a1)
    
    # Initialize particles: each column is one particle state vector.
    particles = zeros(state_dim, N_particles)
    for i in 1:N_particles
        particles[:, i] = a1 + rand(MvNormal(P1))
    end
    
    # Initialize weights and filtered state storage.
    weights = zeros(N_particles)
    filtered_state = zeros(state_dim, T_total)
    log_likelihood = 0.0
    
    # --- Time 1: Weight the initial particles ---
    for i in 1:N_particles
        # The observation is given by y[:,1] ~ N(Z * α, H).
        mean_obs = Z * particles[:, i]
        # Here we assume H is 1×1; for multivariate you could use MvNormal(mean_obs, H)
        weights[i] = pdf(Normal(mean_obs[1], H[1,1]), y[1,1])
    end
    wsum = sum(weights)
    if wsum == 0
        weights .= 1/N_particles
    else
        weights ./= wsum
    end
    log_likelihood += log(wsum / N_particles)
    filtered_state[:, 1] = particles * weights

    # --- Recursion: t = 2, …, T_total ---
    for t in 2:T_total
        # Resample particles based on previous weights:
        indices = sample(1:N_particles, Weights(weights), N_particles, replace=true)
        new_particles = zeros(size(particles))
        for i in 1:N_particles
            idx = indices[i]
            particle_prev = particles[:, idx]
            # Propagate the state:
            #   αₜ = T * αₜ₋₁ + R * ηₜ   with ηₜ ~ N(0, Q)
            new_particles[:, i] = T_mat * particle_prev + R * rand(MvNormal(zeros(size(Q,1)), Q))
        end
        particles = new_particles

        # Compute weights based on the measurement equation at time t.
        for i in 1:N_particles
            mean_obs = Z * particles[:, i]
            weights[i] = pdf(Normal(mean_obs[1], H[1,1]), y[1, t])
        end
        wsum = sum(weights)
        if wsum == 0
            weights .= 1/N_particles
        else
            weights ./= wsum
        end
        log_likelihood += log(wsum / N_particles)

        # Filtered state is the weighted average of particles.
        filtered_state[:, t] = particles * weights
    end

    return log_likelihood, filtered_state
end

end
