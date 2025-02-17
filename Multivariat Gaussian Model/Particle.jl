module ParticleFilter

export particle_filter

using Random
using LinearAlgebra
using Statistics
using Distributions

include("State_Space_Model.jl")
using .state_space_model


# A helper function for systematic resampling.
function systematic_resample(weights::Vector{Float64})
    N = length(weights)
    positions = ((0:N-1) .+ rand()) / N
    cumulative_sum = cumsum(weights)
    indices = Vector{Int}(undef, N)
    i = 1
    j = 1
    while i ≤ N && j ≤ N
        if positions[i] < cumulative_sum[j]
            indices[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indices
end


function particle_filter(y, θ, a1, P1, cycle_order, σy; N_particles=1000)
    # Retrieve the state-space matrices from your model.
    Z, H, T_mat, R_mat, Q, _ = state_space(θ, cycle_order, σy)
    state_dim = size(T_mat, 1)
    T_obs = size(y, 2)

    # Initialize particles at time 1.
    particles = zeros(state_dim, N_particles)
    if norm(P1) < 1e-12
        for i in 1:N_particles
            particles[:, i] .= a1
        end
    else
        for i in 1:N_particles
            particles[:, i] = a1 + rand(MvNormal(zeros(state_dim), P1))
        end
    end

    # Initialize uniform weights.
    weights = fill(1.0 / N_particles, N_particles)

    # For smoothing: store ancestor indices and the full set of particles.
    ancestors = Array{Int}(undef, T_obs, N_particles)
    ancestors[1, :] .= 0  # No ancestors for t = 1.
    particles_all = Vector{Matrix{Float64}}(undef, T_obs)
    particles_all[1] = copy(particles)

    logL = 0.0  # Log-likelihood accumulator

    # --- Time 1: weight particles based on the first observation ---
    for i in 1:N_particles
        mean_obs = Z * particles[:, i]
        weights[i] = pdf(MvNormal(mean_obs, H), y[:, 1])
    end
    weight_sum = sum(weights)
    if weight_sum == 0
        return -Inf, zeros(state_dim, T_obs)
    end
    weights /= weight_sum
    logL += log(weight_sum / N_particles)
    idxs = systematic_resample(weights)
    # For time 1 there is no meaningful ancestor; we keep zeros.
    particles = particles[:, idxs]
    particles_all[1] = copy(particles)
    weights .= 1.0 / N_particles

    # --- Time steps 2 to T_obs ---
    for t in 2:T_obs
        # Propagate each particle using the state transition.
        # (Since we resample at every step, the particle order is preserved.)
        for i in 1:N_particles
            η = rand(MvNormal(zeros(size(Q, 1)), Q))
            particles[:, i] = T_mat * particles[:, i] + R_mat * η
        end
        particles_all[t] = copy(particles)

        # Compute the weights given observation y[:, t].
        for i in 1:N_particles
            mean_obs = Z * particles[:, i]
            weights[i] = pdf(MvNormal(mean_obs, H), y[:, t])
        end
        weight_sum = sum(weights)
        if weight_sum == 0
            return -Inf, zeros(state_dim, T_obs)
        end
        weights /= weight_sum
        logL += log(weight_sum / N_particles)

        # Resample and record the ancestor indices.
        idxs = systematic_resample(weights)
        ancestors[t, :] = idxs  # For each new particle, store its parent's index.
        particles = particles[:, idxs]
        particles_all[t] = copy(particles)
        weights .= 1.0 / N_particles
    end

    # --- Backward Simulation: sample one trajectory from the smoothing distribution ---
    trajectory = zeros(state_dim, T_obs)
    # At the final time step, select an index uniformly.
    idx = rand(1:N_particles)
    trajectory[:, T_obs] = particles_all[T_obs][:, idx]
    for t in (T_obs - 1):-1:1
        # For t = 1, ancestors[t+1, idx] is zero so we simply use the particle from time 1.
        if t == 1
            trajectory[:, t] = particles_all[t][:, max(idx, 1)]
        else
            idx = ancestors[t + 1, idx]
            trajectory[:, t] = particles_all[t][:, idx]
        end
    end

    return logL, trajectory
end

end  # module ParticleFilter
