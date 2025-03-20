module ParticleFilter

export particle_filter

include("import_packages.jl")

using Main.state_space_model

# Helper function to extract the diagonal of a matrix.
getdiag(M::AbstractMatrix) = [M[i,i] for i in 1:size(M,1)]
# If already a vector, just return it.
getdiag(v::AbstractVector) = v

# Check if a matrix is (approximately) diagonal.
function is_diag(mat::AbstractMatrix; tol=1e-12)
    d = getdiag(mat)
    return maximum(abs, mat - Diagonal(d)) < tol
end
# For vectors, we consider them diagonal.
is_diag(mat::AbstractVector; tol=1e-12) = true


function particle_filter(y, θ, a1, P1; N_particles=1000)
    # Retrieve state-space matrices.
    # Z: measurement matrix
    # H: observation covariance
    # d: measurement intercept
    # T_mat: transition matrix
    # R_mat: shock loading matrix
    # Q: process noise covariance
    # c: state intercept
    Z, H, d, T_mat, R_mat, Q, c, _ = get_state_space(θ)
    state_dim = size(T_mat, 1)
    n_obs     = size(y, 2)
    shock_dim = size(Q, 1)

    # Check for diagonal covariances.
    diag_P1 = is_diag(P1)
    diag_Q  = is_diag(Q)
    diag_H  = is_diag(H)

    # Precompute factors for initial state sampling.
    if diag_P1
        sqrtP1 = sqrt.(getdiag(P1))
    else
        L_P1 = cholesky(P1).L
    end

    # Precompute factors for process noise sampling.
    if diag_Q
        sqrtQ = sqrt.(getdiag(Q))
    else
        L_Q = cholesky(Q).L
    end

    # Precompute factors for evaluating the observation density.
    if diag_H
        diagH_vals = getdiag(H)
    else
        L_H    = cholesky(H).L
        # logdetH is used in the multivariate normal log-density.
        logdetH = 2 * sum(log.(abs.(diag(L_H))))
    end

    # Initialize particles.
    particles = zeros(state_dim, N_particles)
    for i in 1:N_particles
        if diag_P1
            particles[:, i] = a1 .+ randn(state_dim) .* sqrtP1
        else
            particles[:, i] = a1 .+ L_P1 * randn(state_dim)
        end
    end

    # Initialize weights.
    weights = fill(1.0 / N_particles, N_particles)
    logL = 0.0                     # Log-likelihood accumulator.
    filtered_states = zeros(state_dim, n_obs)

    # Loop over observations.
    for t in 1:n_obs
        y_t = y[:, t]
        log_weights = zeros(N_particles)
        for i in 1:N_particles
            # Propagate particle:
            #   x_t = T_mat * x_{t-1} + c + R_mat * noise
            if diag_Q
                noise = randn(shock_dim) .* sqrtQ
            else
                noise = L_Q * randn(shock_dim)
            end
            particles[:, i] = T_mat * particles[:, i] .+ c .+ R_mat * noise

            # Compute predicted observation: μ_y = Z * x + d
            μ_y = Z * particles[:, i] .+ d

            # Compute the log-density of y_t.
            d_y = length(y_t)
            if diag_H
                logpdf_y = -0.5 * d_y * log(2π) -
                           0.5 * sum(log.(diagH_vals)) -
                           0.5 * sum(((y_t - μ_y).^2) ./ diagH_vals)
                log_weights[i] = logpdf_y
            else
                diff = y_t - μ_y
                sol = L_H \ diff
                logpdf_y = -0.5 * d_y * log(2π) -
                           0.5 * logdetH -
                           0.5 * dot(sol, sol)
                log_weights[i] = logpdf_y
            end
        end

        # Normalize weights in the log domain.
        max_log_weight = maximum(log_weights)
        weights = exp.(log_weights .- max_log_weight)
        weight_sum = sum(weights)
        weights ./= weight_sum

        # Compute the filtered state (weighted average).
        filtered_states[:, t] = particles * weights

        # Update the log-likelihood.
        logL += max_log_weight + log(weight_sum / N_particles)

        # Resample particles (multinomial resampling).
        indices = sample(1:N_particles, Weights(weights), N_particles)
        particles = particles[:, indices]
        weights .= 1.0 / N_particles
    end

    return logL, filtered_states
end

end  # module ParticleFilter




# function particle_filter(model, y, θ, a1, P1; N_particles=1000)
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

# end  # module ParticleFilter
