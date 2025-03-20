module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for wave cycle stochastic drift with no observation noise.
function state_space(θ; cycle_order=1)
    # θ = [ρ, λ, σ_ξ, σ_κ]
    ρ     = θ[1]
    λ     = θ[2]
    σ_ξ   = θ[3]
    σ_κ   = θ[4]


    state_dim = 2 + 2 * cycle_order

    # Observation: yₜ = uₜ + ψ_{max,t} (no observation noise).
    Z = zeros(1, state_dim)
    Z[1, 1] = 1
    Z[1, 3] = 1
    H = [0.0]
    d = [0.0]

    T = zeros(state_dim, state_dim)
    T[1, 1] = 1; T[1, 2] = 1
    T[2, 2] = 1

    for i in 1:cycle_order
        idx = 2 + 2*(i-1) + 1
        T[idx,     idx]   = ρ * cos(λ)
        T[idx,     idx+1] = ρ * sin(λ)
        T[idx+1,   idx]   = -ρ * sin(λ)
        T[idx+1,   idx+1] = ρ * cos(λ)
        if i < cycle_order
            next_idx = idx + 2
            T[idx,     next_idx]   = 1
            T[idx+1,   next_idx+1] = 1
        end
    end

    R = zeros(state_dim, 3)
    R[2, 1] = 1
    idx_low = 2 + 2*(cycle_order - 1) + 1  
    R[idx_low,     2] = 1
    R[idx_low+1,   3] = 1

    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ^2
    Q[2, 2] = σ_κ^2
    Q[3, 3] = σ_κ^2

    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:2, 1:2] = Matrix(I, 2, 2)

    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for wave cycle stochastic drift with no observation noise.
function model_specs()
    θ_elements = ["ρ", "λ", "σ_ξ", "σ_κ"]
    α_elements = ["μ_t", "β_t", "ψ_t", "ψ*_t"]
    θ_true = [0.7, 0.1, 0.01, 0.1]
    support = [
        0.0   1.0;  # ρ
        0.0   π;    # λ
        0.0   Inf;  # σ_ξ
        0.0   Inf   # σ_κ
    ]
    prior_distributions = ("uniform", "uniform", "inverse_gamma", "inverse_gamma")
    prior_hyperparameters = [
        0.0   0.99;
        0.0   π;
        1e-6  1e-6;
        1e-6  1e-6
    ]
    prior_info = (
        support = support,
        distributions = prior_distributions,
        parameters = prior_hyperparameters
    )
    return (θ_elements = θ_elements, α_elements = α_elements, θ_true = θ_true, prior_info = prior_info)
end

end  # module WaveCycleStochasticDriftNoNoiseModel
