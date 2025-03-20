module model

export state_space, model_specs

using LinearAlgebra


# State-space formulation for the wave cycle model with deterministic drift.
function state_space(θ; cycle_order=1)
    # θ = [μ, ρ, λ, σ_ϵ, σ_ξ, σ_κ]
    μ    = θ[1]
    ρ    = θ[2]
    λ    = θ[3]
    σ_ϵ   = θ[4]
    σ_ξ  = θ[5]
    σ_κ  = θ[6]
 

    # Trend part: only a level state.
    # Cycle part: 2 states for each cycle block.
    state_dim = 1 + 2 * cycle_order

    # Observation: yₜ = uₜ + ψ_{max,t} + εₜ.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1       # level state uₜ
    Z[1, 2] = 1       # first element of cycle block
    H = [σ_ϵ^2]
    d = [0.0]

    # Transition matrix.
    T = zeros(state_dim, state_dim)
    T[1, 1] = 1   # uₜ = uₜ₋₁ + μ + ξₜ

    for i in 1:cycle_order
        idx = 2 + 2*(i-1)
        T[idx,   idx]   = ρ * cos(λ)
        T[idx,   idx+1] = ρ * sin(λ)
        T[idx+1, idx]   = -ρ * sin(λ)
        T[idx+1, idx+1] = ρ * cos(λ)
        if i < cycle_order
            next_idx = idx + 2
            T[idx,   next_idx]   = 1
            T[idx+1, next_idx+1] = 1
        end
    end

    # Selection matrix: trend innovation and cycle innovation.
    R = zeros(state_dim, 3)
    R[1, 1] = 1  # trend innovation enters uₜ
    idx_low = 2 + 2*(cycle_order - 1)
    R[idx_low,     2] = 1  # first element of lowest cycle block gets κₜ
    R[idx_low+1,   3] = 1  # second element gets κₜ^*

    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ^2
    Q[2, 2] = σ_κ^2
    Q[3, 3] = σ_κ^2

    c = zeros(state_dim)
    c[1] = μ  # incorporate deterministic drift

    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1, 1] = 1

    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for wave cycle deterministic drift.
function model_specs()
    θ_elements = ["μ", "ρ", "λ", "σ_ϵ", "σ_ξ", "σ_κ"]
    α_elements = ["u", "ψ", "ψ*"]
    θ_true = [0.02, 0.5, π/3, 0.05, 0.01, 0.2]
    support = [
        -Inf  Inf;   # μ
        0.0   1.0;   # ρ
        0.0   π;     # λ
        0.0   Inf;   # σ_ϵ
        0.0   Inf;   # σ_ξ
        0.0   Inf    # σ_κ
    ]
    prior_distributions = ("normal", "uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma")
    prior_hyperparameters = [
        0.5   5.0;
        0.0   1.0;
        0.0   π;
        1e-6  1e-6;
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

end  # module WaveCycleDeterministicDriftModel
