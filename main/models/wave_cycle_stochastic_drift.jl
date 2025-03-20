module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the wave cycle model with stochastic drift.
function state_space(θ; cycle_order=1)
    # θ = [ρ, λ, σ_ϵ, σ_ξ, σ_κ]
    ρ     = θ[1]
    λ     = θ[2]
    σ_ϵ    = θ[3]
    σ_ξ    = θ[4]
    σ_κ    = θ[5]

    # State ordering:
    # [trend (μₜ), cycle cosine (ψₜ), trend slope (βₜ), cycle sine (ψ*ₜ), additional cycle blocks...]
    state_dim = 2 + 2 * cycle_order

    # Observation equation: yₜ = μₜ + ψₜ + εₜ.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1       # trend level μₜ
    Z[1, 2] = 1       # cycle cosine ψₜ
    H = [σ_ϵ^2]       # measurement noise variance
    d = [0.0]

    # Transition matrix.
    T = zeros(state_dim, state_dim)
    # Trend equations:
    # μₜ = μₜ + βₜ  and  βₜ = βₜ.
    T[1, 1] = 1;  T[1, 3] = 1
    T[3, 3] = 1

    # Cycle equations:
    # For the first cycle block, we assign:
    # ψₜ (cosine) at state[2] and ψ*ₜ (sine) at state[4].
    T[2, 2] = ρ * cos(λ)
    T[2, 4] = ρ * sin(λ)
    T[4, 2] = -ρ * sin(λ)
    T[4, 4] = ρ * cos(λ)
    # If more than one cycle block, link block 1 to block 2.
    if cycle_order > 1
        # For block 2, the indices will be 5 (cosine) and 6 (sine)
        T[2, 5] = 1
        T[4, 6] = 1
    end

    # For additional cycle blocks (i ≥ 2) beyond the first:
    for i in 2:cycle_order
        # Compute the starting index for cycle block i.
        # For i = 2: idx = 4 + 2*(0) + 1 = 5; for i = 3: idx = 7; etc.
        idx = 4 + 2*(i - 2) + 1
        T[idx, idx]     = ρ * cos(λ)
        T[idx, idx + 1] = ρ * sin(λ)
        T[idx + 1, idx] = -ρ * sin(λ)
        T[idx + 1, idx + 1] = ρ * cos(λ)
        if i < cycle_order
            next_idx = idx + 2
            T[idx, next_idx] = 1
            T[idx + 1, next_idx + 1] = 1
        end
    end

    # Selection matrix: innovations are allocated to the trend slope and the lowest-order cycle block.
    R = zeros(state_dim, 3)
    # Trend slope innovation: βₜ (now at state[3]) gets ξₜ.
    R[3, 1] = 1
    # Cycle innovation: assign to the last cycle block.
    if cycle_order == 1
        # For a one-cycle model, the cycle block occupies states 2 (cosine) and 4 (sine).
        R[2, 2] = 1   # cosine receives κₜ
        R[4, 3] = 1   # sine receives κₜ
    else
        # For cycle_order > 1, determine the starting index of the last cycle block.
        idx_last = 4 + 2*(cycle_order - 2) + 1
        R[idx_last, 2] = 1
        R[idx_last + 1, 3] = 1
    end

    # Process noise covariance.
    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ^2
    Q[2, 2] = σ_κ^2
    Q[3, 3] = σ_κ^2

    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)

    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for wave cycle stochastic drift.
function model_specs(;cycle_order=1)
    
    θ_elements = ["ρ", "λ", "σ_ϵ", "σ_ξ", "σ_κ"]
    if cycle_order == 1
        α_elements = ["μ_t", "ψ_t", "β_t", "ψ*_t"]
    elseif cycle_order == 2
        α_elements = ["μ_t", "ψ_t", "β_t", "ψ*_t", "ψ_t2", "ψ*_t2"]
    else
        error("Cycle order not supported.")
    end

    θ_true = [0.5, 0.1, 0.05, 0.01, 0.1]
    support = [
        0.0   1.0;  # ρ
        0.0   π;    # λ
        0.0   Inf;  # σₑ
        0.0   Inf;  # σₓᵢ
        0.0   Inf   # σₖ
    ]
    prior_distributions = ("uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma")
    prior_hyperparameters = [
        0.0   0.99;
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

    model = "wave_cycle_stochastic_drift"
    return (model = model, θ_elements = θ_elements, α_elements = α_elements, θ_true = θ_true, prior_info = prior_info)
end

end  # module WaveCycleStochasticDriftModel
