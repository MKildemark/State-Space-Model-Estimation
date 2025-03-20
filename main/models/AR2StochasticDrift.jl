module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the model with stochastic drift and an AR(2) cycle.
# The cycle is represented in companion form:
#    ψ_t = φ₁ ψ_{t-1} + φ₂ ψ_{t-2} + κ_t.
#
# The full state vector is:
#    [μ_t, β_t, ψ_t, ψ_{t-1}],
# where μ_t and β_t are the trend level and slope.
#
# Note: Only cycle_order==1 is supported.
function state_space(θ)


    # θ = [φ1, φ2, σ_ϵ, σ_ξ, σ_κ]
    φ1    = θ[1]
    φ2    = θ[2]
    σ_ϵ   = θ[3]
    σ_ξ   = θ[4]
    σ_κ   = θ[5]

    # Total state dimension: 2 for the trend and 2 for the AR(2) cycle.
    state_dim = 2 + 2

    # Observation equation: yₜ = μₜ + ψₜ + εₜ.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1       # trend level μₜ
    Z[1, 3] = 1       # AR(2) cycle: first element (ψₜ)
    H = [σ_ϵ^2]       # measurement noise variance
    d = [0.0]

    # Transition matrix.
    T = zeros(state_dim, state_dim)
    # Trend equations:
    T[1, 1] = 1; T[1, 2] = 1
    T[2, 2] = 1

    # Cycle equations (AR(2) companion form).
    # Let the cycle block start at index 3.
    T[3, 3] = φ1   # ψₜ = φ₁ ψₜ₋₁ + φ₂ ψₜ₋₂ + κₜ
    T[3, 4] = φ2
    T[4, 3] = 1    # shifting the lagged state: ψₜ₋₁ becomes ψₜ
    T[4, 4] = 0

    # Selection matrix:
    # - Trend slope gets the innovation ξₜ.
    # - Cycle innovation (κₜ) enters the equation for ψₜ.
    R = zeros(state_dim, 2)
    R[2, 1] = 1    # trend slope noise ξₜ
    R[3, 2] = 1    # AR(2) cycle: innovation enters only the first component

    # Process noise covariance.
    Q = zeros(2, 2)
    Q[1, 1] = σ_ξ^2
    Q[2, 2] = σ_κ^2

    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    # (Optional: specify diffuse initial state uncertainty for the trend and cycle if desired.)

    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for the AR(2) cycle with stochastic drift.
function model_specs()
    

    θ_elements = ["φ₁", "φ₂", "σₑ", "σ_ξ", "σ_κ"]
    α_elements = ["μₜ", "βₜ", "ψₜ", "ψₜ₋₁"]

    # Example true parameter values.
    θ_true = [0.5, -0.2, 0.05, 0.01, 0.1]

    # Support for each parameter.
    support = [
        -1.0  2.0;   # φ₁
        -1.0  2.0;   # φ₂
        0.0   Inf;   # σₑ
        0.0   Inf;   # σ_ξ
        0.0   Inf    # σ_κ
    ]

    # Prior distributions for each parameter.
    # (Normal priors for AR parameters; inverse-gamma for variance parameters.)
    prior_distributions = ("uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma")
    prior_hyperparameters = [
        -1.0   2.0;    # mean and variance (or precision) for φ₁
        -1.0   2.0;    # φ₂
        1e-6  1e-6;   # hyperparameters for σₑ
        1e-6  1e-6;   # σ_ξ
        1e-6  1e-6    # σ_κ
    ]
    prior_info = (
        support = support,
        distributions = prior_distributions,
        parameters = prior_hyperparameters
    )
    return (θ_elements = θ_elements, α_elements = α_elements, θ_true = θ_true, prior_info = prior_info)
end

end  # module WaveCycleAR2StochasticDriftModel
