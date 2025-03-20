module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the univariate wave cycle model.
function state_space(θ)
    # θ = [ρ, λ, σₑ, σ_κ]
    ρ    = θ[1]
    λ    = θ[2]
    σ_ε  = θ[3]
    σ_κ  = θ[4]

    state_dim = 2  # two cycle states.
    obs_dim   = 1

    # Observation: yₜ = ψₜ + εₜ.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0
    H = [σ_ε^2]
    d = [0.0]

    # Transition: a 2×2 rotation scaled by ρ.
    T = ρ * [cos(λ)  sin(λ);
             -sin(λ) cos(λ)]
    
    # Process noise covariance.
    Q = σ_κ^2 * Matrix{Float64}(I, state_dim, state_dim)
    R = Matrix{Float64}(I, state_dim, state_dim)
    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for the wave cycle.
function model_specs()
    θ_elements = ["ρ", "λ", "σ_e", "σ_κ"]
    α_elements = ["ψ_t", "ψ*_t"]
    θ_true = [0.9, π/3, 0.1, 0.3]
    support = [
        0.0   1.0;  # ρ
        0.0   π;    # λ
        0.0   Inf;  # σ_e
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

end  # module WaveCycleModel
