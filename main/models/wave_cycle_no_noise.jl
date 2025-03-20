module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the wave cycle model without observation noise.
function state_space(θ)
    # θ = [ρ, λ, σ_κ]
    ρ   = θ[1]
    λ   = θ[2]
    σ_k = θ[3]
    
    state_dim = 2
    obs_dim   = 1

    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0
    H = [0.0]  # No observation noise.
    d = [0.0]

    T = ρ * [cos(λ)  sin(λ);
             -sin(λ) cos(λ)]
    
    Q = zeros(state_dim, state_dim)
    Q[2, 2] = σ_k^2
    Q[1,1] = σ_k^2
    R = zeros(state_dim, state_dim)
    R[1, 1] = 1.0
    R[2, 2] = 1.0
    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for the wave cycle with no noise.
function model_specs()
    θ_elements = ["ρ", "λ", "σ_κ"]
    α_elements = ["ψ_t", "ψ*_t"]
    θ_true = [0.9, π/3, 0.3]
    support = [
        0.0   1.0;  # ρ
        0.0   π;    # λ
        0.0   Inf   # σ_κ
    ]
    prior_distributions = ("uniform", "uniform", "inverse_gamma")
    prior_hyperparameters = [
        0.0   0.99;
        0.0   π;
        1e-6  1e-6
    ]
    prior_info = (
        support = support,
        distributions = prior_distributions,
        parameters = prior_hyperparameters
    )
    return (θ_elements = θ_elements, α_elements = α_elements, θ_true = θ_true, prior_info = prior_info)
end

end  # module WaveCycleNoNoiseModel
