module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the AR(1) cycle model without observation noise.
function state_space(θ)
    # θ = [ϕ, σ_κ]
    ϕ    = θ[1]
    σ_k   = θ[2]
  
    # Dimensions.
    state_dim = 2
    obs_dim   = 1

    # Observation equation: yₜ = ψₜ + εₜ.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0
    H = [0.0]
    d = [0.0]
  

    # Transition equation in companion form.
    T = [ϕ  0;
         1  0]

    # Process noise enters only the first state.
    R = zeros(state_dim, 1)
    R[1, 1] = 1.0
    Q = [σ_k^2]

    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for AR(1) cycle with no observation noise.
function model_specs()
    θ_elements = ["ϕ", "σ_κ"]
    α_elements = ["ψₜ", "ψₜ₋₁"]
    θ_true = [0.9, 1.0]
    support = [
        0.0   0.99;   # ϕ
        0.0   Inf     # σ_κ
    ]
    prior_distributions = ("uniform", "inverse_gamma")
    prior_hyperparameters = [
        0.0   0.99;
        1e-6  1e-6
    ]
    prior_info = (
        support = support,
        distributions = prior_distributions,
        parameters = prior_hyperparameters
    )
    return (θ_elements = θ_elements, α_elements = α_elements, θ_true = θ_true, prior_info = prior_info)
end

end  # module AR1CycleNoNoiseModel
