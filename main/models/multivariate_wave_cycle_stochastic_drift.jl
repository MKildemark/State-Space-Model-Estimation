module model

export state_space, model_specs

using LinearAlgebra

# State-space formulation for the multivariate wave cycle model with stochastic drift.
function state_space(θ; cycle_order=1)
    # θ = [ρ, λ_c, c₁, c₂, σ²_ε,y, σ²_ξ,y, σ²_κ,y, σ²_ε,π, σ²_ξ,π, σ²_κ,π]
    ρ          = θ[1]
    λ_c        = θ[2]
    c₁         = θ[3]
    c₂         = θ[4]
    # v_ε, v_ξ, v_κ are set to zero.
    σ2_ε_y     = θ[5]
    σ2_ξ_y     = θ[6]
    σ2_κ_y     = θ[7]
    σ2_ε_π     = θ[8]
    σ2_ξ_π     = θ[9]
    σ2_κ_π     = θ[10]

    state_dim = 6 + 4 * cycle_order

    # Observation equations.
    # yₜ = uₜ^y + ψₜ^y  + εₜ^y
    # πₜ = uₜ^π + ψₜ^π + pₜ, with pₜ = c₁ * (lag₁ ψₜ^y) + c₂ * (lag₂ ψₜ^y)
    Z = zeros(2, state_dim)
    Z[1, 1] = 1           # uₜ^y
    Z[1, 5] = 1           # ψₜ^y (output cycle, first element)
    Z[2, 3] = 1           # uₜ^π
    Z[2, 5 + 2 * cycle_order] = 1  # ψₜ^π (inflation cycle, first element)
    Z[2, state_dim - 1] = c₁  # lag 1 of ψₜ^y
    Z[2, state_dim]     = c₂  # lag 2 of ψₜ^y

    # Measurement error covariance.
    Σ_ε = [σ2_ε_y  0.0;
           0.0     σ2_ε_π]
    H = Σ_ε

    # # Rescaling if σy is a vector of length 2.
    # if length(σy) == 2
    #     σy_diag = Diagonal(σy)
    #     Z = inv(σy_diag) * Z
    #     H = inv(σy_diag) * H * inv(σy_diag)
    # end

    d = zeros(2)

    T = zeros(state_dim, state_dim)
    # Trend block.
    T[1,1] = 1;  T[1,2] = 1
    T[2,2] = 1
    T[3,3] = 1;  T[3,4] = 1
    T[4,4] = 1

    rotation_matrix = ρ * [cos(λ_c)  sin(λ_c);
                           -sin(λ_c) cos(λ_c)]

    # (a) Output cycle block.
    out_start = 5
    for n in 1:cycle_order
        idx = out_start + 2*(n - 1)
        T[idx:idx+1, idx:idx+1] = rotation_matrix
        if n < cycle_order
            T[idx:idx+1, idx+2:idx+3] += I(2)
        end
    end

    # (b) Inflation cycle block.
    inf_start = 5 + 2*cycle_order
    for n in 1:cycle_order
        idx = inf_start + 2*(n - 1)
        T[idx:idx+1, idx:idx+1] = rotation_matrix
        if n < cycle_order
            T[idx:idx+1, idx+2:idx+3] += I(2)
        end
    end

    # (c) Phillips lags block.
    lag_start = state_dim - 1
    T[lag_start, 5] = 1
    T[lag_start+1, lag_start] = 1

    R = zeros(state_dim, 6)
    R[2, 1] = 1    # βₜ^y gets ξₜ^y.
    R[4, 2] = 1    # βₜ^π gets ξₜ^π.
    out_base_idx = out_start + 2*(cycle_order - 1)
    R[out_base_idx:out_base_idx+1, 3:4] = I(2)
    inf_base_idx = inf_start + 2*(cycle_order - 1)
    R[inf_base_idx:inf_base_idx+1, 5:6] = I(2)

    Σ_ξ = [σ2_ξ_y  0.0;
           0.0     σ2_ξ_π]
    Σ_κ = [σ2_κ_y  0.0;
           0.0     σ2_κ_π]
    
    Q = zeros(6, 6)
    Q[1:2, 1:2] = Σ_ξ
    Q[3:4, 3:4] = Σ_κ
    Q[5:6, 5:6] = Σ_κ

    c = zeros(state_dim)
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:4, 1:4] = Matrix(I, 4, 4)

    return Z, H, d, T, R, Q, c, P_diffuse
end

# Model specifications for the multivariate wave cycle stochastic drift.
function model_specs()
    θ_elements = ["ρ", "λ_c", "c₁", "c₂", "σ²_ε,y", "σ²_ξ,y", "σ²_κ,y", "σ²_ε,π", "σ²_ξ,π", "σ²_κ,π"]
    α_elements = ["u_t^y", "β_t^y", "u_t^π", "β_t^π", "ψ^y", "ψ^{y*}", "ψ^π", "ψ^{π*}", "tilde{ψ}_t^y", "tilde{ψ}_{t-1}^y"]
    θ_true = [0.9, π/3, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    support = [
        0.0   1.0;    # ρ
        0.0   π;      # λ_c
        -Inf  Inf;    # c₁
        -Inf  Inf;    # c₂
        0.0   Inf;    # σ²_ε,y
        0.0   Inf;    # σ²_ξ,y
        0.0   Inf;    # σ²_κ,y
        0.0   Inf;    # σ²_ε,π
        0.0   Inf;    # σ²_ξ,π
        0.0   Inf     # σ²_κ,π
    ]
    prior_distributions = ("uniform", "uniform", "uniform", "uniform", 
                           "inverse_gamma", "inverse_gamma", "inverse_gamma", 
                           "inverse_gamma", "inverse_gamma", "inverse_gamma")
    prior_hyperparameters = [
        0.0   1.0;
        0.0   π;
        -10.0 10.0;
        -10.0 10.0;
        1e-6  1e-6;
        1e-6  1e-6;
        1e-6  1e-6;
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

end  # module MultivariateWaveCycleStochasticDriftModel
