module ModelSpecs

export get_model_info

function get_model_info(model::String)

    if model == "ar1 cycle"
        θ_elements = ["ϕ", "σ_ϵ", "σ_κ"]
        α_elements = ["ψ_{t}", "ψ_{t-1}"]
        θ_true = [0.5, 0.2, 0.8]
        support = [
            0.0  0.99;    # for ϕ
            0.0 Inf;      # for σ_ϵ
            0.0 Inf       # for σ_κ
        ]
        prior_distributions = ("uniform", "inverse_gamma", "inverse_gamma")
        prior_hyperparameters = [
            0.0   0.98;
            3.0  0.5;
            1e-6  1e-6
        ]
        prior_info = (
            support = support,
            distributions = prior_distributions,
            parameters = prior_hyperparameters
        )
        prior_info_collection = [prior_info]

    elseif model == "ar1 cycle no noise"
        θ_elements = ["ϕ", "σ_κ"]
        α_elements = ["ψ_{t}", "ψ_{t-1}"]
        θ_true = [0.9, 1.0]
        support = [
            0.0  0.99;   # for ϕ
            0.0 Inf      # for σ_κ
        ]
        prior_distributions = ("uniform", "inverse_gamma")
        prior_hyperparameters = [
            0.0   0.98;
            1e-6  1e-6
        ]
        prior_info = (
            support = support,
            distributions = prior_distributions,
            parameters = prior_hyperparameters
        )
        prior_info_collection = [prior_info]

    elseif model == "wave cycle"
        θ_elements = ["ρ", "λ", "σ_ϵ", "σ_κ"]
        α_elements = ["ψ_t", "ψ*_t"]
        θ_true = [0.9, π/3, 0.1, 0.3]
        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf;
            0.0  Inf
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
        prior_info_collection = [prior_info]

    elseif model == "wave cycle no noise"
        θ_elements = ["ρ", "λ", "σ_κ"]
        α_elements = ["ψ_t", "ψ*_t"]
        θ_true = [0.9, π/3, 0.3]
        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf
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
        prior_info_collection = [prior_info]

    elseif model == "wave cycle stochastic drift"
        θ_elements = ["ρ", "λ", "σ_ϵ", "σ_ξ", "σ_κ"]
        α_elements = ["μ_t", "β_t", "ψ_t", "ψ*_t"]
        θ_true = [0.5, 0.1, 0.05, 0.1, 0.01]
        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf;
            0.0  Inf;
            0.0  Inf
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
        prior_info_collection = [prior_info, prior_info]

    elseif model == "wave cycle stochastic drift no noise"
        θ_elements = ["ρ", "λ", "σ_ξ", "σ_κ"]
        α_elements = ["μ_t", "β_t", "ψ_t", "ψ*_t"]
        θ_true = [0.7, 0.1, 0.01, 0.1]
        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf;
            0.0  Inf
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
        prior_info_collection = [prior_info]


    elseif model == "multivariate wave cycle stochastic drift no noise"
        θ_elements = ["ρ", "λ", "c₁", "c₂", "σ²_ξ,y", "σ²_κ,y", "σ²_ξ,π", "σ²_κ,π"]
        α_elements = ["uₜ^y", "βₜ^y", "uₜ^π", "βₜ^π", "ψₜ^y", "ψ*ₜ^y", "ψₜ^π", "ψ*ₜ^π", "tilde{ψ}ₜ^y", "tilde{ψ}_{t-1}^y"]
        θ_true = [0.9, π/3, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]
        support = [
            0.0   1.0;   # for ρ
            0.0   π;     # for λ
            -Inf  Inf;   # for c₁
            -Inf  Inf;   # for c₂
            0.0   Inf;   # for σ²_ξ,y
            0.0   Inf;   # for σ²_κ,y
            0.0   Inf;   # for σ²_ξ,π
            0.0   Inf    # for σ²_κ,π
        ]
        prior_distributions = ("uniform", "uniform", "uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma", "inverse_gamma")
        prior_hyperparameters = [
            0.0   1.0;
            0.0   π;
            -10.0 10.0;
            -10.0 10.0;
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
        prior_info_collection = [prior_info]

    

    elseif model == "multivariate wave cycle stochastic drift"
        θ_elements = ["ρ", "λ_c", "c₁", "c₂", "σ²_ε, y", "σ²_ξ, y", "σ²_κ, y", "σ²_ε, π", "σ²_ξ, π", "σ²_κ, π"]
        α_elements = ["u_t^y", "β_t^y", "u_t^π", "β_t^π", "ψ^y", "ψ^{y*}", "ψ^π", "ψ^{π*}","tilde{ψ}_t^y", "tilde{ψ}_{t-1}^y"]
        θ_true = [0.9, π/3, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        support = [
            0.0   1.0;   # for ρ
            0.0   π;     # for λ_c
            -Inf  Inf;   # for c₁
            -Inf  Inf;   # for c₂
            0.0   Inf;   # for σ²_ε, y
            0.0   Inf;   # for σ²_ξ, y
            0.0   Inf;   # for σ²_κ, y
            0.0   Inf;   # for σ²_ε, π
            0.0   Inf;   # for σ²_ξ, π
            0.0   Inf    # for σ²_κ, π
        ]
        prior_distributions = ("uniform", "uniform", "uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma", "inverse_gamma", "inverse_gamma", "inverse_gamma")
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
        prior_info_collection = [prior_info]


    elseif model == "wave cycle deterministic drift"
        θ_elements = ["μ", "ρ", "λ", "σ_ϵ", "σ_ξ", "σ_κ"]
        α_elements = ["u", "ψ", "ψ*"]
        θ_true = [0.02, 0.5, π/3, 0.05, 0.01, 0.2]
        support = [
            -Inf  Inf;   # for μ
            0.0   1.0;   # for ρ
            0.0   π;     # for λ
            0.0   Inf;   # for σ_ϵ
            0.0   Inf;   # for σ_ξ
            0.0   Inf    # for σ_κ
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
        prior_info_collection = [prior_info]


    else
        error("Unknown model specification: $model")
    end

    return (
        θ_elements = θ_elements,
        α_elements = α_elements,
        prior_info_collection = prior_info_collection,
        θ_true = θ_true,
    )

end

end  # module ModelSpecs
