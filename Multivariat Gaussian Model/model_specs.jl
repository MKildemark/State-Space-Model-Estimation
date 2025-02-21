# File: ModelSpecs.jl
module ModelSpecs

export get_model_info


function get_model_info(model::String)


    if model == "ar1 cycle"
        θ_elements = ["ϕ", "σ_ϵ", "σ_κ"]
        α_elements = ["ψ_{t}", "ψ_{t-1}"]
        
        θ_true = [0.5, 0.2, 0.2]

        support = [
            0.0  0.99;    # for ϕ
            0.0 Inf;     # for σ_ϵ
            0.0 Inf      # for σ_κ
        ]
        prior_distributions = ("uniform", "inverse_gamma", "inverse_gamma")

        prior_hyperparameters = [
            0.0  0.98;
            1e-6 1e-6;
            1e-6 1e-6
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

        θ_true = [0.95, π/4, 0.1, 0.1]

        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf;
            0.0  Inf
        ]
        prior_distributions = ("uniform", "uniform", "inverse_gamma", "inverse_gamma")
        prior_hyperparameters = [
            0.0  0.99;
            0.0  π;
            1e-6 1e-6;
            1e-6 1e-6
        ]
        prior_info = (
            support = support,
            distributions = prior_distributions,
            parameters = prior_hyperparameters
        )

        prior_info_collection = [prior_info]



    
    elseif model == "wave cycle stochastic drift"
        θ_elements = ["ρ", "λ", "σ_ϵ", "σ_κ", "σ_ξ"]
        α_elements = ["ψ_t", "ψ*_t"]

        θ_true = [0.95, π/4, 0.1, 0.1, 0.1]

        support = [
            0.0  1.0;
            0.0  π;
            0.0  Inf;
            0.0  Inf;
            0.0  Inf
        ]
        prior_distributions = ("uniform", "uniform", "inverse_gamma", "inverse_gamma", "inverse_gamma")
        prior_hyperparameters = [
            0.0  0.99;
            0.0  π;
            1e-6 1e-6;
            1e-6 1e-6;
            1e-6 1e-6
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
        θ_true =  θ_true,
    )
end

end  # module ModelSpecs
