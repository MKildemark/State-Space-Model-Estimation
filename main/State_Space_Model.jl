module state_space_model

export simulate_data, standardize_data, get_model_spec, get_state_space

include("import_packages.jl")


# Import the models
using Main.model





#########################
# Get Model Specification
#########################

function get_model_spec()
    return model.model_specs()
end

#########################
# Get State-Space Matrices
#########################

function get_state_space(θ)
    return model.state_space(θ)
end

#########################
# Simulation Function
#########################

function rand_draw(dim, Σ; rng=Random.GLOBAL_RNG)
    draws = zeros(dim)
    nonzero_variances = findall(i -> abs(Σ[i,i]) > 0, 1:dim)
    if !isempty(nonzero_variances)
        Σ_sub = Σ[nonzero_variances, nonzero_variances]
        draws[nonzero_variances] = rand(rng, MvNormal(zeros(length(nonzero_variances)), Σ_sub))
    end
    return draws
end

function simulate_data(θ, n_obs; X = nothing)
    # Retrieve system matrices from the state-space function.
    Z, H, d, T, R, Q, c = get_state_space(θ)

    state_dim = size(T, 1)
    obs_dim = size(Z, 1)
    
    # Initialize arrays.
    α = zeros(state_dim, n_obs)
    y = zeros(obs_dim, n_obs)
    
    α_current = zeros(state_dim)
    for t in 1:n_obs
        # simulate state evolution:
        if X !== nothing 
            α_current = c*X[:,t] + T * α_current + R * rand(MvNormal(zeros(size(Q,1)), Q))
        else
            α_current = c + T * α_current + R * rand(MvNormal(zeros(size(Q,1)), Q))
        end
        α[:, t] = α_current
        # simulate measurement:
        # ϵ = (norm(H) < 1e-12) ? zeros(obs_dim) : rand(MvNormal(zeros(obs_dim), H))
        ϵ = rand_draw(obs_dim, H)
        y[:, t] = d + Z * α_current + ϵ
    end
    
    return y, α
end

#########################
# Standardise (First differences of) Data 
#########################

function standardize_data(y)
    n_vars, n_obs = size(y)
    y_std = similar(y)
    σy = zeros(n_vars)
    
    # Loop over each variable (row)
    for j in 1:n_vars
        s = std(diff(y[j, :]))
        σy[j] = s
        y_std[j, :] = y[j, :] ./ s
    end
    
    return y_std, σy
end

end  # module state_space_model
