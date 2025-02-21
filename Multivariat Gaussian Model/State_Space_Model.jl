module state_space_model

export state_space, simulate_data, standardize_data

using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

Random.seed!(123)

#########################
# Wave Cycle Stochastic Drift 
#########################

function wave_cycle_stochastic_drift(θ, cycle_order, σʸ)
    # Parameter vector for the univariate model:
    #   θ = [ ρ, λ_c, σ_ε, σ_ξ, σ_κ ]
    ρ     = θ[1]
    λ_c   = θ[2]
    σ_ε   = θ[3]
    σ_ξ   = θ[4]
    σ_κ   = θ[5]
    σʸ   = σʸ[1]

    # Total state dimension: 2 for trend (u_t and β_t) and 2 for each cycle block.
    state_dim = 2 + 2 * cycle_order

    # The observation equation is:
    #   y_t = u_t + ψ_{max,t} + ε_t
    # where u_t is the first state and ψ_{max,t} is the cosine-part of the highest cycle order.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1       # u_t
    Z[1, 3] = 1       # ψ_{max,t} (the first element in the first cycle block)
    # rescale
    # Z = Z ./ σʸ

    # Measurement error covariance:
    # H = [σ_ε / (σʸ^2)]
    H = [σ_ε]

    # Transition matrix T 
    T = zeros(state_dim, state_dim)
    # -- Trend equations --
    # u_t = u_{t-1} + β_{t-1}
    T[1, 1] = 1;   T[1, 2] = 1
    # β_t = β_{t-1}  (plus shock later)
    T[2, 2] = 1

    # Cycle equations 
    for i in 1:cycle_order
        idx = 2 + 2*(i-1) + 1       # starting index of block i
        # Set the 2x2 rotation part:
        T[idx,     idx]     = ρ * cos(λ_c)
        T[idx,     idx+1]   = ρ * sin(λ_c)
        T[idx+1,   idx]     = -ρ * sin(λ_c)
        T[idx+1,   idx+1]   = ρ * cos(λ_c)

        # For blocks i=1,...,cycle_order-1 (i.e. not the lowest order), add the next block:
        if i < cycle_order
            next_idx = idx + 2
            T[idx,     next_idx]   = 1   # adds ψ from lower-order block
            T[idx+1,   next_idx+1] = 1
        end
    end
    # rescale
    # T = T ./ σʸ

    # Selection matrix R
    R = zeros(state_dim, 3)
    R[2, 1] = 1   # β_t gets ξ_t

    # Determine indices for the lowest cycle block:
    idx_low = 2 + 2*(cycle_order - 1) + 1  
    R[idx_low,     2] = 1   # ψ_{1,t} gets κ_t
    R[idx_low+1,   3] = 1   # ψ^*_{1,t} gets κ^*_t

    # Process noise covariance matrix Q (3x3):
    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ
    Q[2, 2] = σ_κ
    Q[3, 3] = σ_κ

    # Diffuse prior for the nonstationary trend states: u_t and β_t.
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:2, 1:2] = Matrix(I, 2, 2)

    return Z, H, T, R, Q, P_diffuse
end

#########################
# Wave Cycle Stochastic Drift No Noise
#########################

function wave_cycle_stochastic_drift_no_obs(θ, cycle_order, σʸ)
    # Parameter vector for the univariate model without observation noise:
    #   θ = [ ρ, λ_c, σ_ξ, σ_κ ]
    ρ     = θ[1]
    λ_c   = θ[2]
    σ_ξ   = θ[3]
    σ_κ   = θ[4]
    σʸ   = σʸ[1]

    # Total state dimension: 2 for trend (u_t and β_t) and 2 for each cycle block.
    state_dim = 2 + 2 * cycle_order

    # The observation equation is:
    #   y_t = u_t + ψ_{max,t}
    # where u_t is the first state and ψ_{max,t} is the cosine-part of the highest cycle order.
    Z = zeros(1, state_dim)
    Z[1, 1] = 1       # u_t
    Z[1, 3] = 1       # ψ_{max,t} (the first element in the first cycle block)
    # rescale
    # Z = Z ./ σʸ

    # Measurement error covariance: No observation noise.
    H = [0.0]

    # Transition matrix T 
    T = zeros(state_dim, state_dim)
    # -- Trend equations --
    # u_t = u_{t-1} + β_{t-1}
    T[1, 1] = 1;   T[1, 2] = 1
    # β_t = β_{t-1}
    T[2, 2] = 1

    # Cycle equations 
    for i in 1:cycle_order
        idx = 2 + 2*(i-1) + 1       # starting index of block i
        # Set the 2x2 rotation part:
        T[idx,     idx]     = ρ * cos(λ_c)
        T[idx,     idx+1]   = ρ * sin(λ_c)
        T[idx+1,   idx]     = -ρ * sin(λ_c)
        T[idx+1,   idx+1]   = ρ * cos(λ_c)

        # For blocks i=1,...,cycle_order-1 (i.e. not the lowest order), add the next block:
        if i < cycle_order
            next_idx = idx + 2
            T[idx,     next_idx]   = 1   # adds ψ from lower-order block
            T[idx+1,   next_idx+1] = 1
        end
    end
    # rescale
    # T = T ./ σʸ

    # Selection matrix R (same structure as in the noisy version)
    R = zeros(state_dim, 3)
    R[2, 1] = 1   # β_t gets ξ_t

    # Determine indices for the lowest cycle block:
    idx_low = 2 + 2*(cycle_order - 1) + 1  
    R[idx_low,     2] = 1   # ψ_{1,t} gets κ_t
    R[idx_low+1,   3] = 1   # ψ^*_{1,t} gets κ^*_t

    # Process noise covariance matrix Q (3x3):
    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ
    Q[2, 2] = σ_κ
    Q[3, 3] = σ_κ

    # Diffuse prior for the nonstationary trend states: u_t and β_t.
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:2, 1:2] = Matrix(I, 2, 2)

    return Z, H, T, R, Q, P_diffuse
end

#########################
# Multivariate Wave Cycle Stochastic Drift
#########################

function multivariate_wave_cycle_stochastic_drift(θ, cycle_order, σʸ)
    # Unpack parameters.
    # θ = [ ρ, λ_c, c₁, c₂, v_ε, v_ξ, v_κ, σ²_ε,y, σ²_ξ,y, σ²_κ,y, σ²_ε,π, σ²_ξ,π, σ²_κ,π ]
    ρ          = θ[1]
    λ_c        = θ[2]
    c_1        = θ[3]
    c_2        = θ[4]
    v_ε        = 0.0
    v_ξ        = 0.0
    v_κ        = 0.0
    σ2_ε_y     = θ[5]
    σ2_ξ_y     = θ[6]
    σ2_κ_y     = θ[7]
    σ2_ε_π     = θ[8]
    σ2_ξ_π     = θ[9]
    σ2_κ_π     = θ[10]

    state_dim = 6 + 4*cycle_order

    # The observations are:
    #   yₜ = uₜ^y + ψₜ^y  + εₜ^y
    #   πₜ = uₜ^π + ψₜ^π + pₜ + εₜ^π,   with   pₜ = c1 * (lag₁ ψₜ^y) + c2 * (lag₂ ψₜ^y)

    Z = zeros(2, state_dim)
    # Equation for yₜ:
    Z[1, 1] = 1            # uₜ^y
    Z[1, 5] = 1            # ψₜ^y (output cycle, first element)
    # Equation for πₜ:
    Z[2, 3] = 1            # uₜ^π
    Z[2, 5 + 2*cycle_order] = 1  # ψₜ^π (inflation cycle, first element)
    Z[2, state_dim-1] = c_1   # lag 1 of ψₜ^y
    Z[2, state_dim]   = c_2   # lag 2 of ψₜ^y

    # Measurement error covariance matrix Σ_ε (2×2)
    Σ_ε = [σ2_ε_y   v_ε;
           v_ε      σ2_ε_π]
    H = Σ_ε

    # rescale
    if length(σʸ) == 2
        σʸ = Diagonal(σʸ)
        Z = inv(σʸ) * Z
        H = inv(σʸ) * H * inv(σʸ)
    end

    # Transition matrix T
    T = zeros(state_dim, state_dim)
    # Trend block (states 1–4)
    T[1,1] = 1;  T[1,2] = 1      # uₜ^y = uₜ₋₁^y + βₜ₋₁^y
    T[2,2] = 1                  # βₜ^y = βₜ₋₁^y + ξₜ^y
    T[3,3] = 1;  T[3,4] = 1      # uₜ^π = uₜ₋₁^π + βₜ₋₁^π
    T[4,4] = 1                  # βₜ^π = βₜ₋₁^π + ξₜ^π

    # Rotation matrix for cycles:
    rotation_matrix = ρ * [ cos(λ_c)  sin(λ_c);
                            -sin(λ_c)  cos(λ_c) ]

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
    T[lag_start, 5] = 1                   # tilde{ψ}ₜ^y = ψₜ₋₁^y (using output cycle's first element)
    T[lag_start+1, lag_start] = 1          # shift: tilde{ψ}_{t-1}^y = previous tilde{ψ}ₜ^y

    # Innovations:
    R = zeros(state_dim, 6)
    # Trend shocks.
    R[2, 1] = 1    # βₜ^y gets ξₜ^y.
    R[4, 2] = 1    # βₜ^π gets ξₜ^π.
    # Cycle shocks for output base cycle (last 2 states of output cycle block)
    out_base_idx = out_start + 2*(cycle_order - 1)
    R[out_base_idx:out_base_idx+1, 3:4] = I(2)
    # Cycle shocks for inflation base cycle (last 2 states of inflation cycle block)
    inf_base_idx = inf_start + 2*(cycle_order - 1)
    R[inf_base_idx:inf_base_idx+1, 5:6] = I(2)

    # Process noise covariance (6×6):
    # Innovations for the trends: Σ_ξ (2×2)
    Σ_ξ = [σ2_ξ_y   v_ξ;
           v_ξ      σ2_ξ_π]
    # Innovations for the cycles: Σ_κ (2×2) (used twice for the two cycle equations)
    Σ_κ = [σ2_κ_y   v_κ;
           v_κ      σ2_κ_π]
    
    Q = zeros(6, 6)
    Q[1:2, 1:2] = Σ_ξ
    Q[3:4, 3:4] = Σ_κ
    Q[5:6, 5:6] = Σ_κ

    # Diffuse prior for the non-stationary states uₜ^y,βₜ^y,uₜ^π,βₜ^π,
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:4, 1:4] = Matrix(I, 4, 4)

    return Z, H, T, R, Q, P_diffuse
end

#########################
# Cycle-Only models
#########################

function wave_cycle(θ)
    # θ = [ρ, λ, σ²ₖ, σ²ₑ]
    ρ    = θ[1]
    λ    = θ[2]
    σ²ε  = θ[3]^2
    σ²κ  = θ[4]^2

    # State dimension: two cycle states
    state_dim = 2
    obs_dim   = 1

    # Observation equation: yₜ = ψₜ + εₜ
    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0

    # Measurement noise variance.
    H = [σ²ε]

    # Transition equation:
    T = ρ * [cos(λ)  sin(λ);
             -sin(λ) cos(λ)]
    
    # Process noise covariance matrix: independent shocks in both cycle states.
    Q = σ²κ * Matrix{Float64}(I, state_dim, state_dim)
    
    # In this model, the process noise enters directly.
    R = Matrix{Float64}(I, state_dim, state_dim)
    
    # Diffuse prior for the state.
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, T, R, Q, P_diffuse
end

function ar1_cycle(θ)
    # θ = [ϕ, σ_ϵ, σ_κ] where σ_ϵ and σ_κ are standard deviations.
    ϕ    = θ[1]
    σ_ϵ  = θ[2]
    σ_κ  = θ[3]
    
    # State dimension: two elements [ψₜ, ψₜ₋₁]
    state_dim = 2
    obs_dim   = 1

    # Observation equation: yₜ = ψₜ + εₜ
    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0

    # Measurement noise variance.
    H = [σ_ϵ^2]

    # Transition equation in companion form.
    T = [ϕ  0;
         1  0]
    
    # Process noise enters only the first element.
    # R is now a state_dim x 1 matrix.
    R = zeros(state_dim, 1)
    R[1, 1] = 1.0
    # Q is scalar (variance for the state noise)
    Q = [σ_κ^2]
    
    # Diffuse prior for the state.
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, T, R, Q, P_diffuse
end

function ar1_cycle_no_obs(θ)
    ϕ   = θ[1]
    σ_k  = θ[2]
    
    state_dim = 2
    obs_dim   = 1

    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0

    H = [0.0]  # No observation noise

    T = [ϕ  0;
         1  0]
    
    R = zeros(state_dim, 1)
    R[1, 1] = 1.0
    Q = [σ_k^2]
    
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, T, R, Q, P_diffuse
end

function wave_cycle_no_obs(θ)
    ρ   = θ[1]
    λ   = θ[2]
    σ_k  = θ[3]
    
    state_dim = 2
    obs_dim   = 1

    Z = zeros(1, state_dim)
    Z[1, 1] = 1.0

    H = [0.0]  # No observation noise

    T = ρ * [cos(λ)  sin(λ);
             -sin(λ) cos(λ)]
    
    Q = (σ_k^2) * Matrix{Float64}(I, state_dim, state_dim)
    R = Matrix{Float64}(I, state_dim, state_dim)
    
    P_diffuse = zeros(state_dim, state_dim)
    
    return Z, H, T, R, Q, P_diffuse
end

#########################
# Choose Model
#########################

function state_space(model, θ, σʸ; cycle_order = 1)
    if model == "wave cycle stochastic drift"
        return wave_cycle_stochastic_drift(θ, cycle_order, σʸ)
    elseif model == "multivariate wave cycle stochastic drift"
        return multivariate_wave_cycle_stochastic_drift(θ, cycle_order, σʸ)
    elseif model == "wave cycle"
        return wave_cycle(θ)
    elseif model == "ar1 cycle"
        return ar1_cycle(θ)
    elseif model == "wave cycle no noise"
        return wave_cycle_no_obs(θ)
    elseif model == "ar1 cycle no noise"
        return ar1_cycle_no_obs(θ)
    elseif model == "wave cycle stochastic drift no noise"
        return wave_cycle_stochastic_drift_no_obs(θ, cycle_order, σʸ)
    else
        error("Unknown model specification: $model")
    end
end

#########################
# Simulation Function
#########################

function simulate_data(model, θ, n_obs)
    # Retrieve system matrices from the state-space function.
    Z, H, T, R, Q = state_space(model, θ, 1.0)
    state_dim = size(T, 1)
    obs_dim = size(Z, 1)
    
    # Initialize arrays.
    α = zeros(state_dim, n_obs)
    y = zeros(obs_dim, n_obs)
    
    α_current = zeros(state_dim)
    for t in 1:n_obs
        # simulate state evolution:
        α_current = T * α_current + R * rand(MvNormal(zeros(size(Q,1)), Q))
        α[:, t] = α_current
        # simulate measurement:
        ϵ = rand(MvNormal(zeros(obs_dim), H))
        y[:, t] = Z * α_current + ϵ
    end
    
    return y, α
end

#########################
# Standardise (First differences of) data 
#########################

function standardize_data(y)
    n_vars, n_obs = size(y)
    y_std = similar(y)
    σʸ = zeros(n_vars)
    
    # Loop over each variable (row)
    for j in 1:n_vars
        s = std(diff(y[j, :]))
        σʸ[j] = s
        y_std[j, :] = y[j, :] ./ s
    end
    
    return y_std, σʸ
end

end  # module state_space_model
