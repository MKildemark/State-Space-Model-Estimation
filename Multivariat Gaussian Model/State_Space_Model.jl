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
# Univariat State-Space
#########################
function uni_state_space(θ, cycle_order, σʸ)
    # Parameter vector for the univariate model:
    #   θ = [ ρ, λ_c, σ_ε, σ_ξ, σ_κ ]
    ρ     = θ[1]
    λ_c   = θ[2]
    σ_ε   = θ[3]
    σ_ξ   = θ[4]
    σ_κ   = θ[5]

    # The state vector is defined as:
    #   [ u_t, β_t, ψ_{2,t}, ψ_{2,t}*, ψ_{1,t}, ψ_{1,t}* ]
    state_dim = 2 + 2*cycle_order

    ##########################
    # 1. Measurement Equation
    ##########################
    # y_t = u_t + ψ_{2,t} + ε_t, with ε_t ~ N(0, σ_ε)
    # Z selects u_t (state[1]) and ψ_{2,t} (state[3])
    Z = zeros(1, state_dim)
    Z[1, 1] = 1      # u_t
    Z[1, 3] = 1      # ψ_{2,t}
    # rescale 

    Z = Z ./ σʸ

    # Measurement error covariance 
    H = [σ_ε/(σʸ^2)]
    
    

    ##########################
    # 2. Transition Equation
    ##########################
    T = zeros(state_dim, state_dim)
    # -- Trend equations --
    # u_t = u_{t-1} + β_{t-1}
    T[1, 1] = 1;  T[1, 2] = 1
    # β_t = β_{t-1} + ξ_t
    T[2, 2] = 1

    # -- Second-order cycle for ψ_{2,t} and ψ_{2,t}* --
    # ψ_{2,t}   = ρcosλ_c·ψ_{2,t-1} + ρsinλ_c·ψ_{2,t-1}* + ψ_{1,t-1}
    T[3, 3] = ρ*cos(λ_c);  T[3, 4] = ρ*sin(λ_c);  T[3, 5] = 1
    # ψ_{2,t}*  = -ρsinλ_c·ψ_{2,t-1} + ρcosλ_c·ψ_{2,t-1}* + ψ_{1,t-1}*
    T[4, 3] = -ρ*sin(λ_c); T[4, 4] = ρ*cos(λ_c);  T[4, 6] = 1

    # -- First-order cycle for ψ_{1,t} and ψ_{1,t}* --
    # ψ_{1,t}  = ρcosλ_c·ψ_{1,t-1} + ρsinλ_c·ψ_{1,t-1}* + κ_t
    T[5, 5] = ρ*cos(λ_c);  T[5, 6] = ρ*sin(λ_c)
    # ψ_{1,t}* = -ρsinλ_c·ψ_{1,t-1} + ρcosλ_c·ψ_{1,t-1}* + κ_t^*
    T[6, 5] = -ρ*sin(λ_c); T[6, 6] = ρ*cos(λ_c)

    ##########################
    # 3. Selection and Process Noise
    ##########################
    # The shocks:
    #   ξ_t   affects β_t (second state)
    #   κ_t   affects ψ_{1,t} (fifth state)
    #   κ_t^* affects ψ_{1,t}* (sixth state)
    R = zeros(state_dim, 3)
    R[2, 1] = 1   # β_t gets ξ_t
    R[5, 2] = 1   # ψ_{1,t} gets κ_t
    R[6, 3] = 1   # ψ_{1,t}* gets κ_t^*

    # Process noise covariance matrix for the shocks:
    Q = zeros(3, 3)
    Q[1, 1] = σ_ξ
    Q[2, 2] = σ_κ
    Q[3, 3] = σ_κ

    # Diffuse prior for the nonstationary states: u_t and β_t.
    P_diffuse = zeros(state_dim, state_dim)
    P_diffuse[1:2, 1:2] = Matrix(I, 2, 2)

    return Z, H, T, R, Q, P_diffuse
end





#########################
# Multivariate State-Space
#########################
function multi_state_space(θ, cycle_order, σʸ)
    # Unpack parameters.
    #theta = [ ρ, λ_c, c₁, c₂, v_ε, v_ξ, v_κ, σ²_ε,y, σ²_ξ,y, σ²_κ,y, σ²_ε,π, σ²_ξ,π, σ²_κ,π ]
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


    # Define the state vector dimension.
    # Trend: 4 states
    # Output cycle: 2*cycle_order states
    # Inflation cycle: 2*cycle_order states
    # Phillips lags: 2 states
    # The state vector is ordered as:
    #   [ uₜ^y,
    #     βₜ^y,
    #     uₜ^π,
    #     βₜ^π,
    #     (Output cycle states, 2*cycle_order elements),
    #     (Inflation cycle states, 2*cycle_order elements),
    #     (Phillips lags, 2 elements) ]
    state_dim = 6 + 4*cycle_order

    ##########################
    # 1. Observation Equation
    ##########################
    # The observations are:
    #   yₜ = uₜ^y + ψₜ^y  + εₜ^y
    #   πₜ = uₜ^π + ψₜ^π + pₜ + εₜ^π,   with   pₜ = c1 * (lag₁ ψₜ^y) + c2 * (lag₂ ψₜ^y)
    #
    # With the state–vector ordering, we take:
    #   uₜ^y       → state[1]
    #   uₜ^π       → state[3]
    #   ψₜ^y (for measurement) is chosen as the first element
    #     of the output cycle block → state[5]
    #   ψₜ^π (for measurement) is the first element
    #     of the inflation cycle block → state[5 + 2*cycle_order]
    #   The Phillips lags are stored in the final two states:
    #     tilde{ψ}ₜ^y and tilde{ψ}_{t-1}^y → state[state_dim-1] and state[state_dim]
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

    #rescale
    # if sigma is 2 dimensional, rescale
    if length(σʸ) == 2
        σʸ = Diagonal(σʸ)
        Z = inv(σʸ) * Z
        H = inv(σʸ) * H * inv(σʸ)
    end



    ###########################
    # 2. Transition Equation
    ###########################
  
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

    ###########################
    # 3. Selection and Process Noise
    ###########################
    # Innovations:
    #   - Trend shocks: ξₜ = [ξₜ^y, ξₜ^π]′,
    #   - Cycle shocks for base cycles: for output [κₜ^y, κₜ^{y*}]′ and for inflation [κₜ^π, κₜ^{π*}]′.
    # Total of 6 shocks:
    #    ηₜ = [ ξₜ^y, ξₜ^π, κₜ^y, κₜ^{y*}, κₜ^π, κₜ^{π*} ]′.
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


########################
# Simple bivariat for testing
########################

function simple_bi_state_space(θ, σʸ)
    # θ = [σ²_ε_y, σ²_η_y, σ²_ε_π, σ²_η_π]
    σ2_ε_y = θ[1]
    σ2_η_y = θ[2]
    σ2_ε_π = θ[3]
    σ2_η_π = θ[4]

    state_dim = 2  # [μ^y, μ^π]
    obs_dim   = 2  # Observations: [y, π]

    # 1. Observation Equation:
    # y_t = μ_t^y + ε_t^y,  π_t = μ_t^π + ε_t^π
    Z = Matrix{Float64}(I, obs_dim, state_dim)

    # Measurement error covariance:
    H = [σ2_ε_y   0.0;
         0.0    σ2_ε_π]

    # Optionally, if you want to rescale the observations, you can use σʸ.
    # (Here we assume σʸ is either 1 or a 2-element vector.)
    if length(σʸ) == 2
        σ_scale = Diagonal(σʸ)
        Z = inv(σ_scale) * Z
        H = inv(σ_scale) * H * inv(σ_scale)
    end

    # 2. Transition Equation:
    # μ_t = μ_{t-1} + η_t, so T is the identity matrix.
    T = Matrix{Float64}(I, state_dim, state_dim)

    # 3. Selection and Process Noise:
    # The state noise enters directly:
    R = Matrix{Float64}(I, state_dim, state_dim)
    Q = [σ2_η_y   0.0;
         0.0    σ2_η_π]

    # Diffuse prior for the state (set as an identity matrix)
    # P_diffuse = Matrix{Float64}(I, state_dim, state_dim)
    P_diffuse = zeros(state_dim, state_dim)

    return Z, H, T, R, Q, P_diffuse
end



#########################
# Choose Model
#########################

function state_space(θ, cycle_order, σʸ)
    if length(θ) == 5
        return uni_state_space(θ, cycle_order, σʸ)
    elseif length(θ) == 10
        return multi_state_space(θ, cycle_order,σʸ)
    elseif length(θ) == 4
        return simple_bi_state_space(θ, σʸ)
    else
        error("Incorrect number of parameters.")
    end
end


#########################
# Simulation Function
#########################
function simulate_data(θ, cycle_order, n_obs)
    # Retrieve system matrices from the multivariate state-space function.

    Z, H, T, R, Q = state_space(θ, cycle_order, 1.0)
    state_dim = size(T, 1)
    obs_dim = size(Z, 1)  
    
    # Initialize arrays.
    α = zeros(state_dim, n_obs)
    y = zeros(obs_dim, n_obs)

    α_current = zeros(state_dim)
    for t in 1:n_obs
        # simulate state evolution:
        η = rand(MvNormal(zeros(size(Q,1)), Q))
        α_current = T * α_current + R * η
        α[:, t] = α_current
        # simulate measurement:
        ε = rand(MvNormal(zeros(obs_dim), H))
        y[:, t] = Z * α_current + ε
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
    
    # Loop over each column (variable)
    for j in 1:n_vars
        # Compute standard deviation of first differences.
    
        s = std(diff(y[j, :]))
        σʸ[j] = s
        # Scale the entire series by dividing by s.
        y_std[j, :] = y[j, :] ./ s
    end
    
    return y_std, σʸ
end


end