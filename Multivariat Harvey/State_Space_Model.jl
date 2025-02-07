module state_space_model

export state_space, simulate_data #, kalman_filter, kalman_smoother,
      # neg_log_likelihood

using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

Random.seed!(123)

#########################
# 1. Multivariate State-Space
#########################
function state_space(θ, cycle_order)
    # Unpack parameters.
    #theta = [ ρ, λ_c, c₁, c₂, v_ε, v_ξ, v_κ, σ²_ε,y, σ²_ξ,y, σ²_κ,y, σ²_ε,π, σ²_ξ,π, σ²_κ,π ]
    ρ          = θ[1]
    λ_c        = θ[2]
    c_1        = θ[3]
    c_2        = θ[4]
    v_ε        = θ[5]
    v_ξ        = θ[6]
    v_κ        = θ[7]
    σ2_ε_y     = θ[8]
    σ2_ξ_y     = θ[9]
    σ2_κ_y     = θ[10]
    σ2_ε_π     = θ[11]
    σ2_ξ_π     = θ[12]
    σ2_κ_π     = θ[13]


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


#########################
# 2. Simulation Function
#########################
function simulate_data(θ, cycle_order, n_obs)
    # Retrieve system matrices from the multivariate state-space function.
    Z, H, T, R, Q = state_space(θ, cycle_order)
    state_dim = size(T, 1)
    obs_dim = size(Z, 1)  
    
    # Initialize arrays.
    α = zeros(n_obs, state_dim)
    y = zeros(n_obs, obs_dim)

    α_current = zeros(state_dim)
    for t in 1:n_obs
        # simulate state evolution:
        η = rand(MvNormal(zeros(size(Q,1)), Q))
        α_current = T * α_current + R * η
        α[t, :] = α_current
        # simulate measurement:
        ε = rand(MvNormal(zeros(obs_dim), H))
        y[t, :] = Z * α_current + ε
    end
    
    return y, α
end

end