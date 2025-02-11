module kalman

export kalman_filter, kalman_smoother,
       neg_log_likelihood, diffuse_kalman_filter

using Random
using LinearAlgebra
using Statistics
using Distributions
using ProgressMeter
using SpecialFunctions

include("State_Space_Model.jl")
using .state_space_model

Random.seed!(123)

#########################
#  Helper Random Draws
#########################
function rand_draw(dim, Σ)
    # store
    draws = zeros(dim)
    # find rows with non-zero variance
    nonzero_variances = findall(i -> abs(Σ[i,i]) > 0, 1:dim)
    if !isempty(nonzero_variances)
        # Extract the submatrix for the nonzero variances.
        Σ_sub = Σ[nonzero_variances, nonzero_variances]
        # Draw from the multivariate normal for the nonzero indices.
        draws[nonzero_variances] = rand(MvNormal(zeros(length(nonzero_variances)), Σ_sub))
    end
    return draws
end


#########################
#  Diffuse Kalman Filter 
#########################
function diffuse_kalman_filter(y, θ, α1, P1, cycle_order, do_smooth, do_sim_smooth, diffuse_tol = 1e-6)
    Z, H, T, R, Q, P_diffuse = state_space(θ, cycle_order)

    # Dimensions:
    #   Z is (obs_dim, state_dim) 
    #   H is (obs_dim,obs_dim),
    #   T is (state_dim,state_dim), etc.
    #   R is (state_dim,shock_dim)
    #   Q is (shock_dim,shock_dim)
    n_obs    = size(y, 1)
    state_dim = size(T, 1)
    obs_dim   = size(Z, 1)
    
    # Preallocate arrays for filtered states and covariances.
    α_f = zeros(n_obs, state_dim)           # filtered state estimates
    P_f = Vector{Matrix{Float64}}(undef, n_obs)  # filtered state covariances
    P_diff_f = Vector{Matrix{Float64}}(undef, n_obs)  # filtered state covariances for diffuse part

    if do_smooth
        α_s = zeros(n_obs, state_dim)           # smoothed state estimates
        P_s = Vector{Matrix{Float64}}(undef, n_obs)  # smoothed state covariances
        v_f = zeros(n_obs, obs_dim)             # filtered innovations
        F_f = zeros(n_obs, obs_dim)             # filtered innovation variances
        K_f = zeros(n_obs, obs_dim, state_dim)  # filtered Kalman gains
        F_diff_f = zeros(n_obs, obs_dim)      # filtered F_diff
        K_diff_f = zeros(n_obs, obs_dim, state_dim)  # filtered Kalman gains for F_diff
    end


    # --- Transformation if H is not diagonal ---
    # Check if H is (exactly) diagonal
    if size(H,1) > 1
        # Check if H is diagonal
        if !isdiag(H)
            # Schur decomposition
            Λ,M  = schur(H)
            y = y * M
            Z = M' * Z
            H = Λ
        end
    end
    # --- End transformation ---

    # --- Initialize ---
    P_diff = P_diffuse
    # Set variance off the diffuse parts to zero
    P1[P_diff .==1].=0
    P = P1
    α = α1
    
    α_f[1, :] = α
    P_f[1] = P
    P_diff_f[1] = P_diffuse
    LogL = 0.0



    # ---- Simulation smoother data series  Koopman (2002) ----
    if do_sim_smooth 
        α⁺ = zeros(n_obs, state_dim)
        y⁺ = zeros(n_obs, obs_dim)

        for t in 1:n_obs
            # first period draw α⁺ from N(0,P1)
            if t == 1
                α⁺[t, :] = rand_draw(state_dim, P1) #How to handle diffuse here? Now variance of diffuse states is zero
            else
                η = rand(MvNormal(zeros(size(Q,1)), Q))
                α⁺[t, :] = T * α⁺[t-1, :] + R * η 
            end
            ε = rand(MvNormal(zeros(obs_dim), H))
            y⁺[t, :] = Z * α⁺[t, :] + ε
        end
        y = y - y⁺
    end



    # --- Univariate (sequential) Kalman Filter (Durbin & Koopman 2000)  ---
    for t in 1:n_obs
        for i in 1:obs_dim

            # Compute the prediction error variance (a scalar):
            F = Z[i, :]' * P * Z[i, :] + H[i, i]
            F_diff = Z[i, :]' * P_diff * Z[i, :]
            # Compute the Kalman gain (a column vector):
            K = P * Z[i, :]
            K_diff = P_diff * Z[i, :]
            # Compute the innovation (measurement residual)
            v = y[t, i] - dot(Z[i, :], α)

            # --- Update state and covariance and increment likelihood ---
            if F_diff < diffuse_tol # usual Kalman filter
                α = α + K * (v / F)
                P = P - K * (K' / F)
                LogL += -0.5 * (log(2π) + log(F) + (v^2)/F)
            else # diffuse Kalman filter
                α = α + K_diff / F_diff * v
                P = P + K_diff * K_diff' * F / (F_diff^2) - (K*K_diff' + K_diff*K') / F_diff
                P_diff = P_diff - K_diff * K_diff' / F_diff
                LogL += -0.5 * (log(2π) + log(F_diff))
            end 
        
            if do_smooth  #store results for smoother
                v_f[t, i] = v
                F_f[t, i] = F
                K_f[t, i, :] = K
                F_diff_f[t, i] = F_diff
                K_diff_f[t, i, :] = K_diff
            end
          
        end

        # --- Prediction step ---
        α = T * α
        P = T * P * T' + R * Q * R'
        P_diff = T * P_diff * T'



        # Store filtered estimates (for t < n_obs, we store the prediction for the next time step)
        if t < n_obs
            α_f[t+1, :] = α
            P_f[t+1] = P
            P_diff_f[t+1] = P_diff
        end
    end



    # --- Diffuse Kalman Smoother  (Durbin & Koopman 2000)---
    if do_smooth
        # Initialize r and N:
        # double the state dimension to accommodate both usual and diffuse parts
        r = zeros(2*state_dim)
        N = zeros(2*state_dim, 2*state_dim)
        TT = kron(Matrix(I,2,2), T)

        # backwards recursion
        #backwards through time
        for t=n_obs:-1:1
            if t < n_obs
                r = TT' * r
                N = TT' * N * TT
            end

            #backwards through states
            for i=obs_dim:-1:1
                
                K = K_f[t, i, :]
                F = F_f[t, i]
                v = v_f[t, i]
                K_diff = K_diff_f[t, i, :]
                F_diff = F_diff_f[t, i]

                if F_diff_f[t, i] < diffuse_tol  # usual Kalman smoother
                    L = Matrix(I, state_dim, state_dim) - K * Z[i, :]' / F
                    M = [L zeros(size(L)); zeros(size(L)) L]

                    r = [Z[i,:]/F*v;zeros(state_dim)] + M' * r
                    N = [Z[i,:]/F*Z[i,:]'  zeros(state_dim,state_dim) ; zeros(state_dim, 2* state_dim)] + M' * N * M

                else  # diffuse Kalman smoother
                    L = (K_diff * F / F_diff - K) * Z[i,:]' / F_diff
                    L_diff = Matrix(I, state_dim, state_dim) - K_diff * Z[i,:]' / F_diff
                    M = [L_diff L; zeros(size(L)) L_diff]

                    r = [zeros(state_dim); Z[i,:]/F_diff*v] + M' * r
                    x = Z[i,:]/F_diff*Z[i,:]'
                    y = Z[i,:]/(F_diff^2)*Z[i,:]'*F
                    N = [zeros(state_dim, state_dim) x ; x  y] + M' * N * M
                end
            end
        
            # smoothed state and covariance
            P_dagger = [P_f[t] P_diff_f[t]]
            α_s[t, :] = α_f[t, :] + P_dagger * r
            P_s[t] = P_f[t] - P_dagger * N * P_dagger'
        end
    end

    # --- States and covariances ---
    if do_sim_smooth
        # Get sate draws from the simulation smoother (Koopman 2002)
        α = α_s + α⁺
        P = P_s
    elseif do_smooth
        α = α_s
        P = P_s
    else
        α = α_f
        P = P_f
    end

    return LogL, α, P

end


#########################
# Negative Likelihood for MLE
#########################

function neg_log_likelihood(θ, y,  α0 , P0, cycle_order)
    LogL, _, _ = diffuse_kalman_filter(y,θ,α0, P0, cycle_order, false, false)
    return -LogL
end










#########################
#  Normal Kalman Filter
#########################
function kalman_filter(y, θ,  α0 , P0, cycle_order)
    Z, H, T, R, Q, P_diffuse = state_space(θ, cycle_order)
    n_obs = size(y, 1)
    state_dim = size(T, 1)
    obs_dim = size(Z, 1)
    
    # Preallocate arrays.
    α_p = zeros(n_obs, state_dim) # α[t|t-1]
    P_p = Vector{Matrix{Float64}}(undef, n_obs)  # P[t|t-1]
    α_f = zeros(n_obs,state_dim) # α[t|t]
    P_f = Vector{Matrix{Float64}}(undef, n_obs)  # P[t|t]

    α = α0
    P = P0
    LogL = 0.0

    for t in 1:n_obs
        # Prediction step
        α_p[t, :] = T * α
        P_p[t] = T * P * T' + R * Q * R'

        # Update step
        yhat = Z * α_p[t, :]
        v = y[t, :] - yhat
        F = Z * P_p[t] * Z' + H
        K = P_p[t] * Z' * inv(F)
        α = α_p[t, :] + K * v
        P = P_p[t] - K * Z * P_p[t]

        # Store results
        α_f[t, :] = α
        P_f[t] = P

        # Check if F is valid add to likelihood else stop
        if det(F) <= 0
            LogL = -Inf  
            break      
        else
            LogL += -0.5 * (obs_dim*log(2π) + log(det(F)) + v' * inv(F) * v)
        end
        
    end

    return LogL, α_f, P_f, α_p, P_p
    
end

#########################
# Kalman Smoother (Rauch–Tung–Striebel)
#########################

function kalman_smoother(θ, cycle_order, α_f, P_f, α_p, P_p)
    Z, H, T, R, Q = state_space(θ, cycle_order)

    n_obs, state_dim = size(α_f)

    α_s = zeros(n_obs, state_dim)  #α[t|T]
    P_s = Vector{Matrix{Float64}}(undef, n_obs)  #P[t|T]


    α_s[n_obs, :] = α_f[n_obs, :]
    P_s[n_obs]    = P_f[n_obs]

    for t in (n_obs-1):-1:1
        J = P_f[t] * transpose(T) * inv(P_p[t+1])
        α_s[t, :] = α_f[t, :] + J*(α_s[t+1,:] - α_p[t+1,:])
        P_s[t]    = P_f[t] + J*(P_s[t+1] - P_p[t+1])*transpose(J)
    end

    return α_s, P_s
end

end
