module kalman

export diffuse_kalman_filter

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



function diffuse_kalman_filter(y, θ, α1, P1, cycle_order, σʸ, do_smooth, do_sim_smooth, F_tol = 1e-8)


    # Get state-space matrices
    Z, H, T, R, Q, P_diffuse = state_space(θ, cycle_order, σʸ)

    # Dimensions:
    n_obs    = size(y, 2)
    state_dim = size(T, 1)
    obs_dim   = size(y, 1)
    m = n_obs
    k = state_dim
    n = obs_dim

    # Preallocate arrays for filtered states and covariances.
    α_f = zeros(k, m)           # filtered state estimates
    P_f = zeros(k, k, m)        # filtered state covariances
    P_diff_f = zeros(k, k, m)   # filtered covariances for the diffuse part

    if do_smooth
        α_s = zeros(k, m)       # smoothed state estimates
        P_s = zeros(k, k, m)    # smoothed state covariances
        v_f = zeros(m, n)       # filtered innovations
        F_f = zeros(m, n)       # filtered innovation variances
        K_f = zeros(k, n, m)    # filtered Kalman gains
        F_diff_f = zeros(m, n)  # filtered innovation variances for diffuse part
        K_diff_f = zeros(k, n, m) # filtered gains for diffuse part
    end

    # --- Transformation if H is not diagonal ---
    if size(H,1) > 1
        if !isdiag(H)
            Λ, M = schur(H)
            y = y * M
            Z = M' * Z
            H = Λ
        end
    end

    # --- Initialize filter ---
    P_diff = P_diffuse
    # Set variances off the diffuse parts to zero.
    P1[P_diffuse .== 1] .= 0
    P = P1
    α = α1

    α_f[:,1] = α
    P_f[:,:,1] = P
    P_diff_f[:,:,1] = P_diffuse
    LogL = 0.0

    # ---- Simulation smoother (if enabled) ----
    if do_sim_smooth 
        α⁺ = zeros(k, m)
        y⁺ = zeros(n, m)
        for t in 1:m
            if t == 1
                α⁺[:,t] = rand_draw(k, P1)
            else
                η = rand(MvNormal(zeros(size(Q,1)), Q))
                α⁺[:,t] = T * α⁺[:, t-1] + R * η 
            end
            ε = rand(MvNormal(zeros(n), H))
            y⁺[:, t] = Z * α⁺[:,t] + ε 
        end
        # Subtract the simulation smoother draws from the data.
        y = y - y⁺
    end

    # --- Univariate (sequential) Kalman Filter ---
    for t in 1:m
        for i in 1:n
            # Compute prediction error variance (scalar)
            F = Z[i, :]' * P * Z[i, :] + H[i, i]
            F_diff = Z[i, :]' * P_diff * Z[i, :]
            if F > F_tol || F_diff > F_tol
                # Compute gains and innovation
                K = P * Z[i, :]
                K_diff = P_diff * Z[i, :]
                v = y[i, t] - dot(Z[i, :], α)

                # --- Update state (using the diffuse update if F_diff is “active”) ---
                if F_diff > F_tol
                    # Diffuse Kalman update (matching second filter’s logic)
                    α = α + (K_diff / F_diff) * v
                    P = P + (K_diff * K_diff' * F) / (F_diff^2) - (K * K_diff' + K_diff * K') / F_diff
                    P_diff = P_diff - (K_diff * K_diff') / F_diff
                    LogL += -0.5 * (log(2π) + log(F_diff))
                else
                    # Standard Kalman update
                    α = α + K * (v / F)
                    P = P - K * (K' / F)
                    LogL += -0.5 * (log(2π) + log(F) + (v^2)/F)
                end

                if do_smooth
                    v_f[t, i] = v
                    F_f[t, i] = F
                    K_f[:, i, t] = K
                    F_diff_f[t, i] = F_diff
                    K_diff_f[:, i, t] = K_diff
                end
            end
        end
        # Prediction step (done once per time period, after all observations at t)
        if t < m
            α = T * α
            P = T * P * T' + R * Q * R'
            P_diff = T * P_diff * T'
            α_f[:, t+1] = α
            P_f[:, :, t+1] = P
            P_diff_f[:, :, t+1] = P_diff
        end
    end

    # --- Diffuse Kalman Smoother ---
    if do_smooth
        # Initialize r and N (dimension doubled to accommodate the diffuse part)
        r = zeros(2*k)
        N = zeros(2*k, 2*k)
        TT = kron(Matrix(I,2,2), T)

        # Backward recursion (time)
        for t in m:-1:1
            if t < m
                r = TT' * r
                N = TT' * N * TT
            end

            # Backward recursion over measurement equations
            for i in n:-1:1
                F_val = F_f[t, i]
                F_diff_val = F_diff_f[t, i]
                if F_val > F_tol || F_diff_val > F_tol
                    v_val = v_f[t, i]
                    K_val = K_f[:, i, t]
                    K_diff_val = K_diff_f[:, i, t]
                    if F_diff_val > F_tol
                        # Diffuse smoother update
                        L = (K_diff_val * F_val / F_diff_val - K_val) * (Z[i, :]'/F_diff_val)
                        L_diff = Matrix(I, k, k) - K_diff_val * (Z[i, :]'/F_diff_val)
                        M = [L_diff  L; zeros(size(L_diff))  L_diff]

                        r = [zeros(k); (Z[i, :]/F_diff_val)*v_val] + M' * r
                        x = (Z[i, :]/F_diff_val) * Z[i, :]'
                        y_block = (Z[i, :]/(F_diff_val^2)) * Z[i, :]' * F_val
                        N = [zeros(k, k)  x; x  y_block] + M' * N * M
                    else
                        # Standard smoother update
                        L = Matrix(I, k, k) - K_val * (Z[i, :]'/F_val)
                        M = [L  zeros(k, k); zeros(k, k)  L]

                        r = [(Z[i, :]/F_val)*v_val; zeros(k)] + M' * r
                        N = [(Z[i, :]/F_val)*Z[i, :]'  zeros(k, k);
                             zeros(k, 2*k)] + M' * N * M
                    end
                end
            end

            # Smoothed state and covariance
            P_dagger = [P_f[:, :, t]  P_diff_f[:, :, t]]
            α_s[:, t] = α_f[:, t] + P_dagger * r
            P_s[:, :, t] = P_f[:, :, t] - P_dagger * N * P_dagger'
        end
    end

    # --- Return states and covariances ---
    if do_sim_smooth
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
