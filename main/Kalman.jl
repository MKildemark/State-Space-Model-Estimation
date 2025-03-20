module kalman

export diffuse_kalman_filter, kalman_filter, kalman_smoother

include("import_packages.jl")

using Main.state_space_model

#########################
#  Helper Random Draws
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

#########################
#  Diffuse Kalman Filter with Forecasting
#########################
function diffuse_kalman_filter(y, X, θ, α1, P1; 
        do_smooth = true, 
        do_sim_smooth = false, 
        F_tol = 1e-8, 
        forecast_out_sample = 0, 
        forecast_in_sample = 0,
        X_forecast = nothing, 
        rng=Random.GLOBAL_RNG)

    # Get state-space matrices
    Z, H, d, T, R, Q, c, P_diffuse = get_state_space(θ)
    P1 = copy(P1) 

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
    if size(H,1) > 1 && !isdiag(H)
        Λ, M = schur(H)
        y = y * M
        Z = M' * Z
        d = M' * d
        H = Λ
    end

    # --- Initialize filter ---
    P1 = copy(P1)
    P_diff = P_diffuse
    P1[P_diffuse .== 1] .= 0  # Set diffuse-part variances to zero.
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
                α⁺[:,t] = rand_draw(k, P1; rng=rng)
            else
                η = rand(rng, MvNormal(zeros(size(Q,1)), Q))
                α⁺[:,t] = T * α⁺[:, t-1] + R * η 
            end
            ε = rand_draw(n, H; rng=rng)
            y⁺[:, t] = Z * α⁺[:,t] + ε 
        end
        y = y - y⁺  # Subtract simulation smoother draws.
    end

    # --- Univariate (sequential) Kalman Filter ---
    for t in 1:m
        for i in 1:n
            F = Z[i, :]' * P * Z[i, :] + H[i, i]
            F_diff = Z[i, :]' * P_diff * Z[i, :]
            
            if F > F_tol || F_diff > F_tol
                K = P * Z[i, :]
                K_diff = P_diff * Z[i, :]
                v = (y[i, t] .- d[i,:] .- Z[i, :]'* α)[1]
                if F_diff > F_tol
                    α = α + (K_diff / F_diff) * v
                    P = P + (K_diff * K_diff' * F) / (F_diff^2) - (K * K_diff' + K_diff * K') / F_diff
                    P_diff = P_diff - (K_diff * K_diff') / F_diff
                    LogL += -0.5 * (log(2π) + log(F_diff))
                else
                    α = α + K * (v / F)
                    P = P - K * (K' / F)
                    LogL += -0.5 * (log(2π) + log(F) + (v^2)/F)
                end
            else
                v = 0
                K = zeros(k)
                K_diff = zeros(k)
            end
            
            if do_smooth
                v_f[t, i] = v
                F_f[t, i] = F
                K_f[:, i, t] = K
                F_diff_f[t, i] = F_diff
                K_diff_f[:, i, t] = K_diff
            end
        end
        if t < m
            if X !== nothing   # add exogenous data 
                α = T * α + c * X[:, t]
            else
                α = T * α + c 
            end
            P = T * P * T' + R * Q * R'
            P_diff = T * P_diff * T'
            α_f[:, t+1] = α
            P_f[:, :, t+1] = P
            P_diff_f[:, :, t+1] = P_diff
        end
    end

    # --- Diffuse Kalman Smoother (if enabled) ---
    if do_smooth
        r = zeros(2*k)
        N = zeros(2*k, 2*k)
        TT = kron(Matrix(I,2,2), T)
        for t in m:-1:1
            if t < m
                r = TT' * r
                N = TT' * N * TT
            end
            for i in n:-1:1
                F_val = F_f[t, i]
                F_diff_val = F_diff_f[t, i]
                if F_val > F_tol || F_diff_val > F_tol
                    v_val = v_f[t, i]
                    K_val = K_f[:, i, t]
                    K_diff_val = K_diff_f[:, i, t]
                    if F_diff_val > F_tol
                        L = (K_diff_val * F_val / F_diff_val - K_val) * (Z[i, :]'/F_diff_val)
                        L_diff = Matrix(I, k, k) - K_diff_val * (Z[i, :]'/F_diff_val)
                        M_mat = [L_diff  L; zeros(size(L_diff))  L_diff]
                        r = [zeros(k); (Z[i, :]/F_diff_val)*v_val] + M_mat' * r
                        x = (Z[i, :]/F_diff_val) * Z[i, :]'
                        y_block = (Z[i, :]/(F_diff_val^2)) * Z[i, :]' * F_val
                        N = [zeros(k, k)  x; x  y_block] + M_mat' * N * M_mat
                    else
                        L = Matrix(I, k, k) - K_val * (Z[i, :]'/F_val)
                        M_mat = [L  zeros(k, k); zeros(k, k)  L]
                        r = [(Z[i, :]/F_val)*v_val; zeros(k)] + M_mat' * r
                        N = [(Z[i, :]/F_val)*Z[i, :]'  zeros(k, k);
                             zeros(k, 2*k)] + M_mat' * N * M_mat
                    end
                end
            end
            P_dagger = [P_f[:, :, t]  P_diff_f[:, :, t]]
            α_s[:, t] = α_f[:, t] + P_dagger * r
            P_s[:, :, t] = P_f[:, :, t] - P_dagger * N * P_dagger'
        end
    end

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

    # --- Forecasting ---
    if forecast_out_sample > 0 || forecast_in_sample > 0
        forecast_horizon = forecast_out_sample + forecast_in_sample
        Z, H, d, T, R, Q, c, _ = get_state_space(θ)
        α_forecast = zeros(k, forecast_out_sample + forecast_in_sample)
        α_fcast = α[:,end-forecast_in_sample]
        for t in 1:forecast_horizon
            if t <= forecast_in_sample
                if X !== nothing
                    α_fcast = T * α_fcast + c * X[:, end-forecast_in_sample+t]
                else
                    α_fcast = T * α_fcast + c
                end
            else
                if X_forecast !== nothing
                    α_fcast = T * α_fcast + c * X_forecast[:, t-forecast_in_sample]
                else
                    α_fcast = T * α_fcast + c
                end
            end
            α_forecast[:, t] = α_fcast
        end
        # append forecasted states to the end of the state vector
        α = hcat(α[:,1:end-forecast_in_sample], α_forecast)
    end

    return LogL, α, P
end

end  # module kalman
