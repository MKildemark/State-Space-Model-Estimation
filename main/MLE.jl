module MLE

using Optim

using Main.state_space_model
using Main.kalman
using Main.ParticleFilter
using Main.MCMC

export mle_estimation





function run_mle(objective, θ_init, lower_bounds, upper_bounds, method::Symbol)
    solver = nothing
    if method == :nelder_mead
        solver = NelderMead()
    elseif method == :simulated_annealing
        solver = SimulatedAnnealing()
    elseif method == :bfgs
        solver = BFGS()
    elseif method == :lbfgs
        solver = LBFGS()
    elseif method == :conjugate_gradient
        solver = ConjugateGradient()
    elseif method == :gradient_descent
        solver = GradientDescent()
    elseif method == :momentum_gradient_descent
        solver = MomentumGradientDescent()
    elseif method == :accelerated_gradient_descent
        solver = AcceleratedGradientDescent()
    else
        error("Unknown optimization method: $method")
    end
    # Use Fminbox to enforce bounds
    res = Optim.optimize(objective, lower_bounds, upper_bounds, θ_init, Fminbox(solver))
    return res
end


function mle_estimation(θ_init, y, X, α0, P0, prior_info, methods::Vector{Symbol})

    objective(θ) = neg_log_likelihood(θ, y, X, α0, P0)

    lower_bounds = prior_info.support[:, 1]
    upper_bounds = prior_info.support[:, 2]
    
    results = Dict{Symbol, Any}()
    for method in methods
        println("Running MLE with optimizer: ", method)
        results[method] = run_mle(objective, θ_init, lower_bounds, upper_bounds, method)
    end
    return results
end

end  # module
