using Distributions, Random, StatsPlots, Optim

Random.seed!(2024)

function task3_simulation_and_data(p, P, δL, δH, T)
    bern = zeros(T)
    dist_N1 = Normal(0, δL^2)
    dist_N0 = Normal(0, δH^2)

    for i in 1:T
        bern[i] = rand(Bernoulli(P))
    end

    epsilon = zeros(T)
    for i in 1:T
        if bern[i] == 1
            epsilon[i] = rand(dist_N1)
        else
            epsilon[i] = rand(dist_N0)
        end
    end

    Y = zeros(T)
    Y[1] = 0.0 
    for i in 1:(T - 1)
        Y[i+1] = p * Y[i] + epsilon[i+1]
    end
    burn_in_val = 100
    y_data = Y[(burn_in_val + 1):T]
    ϵ_data = epsilon[(burn_in_val + 1):T] 

    return y_data, ϵ_data
end

function calculate_mean(data::Vector{Float64})
    return sum(data) / length(data)
end

function calculate_std(data::Vector{Float64})
    N = length(data)
    μ = calculate_mean(data)
    variance = sum((x - μ)^2 for x in data) / (N - 1)
    return sqrt(variance)
end

function calculate_autocov(data::Vector{Float64}, lag::Int)
    N = length(data)
    if N <= lag
        return 0.0
    end
    
    μ = calculate_mean(data)
    cov_sum = 0.0
    for t in (lag + 1):N
        cov_sum += (data[t] - μ) * (data[t - lag] - μ)
    end
    
    return cov_sum / N 
end


function compute_smm_moments(ρ, P, δL, δH, T)

    y_data, ϵ_data = task3_simulation_and_data(ρ, P, δL, δH, T)
    
    μ_y = calculate_mean(y_data)
    σ_y = calculate_std(y_data)
    μ_ϵ = calculate_mean(ϵ_data)
    σ_ϵ = calculate_std(ϵ_data)
    γ_ϵ_1 = calculate_autocov(ϵ_data, 1)
    γ_ϵ_2 = calculate_autocov(ϵ_data, 2)

    return [μ_y, σ_y, μ_ϵ, σ_ϵ, γ_ϵ_1, γ_ϵ_2]
end


function SMM_objective_function(θ::Vector{Float64}, g_bar_obs::Vector{Float64}; 
                                S::Int=100, T::Int=500)
    
    ρ, P, δL, δH = θ[1], θ[2], θ[3], θ[4]

    if !(0.8 <= ρ <= 0.99 && 0.5 <= P <= 0.95 && δL > 0 && δH > 0)
        return 1e12 
    end

    g_bar_sum = zeros(length(g_bar_obs))
    
    for s in 1:S
        g_s_theta = compute_smm_moments(ρ, P, δL, δH, T)
        g_bar_sum += g_s_theta
    end
    
    g_bar_sim = g_bar_sum / S
    moment_difference = g_bar_obs - g_bar_sim
    
    J_theta = sum(moment_difference .^ 2)
    return J_theta
end

ρ_true = 0.9; P_true = 0.8; δL_true = 0.1; δH_true = 0.3; T = 500


g_bar_obs = compute_smm_moments(ρ_true, P_true, δL_true, δH_true, T)

ρ_init = 0.95; P_init = 0.75; δL_init = 0.15; δH_init = 0.25
θ_init = [ρ_init, P_init, δL_init, δH_init]
lower_bounds = [0.8, 0.5, 0.001, 0.001]
upper_bounds = [0.99, 0.95, 1.0, 1.0] 

J_to_minimize(θ) = SMM_objective_function(θ, g_bar_obs; S=100, T=T)

println("\n Task 6: SMM Optimization")


results = optimize(J_to_minimize, lower_bounds, upper_bounds, θ_init, Fminbox(NelderMead()))

J_min = Optim.minimum(results)
θ_hat = Optim.minimizer(results)
ρ_hat, P_hat, δL_hat, δH_hat = θ_hat[1], θ_hat[2], θ_hat[3], θ_hat[4]

g_bar_sim_hat_vector = zeros(length(g_bar_obs))
for s in 1:100
    global g_bar_sim_hat_vector += compute_smm_moments(ρ_hat, P_hat, δL_hat, δH_hat, T)
end
g_bar_sim_hat_vector /= 100


println("\n Task 7: Presentation of SMM Results")

println(" Parameter Estimates (SMM)")
println("| Parameter | True Value | Initial Guess | SMM Estimate (\\hat{\\theta}) |")
println("|     |     |     |     |")
println("| ρ | $ρ_true | $ρ_init | $(round(ρ_hat, digits=4)) |")
println("| P | $P_true | $P_init | $(round(P_hat, digits=4)) |")
println("| δL | $δL_true | $δL_init | $(round(δL_hat, digits=4)) |")
println("| δH | $δH_true | $δH_init | $(round(δH_hat, digits=4)) |")

println("\nFinal Objective Function Value J(\\hat{\\theta}): $(round(J_min, digits=8))")

println("\n Moment Comparison")
println("| Moment | Observed (\\bar{g}_T) | Simulated at \\hat{\\theta} (\\bar{g}_S(\\hat{\\theta})) | Absolute Difference |")
println("|     |     |     |     |")
for i in 1:length(g_bar_obs)
    moments_desc = ["μ_y", "σ_y", "μ_ϵ", "σ_ϵ", "γ_ϵ_1", "γ_ϵ_2"]
    diff = abs(g_bar_obs[i] - g_bar_sim_hat_vector[i])
    
    println("| $(moments_desc[i]) | $(round(g_bar_obs[i], digits=5)) | $(round(g_bar_sim_hat_vector[i], digits=5)) | $(round(diff, digits=5)) |")
end
