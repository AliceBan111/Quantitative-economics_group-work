using QuantEcon, Optim, Interpolations, Random, Statistics, LinearAlgebra, Plots, DataFrames, Printf, Distributions
cd(@__DIR__)
# 1 Model Solution

function solve_model(γ, R, β, ρ, σ_ϵ, Nz, Na, θ, a_max, method; verbose=true)
    start_time = time()

    # 1. Discretize income using Rouwenhorst
    mc = rouwenhorst(Nz, ρ, σ_ϵ)
    z_nodes = exp.(mc.state_values) 
    P = mc.p

    # 2. Calculate borrowing limit
    z_min = minimum(z_nodes)
    a_underline = -0.6 * (z_min / (R - 1))
    
    if verbose
        println("  z_min = $(round(z_min, digits=4))")
        println("  Borrowing limit = $(round(a_underline, digits=4))")
    end

    # 3. Construct asset grid with curvature
    ω = range(0, 1, length=Na)
    a_grid = a_underline .+ (a_max - a_underline) .* ω.^θ

    # 4. Initialize value function with reasonable guess
    V_old = zeros(Na, Nz)
    for j in 1:Nz
        for i in 1:Na
            # Use flow utility as initial guess
            c_approx = max(R * a_grid[i] + z_nodes[j] - a_grid[i], 0.1)
            if γ ≈ 1.0
                V_old[i, j] = log(c_approx) / (1 - β)
            else
                V_old[i, j] = (c_approx^(1-γ) / (1-γ)) / (1 - β)
            end
        end
    end
    
    V_new = copy(V_old)
    pol_a = zeros(Na, Nz) 
    pol_c = zeros(Na, Nz)
    err = Inf
    iter = 0
    max_iter = 5000
    tol = 1e-10

    # Utility function
    function utility(c, γ)
        c = max(c, 1e-10)
        if γ ≈ 1.0
            return log(c)
        else
            return c^(1-γ) / (1-γ)
        end
    end

    # Value function iteration
    while err > tol && iter < max_iter
        iter += 1
        
        # Compute expected value based on method
        if method == :standard
            # Standard: E[V(a', z')]
            EV = V_old * P'  # (Na × Nz)
        else  # :ces
            # CES transformation: E[V(a', z')^(1-γ)]
            if γ > 1
                # V is negative for γ > 1, handle carefully
                Ṽ = sign.(V_old) .* (abs.(V_old) .^ (1 - γ))
            else
                Ṽ = V_old .^ (1 - γ)
            end
            EV = Ṽ * P'
        end

        # Bellman operator
        for j in 1:Nz
            interp_ev = LinearInterpolation(a_grid, EV[:, j], extrapolation_bc=Line())
            
            for i in 1:Na
                budget = R * a_grid[i] + z_nodes[j]
                a_max_choice = budget - 1e-10  # Ensure c > 0

                if method == :standard
                    # Standard Bellman: V(a,z) = max_c [u(c) + β E V(a',z')]
                    obj = a_p -> begin
                        c = budget - a_p
                        if c <= 1e-10
                            return 1e10
                        end
                        return -(utility(c, γ) + β * interp_ev(a_p))
                    end
                else  # :ces
                    # CES Bellman: V(a,z) = [(1-β)c^(1-γ) + β E Ṽ(a',z')]^(1/(1-γ))
                    obj = a_p -> begin
                        c = budget - a_p
                        if c <= 1e-10
                            return 1e10
                        end
                        u_term = (1 - β) * c^(1 - γ)
                        ev_term = β * interp_ev(a_p)
                        total = u_term + ev_term
                        
                        # Return value with correct sign
                        val = sign(total) * (abs(total)^(1 / (1 - γ)))
                        return -val
                    end
                end
                
                # Maximize using Brent's method
                res = optimize(obj, a_underline, a_max_choice, Brent())
                
                V_new[i, j] = -Optim.minimum(res)
                pol_a[i, j] = Optim.minimizer(res)
                pol_c[i, j] = budget - pol_a[i, j]
            end
        end

        # Convergence check
        err = maximum(abs.(V_new .- V_old) ./ (1 .+ abs.(V_old)))
        
        if verbose && (iter % 200 == 0 || iter <= 5)
            @printf("  Iter %4d: err = %.4e\n", iter, err)
        end
        
        V_old .= V_new
    end
    
    runtime = time() - start_time
    
    if iter >= max_iter
        @warn "Maximum iterations reached! Final error = $err"
    end

    if verbose
        @printf("Converged in %d iterations (%.3f seconds)\n", iter, runtime)
    end

    return (V=V_new, pol_a=pol_a, pol_c=pol_c, a_grid=a_grid, 
            z_nodes=z_nodes, P=P, iter=iter, runtime=runtime,
            a_underline=a_underline, γ=γ, R=R, β=β)
end

# Euler equation errors

function compute_euler_errors(sol)

    γ, R, β = sol.γ, sol.R, sol.β
    a_grid = sol.a_grid
    z_nodes = sol.z_nodes
    pol_c = sol.pol_c
    pol_a = sol.pol_a
    P = sol.P
    Na, Nz = size(pol_c)
    
    ee_errors = zeros(Na, Nz)
    
    # Marginal utility
    up(c) = max(c, 1e-10)^(-γ)
    
    for j in 1:Nz
        # Interpolate consumption policy for next period
        interp_c = LinearInterpolation(a_grid, pol_c[:, j], extrapolation_bc=Line())
        
        for i in 1:Na
            c_today = pol_c[i, j]
            a_tomorrow = pol_a[i, j]
            
            # Compute expected marginal utility tomorrow
            E_up_tomorrow = 0.0
            for j_next in 1:Nz
                c_tomorrow = interp_c(a_tomorrow)
                E_up_tomorrow += P[j, j_next] * up(c_tomorrow)
            end
            
            # Euler equation error
            lhs = up(c_today)
            rhs = β * R * E_up_tomorrow
            
            if abs(lhs) > 1e-10
                ee_errors[i, j] = abs(1 - rhs / lhs)
            else
                ee_errors[i, j] = 0.0
            end
        end
    end
    
    return log10.(ee_errors .+ 1e-16)  # Return log10 of errors
end

# 3 Simulation

function simulate_model(sol, T, burn_in; seed=1234)
    """
    Simulate the model for T periods with burn-in.
    
    Returns time series of assets, consumption, and income.
    """
    Random.seed!(seed)
    
    a_grid = sol.a_grid
    z_nodes = sol.z_nodes
    pol_a = sol.pol_a
    pol_c = sol.pol_c
    P = sol.P
    Nz = length(z_nodes)
    
    # Initialize
    a_sim = zeros(T)
    c_sim = zeros(T)
    z_sim = zeros(T)
    z_idx = zeros(Int, T)
    
    # Start at median income state and a=0
    z_idx[1] = (Nz + 1) ÷ 2
    a_sim[1] = 0.0
    z_sim[1] = z_nodes[z_idx[1]]
    
    # Interpolate policies
    interp_a = [LinearInterpolation(a_grid, pol_a[:, j], extrapolation_bc=Line()) for j in 1:Nz]
    interp_c = [LinearInterpolation(a_grid, pol_c[:, j], extrapolation_bc=Line()) for j in 1:Nz]
    
    # Simulate
    for t in 1:(T-1)
        # Policy functions
        c_sim[t] = interp_c[z_idx[t]](a_sim[t])
        a_next = interp_a[z_idx[t]](a_sim[t])
        
        # Draw next income state
        z_idx[t+1] = rand(Categorical(P[z_idx[t], :]))
        z_sim[t+1] = z_nodes[z_idx[t+1]]
        a_sim[t+1] = a_next
    end
    
    # Last period consumption
    c_sim[T] = interp_c[z_idx[T]](a_sim[T])
    
    # Drop burn-in
    a_sim = a_sim[(burn_in+1):end]
    c_sim = c_sim[(burn_in+1):end]
    z_sim = z_sim[(burn_in+1):end]
    
    return (a=a_sim, c=c_sim, z=z_sim)
end

function compute_simulation_stats(sim, sol)
    a_tol = 1e-3
    
    stats = Dict(
        "Mean assets" => mean(sim.a),
        "SD assets" => std(sim.a),
        "Min assets" => minimum(sim.a),
        "Max assets" => maximum(sim.a),
        "Mean consumption" => mean(sim.c),
        "SD consumption" => std(sim.c),
        "Min consumption" => minimum(sim.c),
        "Max consumption" => maximum(sim.c),
        "Fraction at constraint" => mean(sim.a .<= sol.a_underline + a_tol),
        "Corr(a_t, a_{t-1})" => cor(sim.a[2:end], sim.a[1:end-1]),
        "Corr(c_t, c_{t-1})" => cor(sim.c[2:end], sim.c[1:end-1]),
        "Corr(c_t, z_t)" => cor(sim.c, sim.z),
        "Corr(a_t, z_t)" => cor(sim.a, sim.z)
    )
    
    return stats
end

function compute_euler_errors_simulation(sim, sol)
    γ, R, β = sol.γ, sol.R, sol.β
    a_grid = sol.a_grid
    z_nodes = sol.z_nodes
    pol_c = sol.pol_c
    P = sol.P
    Nz = length(z_nodes)
    
    T = length(sim.a) - 1  # Need t and t+1
    ee_errors = zeros(T)
    
    up(c) = max(c, 1e-10)^(-γ)
    
    # Interpolate consumption policies
    interp_c = [LinearInterpolation(a_grid, pol_c[:, j], extrapolation_bc=Line()) for j in 1:Nz]
    
    for t in 1:T
        c_today = sim.c[t]
        a_tomorrow = sim.a[t+1]
        
        # Find current income state (approximate)
        z_idx = argmin(abs.(z_nodes .- sim.z[t]))
        
        # Expected marginal utility tomorrow
        E_up_tomorrow = 0.0
        for j_next in 1:Nz
            c_tomorrow = interp_c[j_next](a_tomorrow)
            E_up_tomorrow += P[z_idx, j_next] * up(c_tomorrow)
        end
        
        lhs = up(c_today)
        rhs = β * R * E_up_tomorrow
        
        if abs(lhs) > 1e-10
            ee_errors[t] = abs(1 - rhs / lhs)
        end
    end
    
    return log10.(ee_errors .+ 1e-16)
end

# 4 Plot

function plot_policies(sol, method_name)
    a_grid = sol.a_grid
    z_nodes = sol.z_nodes
    pol_c = sol.pol_c
    pol_a = sol.pol_a
    Nz = length(z_nodes)
    
    # Select three income states
    idx_low = 1
    idx_mid = (Nz + 1) ÷ 2
    idx_high = Nz
    
    # Full plots
    p1 = plot(a_grid, pol_c[:, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Consumption c(a,z)",
              title="Consumption Policy - $method_name (γ=$(sol.γ))", legend=:bottomright)
    plot!(p1, a_grid, pol_c[:, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p1, a_grid, pol_c[:, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    
    p2 = plot(a_grid, pol_a[:, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Next period assets a'(a,z)",
              title="Savings Policy - $method_name (γ=$(sol.γ))", legend=:bottomright)
    plot!(p2, a_grid, pol_a[:, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p2, a_grid, pol_a[:, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    plot!(p2, a_grid, a_grid, label="45° line", linestyle=:dash, color=:black, linewidth=1)
    
    # Zoomed plots near borrowing constraint
    zoom_idx = findfirst(a_grid .> sol.a_underline + 5)
    a_zoom = a_grid[1:zoom_idx]
    
    p3 = plot(a_zoom, pol_c[1:zoom_idx, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Consumption c(a,z)",
              title="Consumption (Zoomed) - $method_name (γ=$(sol.γ))", legend=:bottomright)
    plot!(p3, a_zoom, pol_c[1:zoom_idx, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p3, a_zoom, pol_c[1:zoom_idx, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    
    p4 = plot(a_zoom, pol_a[1:zoom_idx, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Next period assets a'(a,z)",
              title="Savings (Zoomed) - $method_name (γ=$(sol.γ))", legend=:bottomright)
    plot!(p4, a_zoom, pol_a[1:zoom_idx, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p4, a_zoom, pol_a[1:zoom_idx, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    plot!(p4, a_zoom, a_zoom, label="45° line", linestyle=:dash, color=:black, linewidth=1)
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
end

function plot_euler_errors(sol, method_name)
    a_grid = sol.a_grid
    z_nodes = sol.z_nodes
    Nz = length(z_nodes)
    
    ee_errors = compute_euler_errors(sol)
    
    # Select three income states
    idx_low = 1
    idx_mid = (Nz + 1) ÷ 2
    idx_high = Nz
    
    # Full plot
    p1 = plot(a_grid, ee_errors[:, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Log10 Euler Error",
              title="Euler Errors - $method_name (γ=$(sol.γ))", legend=:topright)
    plot!(p1, a_grid, ee_errors[:, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p1, a_grid, ee_errors[:, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    
    # Zoomed plot
    zoom_idx = findfirst(a_grid .> sol.a_underline + 5)
    a_zoom = a_grid[1:zoom_idx]
    
    p2 = plot(a_zoom, ee_errors[1:zoom_idx, idx_low], label="z = $(round(z_nodes[idx_low], digits=2))",
              linewidth=2, xlabel="Assets (a)", ylabel="Log10 Euler Error",
              title="Euler Errors (Zoomed) - $method_name (γ=$(sol.γ))", legend=:topright)
    plot!(p2, a_zoom, ee_errors[1:zoom_idx, idx_mid], label="z = $(round(z_nodes[idx_mid], digits=2))", linewidth=2)
    plot!(p2, a_zoom, ee_errors[1:zoom_idx, idx_high], label="z = $(round(z_nodes[idx_high], digits=2))", linewidth=2)
    
    return plot(p1, p2, layout=(1,2), size=(1200, 450))
end

function plot_simulation(sim1, sim2, sol, T_plot=100)
    t_range = 1:T_plot
    
    p1 = plot(t_range, sim1.a[t_range], label="Standard", linewidth=2,
              xlabel="Time", ylabel="Assets", title="Assets (γ=$(sol.γ))")
    plot!(p1, t_range, sim2.a[t_range], label="CES", linewidth=2, linestyle=:dash)
    
    p2 = plot(t_range, sim1.c[t_range], label="Standard", linewidth=2,
              xlabel="Time", ylabel="Consumption", title="Consumption (γ=$(sol.γ))")
    plot!(p2, t_range, sim2.c[t_range], label="CES", linewidth=2, linestyle=:dash)
    
    p3 = plot(t_range, sim1.z[t_range], label="Income z", linewidth=2, color=:black,
              xlabel="Time", ylabel="Income", title="Income Process (γ=$(sol.γ))")
    
    ee1 = compute_euler_errors_simulation(sim1, sol)
    ee2 = compute_euler_errors_simulation(sim2, sol)
    
    p4 = plot(t_range, ee1[t_range], label="Standard", linewidth=2,
              xlabel="Time", ylabel="Log10 Euler Error", title="Euler Errors (γ=$(sol.γ))")
    plot!(p4, t_range, ee2[t_range], label="CES", linewidth=2, linestyle=:dash)
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
end

# Main

# Parameters
scenarios = [
    (γ = 2.0,  R = 1.010),
    (γ = 10.0, R = 1.008)
]
β = 0.99
ρ = 0.90
σ_ϵ = 0.20 * sqrt(1 - ρ^2)
Nz = 11
Na = 100
θ = 3.0
a_max = 500.0

methods = [:standard, :ces]

# Solve all models

results = Dict()

for scenario in scenarios
    println("\n>>> γ = $(scenario.γ), R = $(scenario.R)")
    for method in methods
        println("\n  Method: $(uppercase(string(method)))")
        sol = solve_model(scenario.γ, scenario.R, β, ρ, σ_ϵ, Nz, Na, θ, a_max, method, verbose=true)
        results[(scenario.γ, method)] = sol
    end
end

# Summary table
 
@printf("%-15s %-8s %-8s %-12s\n", "Method", "γ", "Iters", "Time (s)")
for scenario in scenarios
    for method in methods
        sol = results[(scenario.γ, method)]
        @printf("%-15s %-8.1f %-8d %-12.4f\n", string(method), scenario.γ, sol.iter, sol.runtime)
    end
end

# Generate plots
for scenario in scenarios
    γ = scenario.γ
    
    for method in methods
        sol = results[(γ, method)]
        
        # Policy function plots
        p = plot_policies(sol, uppercase(string(method)))
        savefig(p, "$(Int(γ))_$(method).png")
        
        # Euler error plots
        p = plot_euler_errors(sol, uppercase(string(method)))
        savefig(p, "$(Int(γ))_$(method).png")
    end
end

# Simulations
T_sim = 10000
burn_in = 500
seed = 42

simulations = Dict()

for scenario in scenarios
    γ = scenario.γ
    
    for method in methods
        sol = results[(γ, method)]
        sim = simulate_model(sol, T_sim, burn_in, seed=seed)
        simulations[(γ, method)] = sim
        println("$(uppercase(string(method))): T = $T_sim, burn-in = $burn_in")
    end
end

# Simulation statistics
all_stats_rows = []

for scenario in scenarios
    γ = scenario.γ
    for method in methods
        sol = results[(γ, method)]
        sim = simulations[(γ, method)]
        stats = compute_simulation_stats(sim, sol)
        
        # Euler errors in simulation
        ee_sim = compute_euler_errors_simulation(sim, sol)
        ee_sim_clean = ee_sim[isfinite.(ee_sim)]
        
        row = DataFrame(
            γ = γ,
            Method = string(method),
            iterations = sol.iter,
            runtime = sol.runtime,
            mean_a = stats["Mean assets"],
            sd_a = stats["SD assets"],
            min_a = stats["Min assets"],
            max_a = stats["Max assets"],
            mean_c = stats["Mean consumption"],
            sd_c = stats["SD consumption"],
            min_c = stats["Min consumption"],
            max_c = stats["Max consumption"],
            frac_constrained = stats["Fraction at constraint"],
            corr_a_lag = stats["Corr(a_t, a_{t-1})"],
            corr_c_lag = stats["Corr(c_t, c_{t-1})"],
            corr_c_z = stats["Corr(c_t, z_t)"],
            corr_a_z = stats["Corr(a_t, z_t)"],
            ee_mean = mean(ee_sim_clean),
            ee_median = median(ee_sim_clean),
            ee_p10 = quantile(ee_sim_clean, 0.1),
            ee_p90 = quantile(ee_sim_clean, 0.9)
        )
        
        push!(all_stats_rows, row)
    end
end

all_stats = vcat(all_stats_rows...)

# Save statistics table
using CSV
CSV.write("simulation_statistics.csv", all_stats)

# Plot simulation paths
for scenario in scenarios
    γ = scenario.γ
    
    sim_std = simulations[(γ, :standard)]
    sim_ces = simulations[(γ, :ces)]
    sol = results[(γ, :standard)]
    
    p = plot_simulation(sim_std, sim_ces, sol, 100)
    savefig(p, "$(Int(γ)).png")
end

# Create detailed comparison table

comparison = DataFrame(
    Statistic = String[],
    γ2_Standard = Float64[],
    γ2_CES = Float64[],
    γ10_Standard = Float64[],
    γ10_CES = Float64[]
)

stat_names = [
    "Iterations",
    "Runtime",
    "Mean assets",
    "SD assets",
    "Min assets",
    "Max assets",
    "Mean consumption",
    "SD consumption",
    "Min consumption",
    "Max consumption",
    "Fraction constrained",
    "Corr(a_t, a_{t-1})",
    "Corr(c_t, c_{t-1})",
    "Corr(c_t, z_t)",
    "Corr(a_t, z_t)",
    "Mean Euler error",
    "Median Euler error",
    "P10 Euler error",
    "P90 Euler error"
]

col_map = Dict(
    "Iterations" => :iterations,
    "Runtime" => :runtime,
    "Mean assets" => :mean_a,
    "SD assets" => :sd_a,
    "Min assets" => :min_a,
    "Max assets" => :max_a,
    "Mean consumption" => :mean_c,
    "SD consumption" => :sd_c,
    "Min consumption" => :min_c,
    "Max consumption" => :max_c,
    "Fraction constrained" => :frac_constrained,
    "Corr(a_t, a_{t-1})" => :corr_a_lag,
    "Corr(c_t, c_{t-1})" => :corr_c_lag,
    "Corr(c_t, z_t)" => :corr_c_z,
    "Corr(a_t, z_t)" => :corr_a_z,
    "Mean Euler error" => :ee_mean,
    "Median Euler error" => :ee_median,
    "P10 Euler error" => :ee_p10,
    "P90 Euler error" => :ee_p90
)

for stat_name in stat_names
    col = col_map[stat_name]
    push!(comparison, (
        stat_name,
        all_stats[all_stats.γ .== 2.0 .&& all_stats.Method .== "standard", col][1],
        all_stats[all_stats.γ .== 2.0 .&& all_stats.Method .== "ces", col][1],
        all_stats[all_stats.γ .== 10.0 .&& all_stats.Method .== "standard", col][1],
        all_stats[all_stats.γ .== 10.0 .&& all_stats.Method .== "ces", col][1]
    ))
end

println(comparison)
CSV.write("comparison_table.csv", comparison)

include("task2.jl")
