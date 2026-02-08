# Individual MPC computation
function compute_mpc(pol_c, a_grid, z_idx, a, Δ)
    interp_c = LinearInterpolation(a_grid, pol_c[:, z_idx], extrapolation_bc=Line())
    c_0 = interp_c(a)
    c_1 = interp_c(a + Δ)
    return (c_1 - c_0) / Δ
end

function compute_mpc_grid(pol_c, a_grid, transfer_sizes)
    Na, Nz = size(pol_c)
    MPCs = zeros(Na, Nz, length(transfer_sizes))
    
    for (k, Δ) in enumerate(transfer_sizes)
        for j = 1:Nz, i = 1:Na
            MPCs[i, j, k] = compute_mpc(pol_c, a_grid, j, a_grid[i], Δ)
        end
    end
    return MPCs
end

# Stationary distribution computation
function locate_on_grid(x, grid)
    N = length(grid)
    x <= grid[1] && return 1, 1, 1.0
    x >= grid[end] && return N, N, 1.0
    
    i_high = searchsortedfirst(grid, x)
    i_low = i_high - 1
    weight_low = (grid[i_high] - x) / (grid[i_high] - grid[i_low])
    
    return i_low, i_high, weight_low
end

function compute_stationary_distribution(pol_a, a_grid, P; max_iter=100000, tol=1e-7)
    Na, Nz = size(pol_a)
    λ = ones(Na, Nz) / (Na * Nz)
    
    for iter = 1:max_iter
        λ_new = zeros(Na, Nz)
        
        for j = 1:Nz, i = 1:Na
            mass = λ[i, j]
            mass < 1e-16 && continue
            
            a_next = pol_a[i, j]
            i_low, i_high, weight_low = locate_on_grid(a_next, a_grid)
            
            for j_next = 1:Nz
                prob = P[j, j_next]
                prob > 0 || continue
                
                λ_new[i_low, j_next] += mass * prob * weight_low
                i_low != i_high && (λ_new[i_high, j_next] += mass * prob * (1 - weight_low))
            end
        end
        
        maximum(abs.(λ_new .- λ)) < tol && (println("  Distribution converged in $iter iterations"); return λ_new)
        λ .= λ_new
    end
    
    error("Distribution did not converge")
end

# Aggregate dynamics
function shift_distribution(λ, a_grid, Δ)
    Na, Nz = size(λ)
    λ_shifted = zeros(Na, Nz)
    
    for j = 1:Nz, i = 1:Na
        mass = λ[i, j]
        mass < 1e-16 && continue
        
        a_new = a_grid[i] + Δ
        i_low, i_high, weight_low = locate_on_grid(a_new, a_grid)
        
        λ_shifted[i_low, j] += mass * weight_low
        i_low != i_high && (λ_shifted[i_high, j] += mass * (1 - weight_low))
    end
    
    return λ_shifted
end

function iterate_distribution_forward(λ, pol_a, a_grid, P)
    Na, Nz = size(λ)
    λ_next = zeros(Na, Nz)
    
    for j = 1:Nz, i = 1:Na
        mass = λ[i, j]
        mass < 1e-16 && continue
        
        a_next = pol_a[i, j]
        i_low, i_high, weight_low = locate_on_grid(a_next, a_grid)
        
        for j_next = 1:Nz
            prob = P[j, j_next]
            prob > 0 || continue
            
            λ_next[i_low, j_next] += mass * prob * weight_low
            i_low != i_high && (λ_next[i_high, j_next] += mass * prob * (1 - weight_low))
        end
    end
    
    return λ_next
end

function simulate_impulse_response(pol_c, pol_a, a_grid, P, λ_star, Δ, T)
    C_star = sum(λ_star .* pol_c)
    λ_t = shift_distribution(λ_star, a_grid, Δ)
    
    C_path = zeros(T + 1)
    C_path[1] = sum(λ_t .* pol_c)
    
    for t = 1:T
        λ_t = iterate_distribution_forward(λ_t, pol_a, a_grid, P)
        C_path[t+1] = sum(λ_t .* pol_c)
    end
    return C_path, C_path .- C_star, C_star
end

# Plotting functions
function plot_mpcs_problem2(MPCs, a_grid, z_nodes, transfer_sizes, γ)
    Nz = length(z_nodes)
    idx_low, idx_mid, idx_high = 1, (Nz+1)÷2, Nz
    
    p = plot(layout=(3,1), size=(800, 1200))
    
    for (subplot, idx, label) in [(1, idx_low, "Low"), (2, idx_mid, "Med"), (3, idx_high, "High")]
        for (k, Δ) in enumerate(transfer_sizes)
            plot!(p[subplot], a_grid, MPCs[:, idx, k], label="Δ=$Δ", linewidth=2)
        end
        plot!(p[subplot], xlabel="Assets", ylabel="MPC", 
             title="$label income (z=$(round(z_nodes[idx], digits=3)))")
    end
    
    return p
end

function plot_impulse_response_problem2(ΔC, C_star, T, γ)
    pct_dev = 100 * ΔC / C_star
    
    p = plot(0:T, pct_dev, xlabel="Periods", ylabel="% deviation from C*",
            title="Impulse Response (γ=$γ)", label="", linewidth=2.5, size=(900, 500))
    hline!([0], color=:black, linestyle=:dash, linewidth=1, label="")
    
    return p
end

# Main
function run_problem2_analysis(γ_select=2.0, method_select=:standard)
    println("PROBLEM 2: MPC ANALYSIS (γ=$γ_select, method=$method_select)")

    
    # Get solution from Problem 1
    sol = results[(γ_select, method_select)]
    pol_c, pol_a, a_grid, z_nodes, P = sol.pol_c, sol.pol_a, sol.a_grid, sol.z_nodes, sol.P
    
    # Part A: Individual MPCs
    println("\n[Part A] Computing Individual MPCs...")
    transfer_sizes = [0.01, 0.1, 0.5, 1.0, 2.0]
    MPCs = compute_mpc_grid(pol_c, a_grid, transfer_sizes)
    
    # Print summary
    Nz = length(z_nodes)
    for (idx, label) in [(1, "Low"), ((Nz+1)÷2, "Med"), (Nz, "High")]
        println("\n$label income (z=$(round(z_nodes[idx], digits=3))):")
        for (k, Δ) in enumerate(transfer_sizes)
            @printf("  Δ=%.2f: mean=%.4f, min=%.4f, max=%.4f\n", 
                   Δ, mean(MPCs[:, idx, k]), minimum(MPCs[:, idx, k]), maximum(MPCs[:, idx, k]))
        end
    end
    
    # Plot MPCs
    p_mpc = plot_mpcs_problem2(MPCs, a_grid, z_nodes, transfer_sizes, γ_select)
    savefig(p_mpc, "problem2_mpcs_gamma$(Int(γ_select))_$(method_select).png")
    
    # Part B: Stationary distribution
    println("\n[Part B] Computing Stationary Distribution")
    λ_star = compute_stationary_distribution(pol_a, a_grid, P)
    C_star = sum(λ_star .* pol_c)
    println("  Steady state consumption C* = $(round(C_star, digits=4))")
    
    # Part C: Aggregate response
    println("\n[Part C] Simulating Aggregate Response")
    Δ = 0.05 * C_star
    T = 50
    println("  Transfer size Δ = $(round(Δ, digits=4)) (5% of C*)")
    
    C_path, ΔC, _ = simulate_impulse_response(pol_c, pol_a, a_grid, P, λ_star, Δ, T)
    
    # Aggregate MPC
    MPC_0 = ΔC[1] / Δ
    println("  Impact MPC = $(round(MPC_0, digits=4))")
    
    # Cumulative MPCs
    println("\n  Cumulative MPCs:")
    for H in [0, 1, 4, 8, 12, 20]
        MPC_H = sum(ΔC[1:(H+1)]) / Δ
        @printf("    H=%2d: MPC=%.4f (%.1f%% spent)\n", H, MPC_H, 100*MPC_H)
    end
    
    frac_12 = sum(ΔC[1:13]) / Δ
    println("\nFraction spent by H=12: $(round(100*frac_12, digits=2))%")
    
    # Plot impulse response
    p_ir = plot_impulse_response_problem2(ΔC, C_star, T, γ_select)
    savefig(p_ir, "problem2_impulse_gamma$(Int(γ_select))_$(method_select).png")

    
    return (MPCs=MPCs, λ_star=λ_star, C_path=C_path, ΔC=ΔC, MPC_0=MPC_0)
end

# Execute Problem 2 for γ=2 (using the better method from Problem 1)
println("PROBLEM 2 ANALYSIS")


# Run for γ=2 with standard method (or :ces if that was better in Problem 1)
prob2_results = run_problem2_analysis(2.0, :standard)

# Compare both methods
prob2_results_ces = run_problem2_analysis(2.0, :ces)
