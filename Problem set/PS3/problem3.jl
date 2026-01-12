using Plots, Printf, Parameters

# Parameters
Base.@kwdef struct LifecycleParams
    J::Int          # Maximum age
    gamma::Float64  # CRRA coefficient for consumption
    gamma_b::Float64 # CRRA coefficient for bequest
    beta::Float64   # Discount factor
    r::Float64      # Interest rate
    theta::Float64  # Weight on bequest utility
    a_bar::Float64  # Bequest shifter
    y_bar::Float64  # Income scale
    na::Int         # Number of asset grid points
    amax::Float64   # Maximum assets
end

# Utility functions
function u(c::Float64, gamma::Float64)
    if c <= 0
        return -1e10
    end
    if gamma == 1.0
        return log(c)
    else
        return c^(1-gamma) / (1-gamma)
    end
end

function bequest_utility(a::Float64, theta::Float64, a_bar::Float64, gamma_b::Float64)
    if gamma_b == 1.0
        return theta * log(a + a_bar)
    else
        return theta * (a + a_bar)^(1-gamma_b) / (1-gamma_b)
    end
end

# Death probabilities
function death_prob(j::Int)
    return min(0.0005 * 1.14^j, 1.0)
end

# Labor income
function labor_income(j::Int, y_bar::Float64)
    if j <= 40
        return y_bar * (0.8 + 0.02 * j)
    else
        return y_bar * 0.3
    end
end

# Create asset grid with quadratic spacing
function create_asset_grid(na::Int, amax::Float64)
    grid = zeros(na)
    for i in 1:na
        grid[i] = amax * ((i-1) / (na-1))^2
    end
    return grid
end

# Solve the lifecycle model using backward induction
function solve_lifecycle(params::LifecycleParams)
    @unpack J, gamma, gamma_b, beta, r, theta, a_bar, y_bar, na, amax = params
    
    # Create asset grid
    a_grid = create_asset_grid(na, amax)
    
    # Initialize value function and policy functions
    V = zeros(J, na)
    c_policy = zeros(J, na)
    a_policy = zeros(J, na)
    
    # Terminal period (J): agent dies with certainty
    pi_J = 1.0
    y_J = labor_income(J, y_bar)
    
    for (ia, a) in enumerate(a_grid)
        cash_on_hand = (1 + r) * a + y_J
        
        best_value = -Inf
        best_c = 0.0
        best_a_prime = 0.0
        
        for (ia_prime, a_prime) in enumerate(a_grid)
            c = cash_on_hand - a_prime
            if c > 0
                value = u(c, gamma) + beta * bequest_utility(a_prime, theta, a_bar, gamma_b)
                if value > best_value
                    best_value = value
                    best_c = c
                    best_a_prime = a_prime
                end
            end
        end
        
        V[J, ia] = best_value
        c_policy[J, ia] = best_c
        a_policy[J, ia] = best_a_prime
    end
    
    # Backward induction for j = J-1, ..., 1
    for j in (J-1):-1:1
        pi_j = death_prob(j)
        y_j = labor_income(j, y_bar)
        
        for (ia, a) in enumerate(a_grid)
            cash_on_hand = (1 + r) * a + y_j
            
            best_value = -Inf
            best_c = 0.0
            best_a_prime = 0.0
            
            for (ia_prime, a_prime) in enumerate(a_grid)
                c = cash_on_hand - a_prime
                if c > 0
                    # Continuation value if alive
                    V_next = V[j+1, ia_prime]
                    
                    # Bequest utility if dead
                    beq_util = bequest_utility(a_prime, theta, a_bar, gamma_b)
                    
                    # Expected value
                    value = u(c, gamma) + beta * ((1 - pi_j) * V_next + pi_j * beq_util)
                    
                    if value > best_value
                        best_value = value
                        best_c = c
                        best_a_prime = a_prime
                    end
                end
            end
            
            V[j, ia] = best_value
            c_policy[j, ia] = best_c
            a_policy[j, ia] = best_a_prime
        end
    end
    
    return V, c_policy, a_policy, a_grid
end

# Simulate lifecycle path for an agent who survives to age J
function simulate_lifecycle(c_policy, a_policy, a_grid, params::LifecycleParams)
    @unpack J, r, y_bar = params
    
    # Initialize
    c_path = zeros(J)
    a_path = zeros(J+1)
    y_path = zeros(J)
    s_path = zeros(J)  # savings
    
    a_path[1] = 0.0  # Start with zero assets
    
    for j in 1:J
        # Find closest grid point
        ia = argmin(abs.(a_grid .- a_path[j]))
        
        # Record income
        y_path[j] = labor_income(j, y_bar)
        
        # Get optimal consumption and savings
        c_path[j] = c_policy[j, ia]
        a_path[j+1] = a_policy[j, ia]
        
        # Calculate savings
        s_path[j] = a_path[j+1] - a_path[j]
    end
    
    return c_path, a_path[1:J], y_path, s_path
end

# Main execution
function main()
    # Set parameters
    params = LifecycleParams(
        J = 60,
        gamma = 2.0,
        gamma_b = 1.0,
        beta = 0.96,
        r = 0.04167,
        theta = 0.5,
        a_bar = 2.0,
        y_bar = 1.0,
        na = 500,
        amax = 100.0  # Experiment to find suitable value
    )
    
    println("Solving lifecycle model...")
    V, c_policy, a_policy, a_grid = solve_lifecycle(params)
    println("Solution complete!")
    
    # Question 3: Plot policy functions for selected ages
    ages_to_plot = [20, 30, 40, 50, 60]
    
    # Plot consumption policy
    p1 = plot(title="Consumption Policy", xlabel="Assets", ylabel="Consumption", legend=:bottomright)
    for j in ages_to_plot
        plot!(p1, a_grid, c_policy[j, :], label="Age $j", linewidth=2)
    end
    
    # Plot savings policy
    p2 = plot(title="Savings Policy", xlabel="Assets", ylabel="Next Period Assets", legend=:bottomright)
    for j in ages_to_plot
        plot!(p2, a_grid, a_policy[j, :], label="Age $j", linewidth=2)
    end
    plot!(p2, a_grid, a_grid, label="45° line", linestyle=:dash, color=:black)
    
    # Plot value function
    p3 = plot(title="Value Function", xlabel="Assets", ylabel="Value", legend=:bottomright)
    for j in ages_to_plot
        plot!(p3, a_grid, V[j, :], label="Age $j", linewidth=2)
    end
    
    display(plot(p1, p2, p3, layout=(1,3), size=(1400, 400)))
    
    # Question 4: Simulate lifecycle profiles
    c_path, a_path, y_path, s_path = simulate_lifecycle(c_policy, a_policy, a_grid, params)
    
    p4 = plot(1:params.J, c_path, label="Consumption", linewidth=2, xlabel="Age", ylabel="Value")
    plot!(p4, 1:params.J, y_path, label="Income", linewidth=2)
    plot!(p4, 1:params.J, a_path, label="Assets", linewidth=2)
    plot!(p4, 1:params.J, s_path, label="Savings", linewidth=2)
    plot!(p4, title="Lifecycle Profiles", legend=:topright)
    display(p4)
    
    # Question 5: Compare across income levels
    income_levels = [0.5, 1.0, 2.0]
    
    p5 = plot(title="Consumption by Income", xlabel="Age", ylabel="Consumption", legend=:topright)
    p6 = plot(title="Assets by Income", xlabel="Age", ylabel="Assets", legend=:topleft)
    p7 = plot(title="Savings Rate by Income", xlabel="Age", ylabel="Savings Rate", legend=:topright)
    
    for y_bar_val in income_levels
        params_temp = LifecycleParams(params.J, params.gamma, params.gamma_b, params.beta, 
                                      params.r, params.theta, params.a_bar, y_bar_val, 
                                      params.na, params.amax)
        _, c_pol, a_pol, _ = solve_lifecycle(params_temp)
        c_path, a_path, y_path, s_path = simulate_lifecycle(c_pol, a_pol, a_grid, params_temp)
        
        savings_rate = s_path ./ y_path
        
        plot!(p5, 1:params.J, c_path, label="y̅ = $y_bar_val", linewidth=2)
        plot!(p6, 1:params.J, a_path, label="y̅ = $y_bar_val", linewidth=2)
        plot!(p7, 1:params.J, savings_rate, label="y̅ = $y_bar_val", linewidth=2)
    end
    
    display(plot(p5, p6, p7, layout=(1,3), size=(1400, 400)))
    
    # Question 6: Compare with and without bequest motive
    theta_values = [0.0, 0.5]
    
    for theta_val in theta_values
        p_assets = plot(title="Assets (θ = $theta_val)", xlabel="Age", ylabel="Assets", legend=:topleft)
        
        peak_wealth = Dict()
        
        for y_bar_val in income_levels
            params_temp = LifecycleParams(params.J, params.gamma, params.gamma_b, params.beta, 
                                          params.r, theta_val, params.a_bar, y_bar_val, 
                                          params.na, params.amax)
            _, c_pol, a_pol, _ = solve_lifecycle(params_temp)
            c_path, a_path, y_path, s_path = simulate_lifecycle(c_pol, a_pol, a_grid, params_temp)
            
            plot!(p_assets, 1:params.J, a_path, label="y̅ = $y_bar_val", linewidth=2)
            peak_wealth[y_bar_val] = maximum(a_path)
        end
        
        display(p_assets)
        
        println("\nWith θ = $theta_val:")
        for y_bar_val in income_levels
            @printf("  Peak wealth (y̅ = %.1f): %.2f\n", y_bar_val, peak_wealth[y_bar_val])
        end
        
        if haskey(peak_wealth, 2.0) && haskey(peak_wealth, 0.5)
            inequality_ratio = peak_wealth[2.0] / peak_wealth[0.5]
            @printf("  Wealth inequality ratio: %.2f\n", inequality_ratio)
        end
    end
end

main()



"""
Discuss how the bequest motive manifests differently across income levels.

The bequest motive affects savings behavior differently across income groups:
Poor (ȳ=0.5): No change in savings behavior. They spend nearly all wealth by end of life to meet consumption needs.
Middle-income (ȳ=1.0): Modest response to bequest motive. Peak wealth increases 8.9%, showing some concern for leaving bequests.
Rich (ȳ=2.0): Strong response to bequest motive. Peak wealth increases 27.7%, maintaining high assets throughout life.
This shows bequests are a luxury good: only wealthy households with satisfied consumption needs actively save to leave wealth to heirs. Poor households cannot afford to prioritize bequests.
"""

"""
Does the bequest motive amplify or dampen wealth inequality? Why?
The bequest motive amplifies wealth inequality. The wealth inequality ratio increases from 3.94 to 5.03 when we introduce bequest motives, a 27.7% increase.

How does the luxury good nature of bequests contribute to this effect?
Bequests are a luxury good because only the rich care about leaving them. When bequest motives are introduced:
Poor households (ȳ=0.5): peak wealth unchanged (0% increase)
Rich households (ȳ=2.0): peak wealth increases 27.7%
This happens because poor households use all their resources for basic consumption needs throughout life, while rich households have already satisfied consumption needs and can afford to save for bequests. The result is that wealth inequality grows larger across generations.
"""