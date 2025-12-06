using NLsolve, Plots, LinearAlgebra

# Step 1
β = 0.96
α = 0.33
A = 1
δ = 0.1
k̄ = ((1/β - (1-δ))/(α*A))^(1/(α-1))
c̄ = A * k̄^α - δ * k̄

# Step 2
k_0 = 0.5 * k̄
T = 100
γ=2.0
params = (T=T, β=0.96, α=0.33, A=1.0, δ=0.1, γ=γ, k_0=0.5*k̄, c̄=c̄)

function transition_equations(x, params)
    
    T = params.T
    β = params.β
    α = params.α
    A = params.A
    δ = params.δ
    γ = params.γ
    k_0 = params.k_0
    c̄ = params.c̄

    c = @view x[1:T+1]
    k = @view x[T+2:end]

    residuals = zeros(2*T+1)

    for t in 0:(T-1)
        c_t = c[t+1]
        c_tp1 = c[t+2]
        k_tp1 = k[t+1]

        residuals[t+1] = c_t^(-γ) - β * c_tp1^(-γ) * (α * A * (k_tp1)^(α-1) + (1 - δ))
    end
    
    for t in 0:(T-1)
        if t == 0
            k_t = k_0
        else
            k_t = k[t]
        end
        k_tp1 = k[t+1]
        c_t = c[t+1] 

        residuals[T + t + 1] = k_tp1 - ((1-δ)*k_t + A * k_t^α - c_t)
    end

    residuals[2*T+1] = c[T+1] - c̄

    return residuals
end

# step 3
c_guess = fill(c̄, T+1)
k_guess = [k_0 + (t/T) * (k̄ - k_0) for t in 1:T]
x0 = [c_guess; k_guess]

function f!(F, x)
    F[:] = transition_equations(x, params)
end

# γ = 2
result = nlsolve(f!, x0; ftol=1e-8)

println("Solver converged? ", result.f_converged)
println("Final residual norm: ", result.residual_norm)

x_sol = result.zero
k_sol = x_sol[T+2:end]
k_T = k_sol[end]
error = abs(k_T - k̄) / k̄ * 100
println("k_T error: ", round(error, digits=4), "%") # the result is 0.0044%, which is less than 0.1%.

# γ = 0.5
γ_new=0.5
params_new = (T=T, β=0.96, α=0.33, A=1.0, δ=0.1, γ=γ_new, k_0=0.5*k̄, c̄=c̄)

function f_new!(F, x)
    F[:] = transition_equations(x, params_new)
end

result_new = nlsolve(f_new!, x0; ftol=1e-8)

println("Solver converged? ", result_new.f_converged)
println("Final residual norm: ", result_new.residual_norm)

x_sol_new = result_new.zero
k_sol_new = x_sol_new[T+2:end]
k_T_new = k_sol_new[end]
error_new = abs(k_T_new - k̄) / k̄ * 100
println("k_T error: ", round(error_new, digits=4), "%") # the result is less than 0.1%

# step 4
c_path_γ2 = x_sol[1:T+1]          
k_path_γ2 = [k_0; k_sol]                 

c_path_γ05 = x_sol_new[1:T+1]    
k_path_γ05 = [k_0;k_sol_new]          

y_path_γ2 = A .* k_path_γ2.^α
y_path_γ05 = A .* k_path_γ05.^α

ȳ = A * k̄^α

time_capital = 0:T                
time_ratios = 0:T  

# Plot 1: Capital stock k_t
p1 = plot(time_capital, [k_path_γ2 k_path_γ05], 
          label=["γ = 2.0" "γ = 0.5"],
          linewidth=2,
          color=[:blue :red],
          xlabel="Time (t)",
          ylabel="Capital stock (kₜ)",
          title="Capital Stock Transition",
          legend=:bottomright,
          grid=true)

hline!([k̄], label="Steady state k̄", 
       linestyle=:dash, linecolor=:black, linewidth=1.5)

# Plot 2: Consumption rate c_t/y_t
c_rate_γ2 = c_path_γ2 ./ y_path_γ2
c_rate_γ05 = c_path_γ05 ./ y_path_γ05

c̄_rate = c̄ / ȳ

p2 = plot(time_ratios, [c_rate_γ2 c_rate_γ05],
          label=["γ = 2.0" "γ = 0.5"],
          linewidth=2,
          color=[:blue :red],
          xlabel="Time (t)",
          ylabel="Consumption rate (cₜ/yₜ)",
          title="Consumption Rate",
          legend=:bottomright,
          grid=true)

hline!([c̄_rate], label="Steady state c̄/ȳ", 
       linestyle=:dash, linecolor=:black, linewidth=1.5)

# Plot 3: Investment rate i_t/y_t
i_path_γ2 = y_path_γ2 .- c_path_γ2
i_path_γ05 = y_path_γ05 .- c_path_γ05

i_rate_γ2 = i_path_γ2 ./ y_path_γ2
i_rate_γ05 = i_path_γ05 ./ y_path_γ05

ī_rate = δ * k̄ / ȳ

p3 = plot(time_ratios, [i_rate_γ2 i_rate_γ05],
          label=["γ = 2.0" "γ = 0.5"],
          linewidth=2,
          color=[:blue :red],
          xlabel="Time (t)",
          ylabel="Investment rate (iₜ/yₜ)",
          title="Investment Rate",
          legend=:bottomright,
          grid=true)

hline!([ī_rate], label="Steady state ī/ȳ", 
       linestyle=:dash, linecolor=:black, linewidth=1.5)

plot(p1, p2, p3, layout=(3,1), size=(800, 900))

"""
High IES (γ=0.5): fast convergence, volatile consumption, aggressive investment response.
-Fast convergence: They save a lot when returns are high (low capital), so capital grows quickly toward steady state.
-Volatile consumption: They accept big changes in consumption over time to take advantage of investment opportunities.
-Aggressive investment: Initial investment is high but drops rapidly as capital accumulates and returns fall.

Low IES (γ=2.0): slow convergence, smooth consumption, gradual investment adjustment.
-Slow convergence: Households won't reduce consumption much even when investment returns are high due to consumption smooth preference.
-Smooth consumption: They avoid big changes in consumption, keeping it stable across periods.
-Gradual investment adjustment: Investment changes modestly to maintain stable consumption path.
"""