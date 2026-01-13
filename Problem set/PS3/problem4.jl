using QuantEcon, LinearAlgebra, Plots, Random, StatsBase

α, f_bar = 0.1, 1.2
δ = 0.1
γ = 1.5
β = 0.95
ρ = 0.98
σ_ϵ = 0.15
q = 0.5

# Funkcja produkcji f(h)
f(h) = min(h^α + 0.1, f_bar)

h_grid = collect(0:0.05:25.0)
e_grid = collect(0:0.01:1.0)
nh, ne = length(h_grid), length(e_grid)

#Rouwenhorst
nw = 7
mc = rouwenhorst(nw, ρ, σ_ϵ)
w_grid = exp.(mc.state_values)
P = mc.p

#VFI
V = zeros(nh, nw)
V_new = similar(V)
policy_idx = zeros(Int, nh, nw)

tol = 1e-6
max_iter = 1500

for it in 1:max_iter
    V_new .= -Inf
    for j in 1:nw, i in 1:nh
        h, w = h_grid[i], w_grid[j]
        best_val = -Inf
        
        for (k, e) in enumerate(e_grid)
            c = w * f(h) * (1 - e)
            if c <= 0 continue end
            
            u = (c^(1-γ))/(1-γ) - q*e
            
            h_next = clamp((1 - δ) * h + e, h_grid[1], h_grid[end])
            idx_next = clamp(searchsortedfirst(h_grid, h_next), 1, nh)

            ev = dot(P[j, :], V[idx_next, :])
            
            val = u + β * ev
            if val > best_val
                best_val = val
                policy_idx[i, j] = k
            end
        end
        V_new[i, j] = best_val
    end
    
    if maximum(abs.(V_new - V)) < tol
        println("Zbieżność po $it iteracjach")
        break
    end
    V .= V_new
end

#Symulacja
Random.seed!(123)
T = 1000
burn_in = 100
n_paths = 5

h_sim = zeros(T, n_paths)
e_sim = zeros(T, n_paths)
y_sim = zeros(T, n_paths)
w_sim = zeros(T, n_paths)

for p in 1:n_paths
    h_curr = 1.0
    w_idx = 4
    
    for t in 1:T
        i_h = clamp(searchsortedfirst(h_grid, h_curr), 1, nh)
        k_e = policy_idx[i_h, w_idx]
        e_curr = e_grid[k_e]
        
        y_curr = w_grid[w_idx] * f(h_curr) * (1 - e_curr)
        
        h_sim[t, p] = h_curr
        e_sim[t, p] = e_curr
        y_sim[t, p] = y_curr
        w_sim[t, p] = w_grid[w_idx]
        
        h_curr = clamp((1 - δ) * h_curr + e_curr, h_grid[1], h_grid[end])
        w_idx = sample(1:nw, Weights(P[w_idx, :]))
    end
end

h_final = h_sim[burn_in+1:end, :]
e_final = e_sim[burn_in+1:end, :]
y_final = y_sim[burn_in+1:end, :]
w_final = w_sim[burn_in+1:end, :]


w_indices = [1, 4, 7]
labels = ["Low Wage (w1)", "Medium Wage (w4)", "High Wage (w7)"]

c_star = zeros(nh, nw)
for j in 1:nw, i in 1:nh
    e_opt = e_grid[policy_idx[i, j]]
    c_star[i, j] = w_grid[j] * f(h_grid[i]) * (1 - e_opt)
end

p4a = plot(title="Optimal Education Choice e*(h, w)", xlabel="Human Capital h", ylabel="e")
for (idx, label) in zip(w_indices, labels)
    plot!(p4a, h_grid, e_grid[policy_idx[:, idx]], label=label, lw=2)
end

p4b = plot(title="Value Function V(h, w)", xlabel="Human Capital h", ylabel="V")
for (idx, label) in zip(w_indices, labels)
    plot!(p4b, h_grid, V[:, idx], label=label, lw=2)
end

p4c = plot(title="Optimal Consumption c*(h, w)", xlabel="Human Capital h", ylabel="c")
for (idx, label) in zip(w_indices, labels)
    plot!(p4c, h_grid, c_star[:, idx], label=label, lw=2)
end

plot(p4a, p4b, p4c, layout=(3,1), size=(800, 1000))


t_range = 1:100 

p5_h = plot(h_final[t_range, :], title="(i) Human Capital h_t", ylabel="h", label=false)
p5_w = plot(w_final[t_range, :], title="(ii) Wage Shock w_t", ylabel="w", label=false)
p5_e = plot(e_final[t_range, :], title="(iii) Education e_t", ylabel="e", label=false)

p5_c = plot(y_final[t_range, :], title="(iv) Consumption c_t", ylabel="c", label=false)

plot(p5_h, p5_w, p5_e, p5_c, layout=(2,2), size=(900, 700), 
     plot_title="Evolution of 5 paths (periods 101-200)")
